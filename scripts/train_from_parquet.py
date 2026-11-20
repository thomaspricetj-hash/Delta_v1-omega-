#!/usr/bin/env python3
import argparse
import sys
import re
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from infrastructure.true_hybrid.math_aware_weaveformer import MathAwareWeaveFormer

def load_parquet_text(data_dir):
    p = Path(data_dir)
    if not p.exists() or not list(p.glob('train_*.parquet')):
        fallback = Path(__file__).resolve().parents[2] / 'training data'
        if fallback.exists():
            p = fallback
        else:
            alt = Path(__file__).resolve().parents[1] / 'trainingData'
            if alt.exists():
                p = alt
    files = list(p.glob('train_*.parquet'))
    if not files:
        return []
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    for col in ['text','content','prompt','response']:
        if col in df.columns:
            return df[col].fillna('').astype(str).tolist()
    for col in df.columns:
        if df[col].dtype == object:
            return df[col].fillna('').astype(str).tolist()
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    texts = load_parquet_text(args.data)
    if not texts:
        print('No texts found')
        return

    # Mix curriculum math
    curriculum_path = Path(__file__).resolve().parents[1] / 'runtime' / 'logs' / 'math_curriculum.jsonl'
    curriculum_texts = []
    curriculum_types = []
    if curriculum_path.exists():
        with open(curriculum_path,'r',encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    prompt = obj.get('prompt') or obj.get('question','')
                    ans = obj.get('answer','')
                    typ = obj.get('type','forward')
                    curriculum_texts.append(f"{prompt} {ans}")
                    curriculum_types.append(typ)
                except Exception:
                    pass
    if curriculum_texts:
        import random
        forward_idx = [i for i,t in enumerate(curriculum_types) if t=='forward']
        reverse_idx = [i for i,t in enumerate(curriculum_types) if t=='reverse']
        random.shuffle(forward_idx); random.shuffle(reverse_idx)
        n_cur = min(len(texts), len(forward_idx)+len(reverse_idx))
        n_fwd = min(len(forward_idx), n_cur//2)
        n_rev = min(len(reverse_idx), n_cur - n_fwd)
        sel = forward_idx[:n_fwd] + reverse_idx[:n_rev]
        random.shuffle(sel)
        texts = texts + [curriculum_texts[i] for i in sel]
        random.shuffle(texts)

    def clean_text(s):
        s = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()
    texts = [clean_text(t) for t in texts]

    seq_len = 32
    vocab_size = 512
    from infrastructure.tokenizer.bpe_tokenizer import get_bpe_tokenizer
    tok = get_bpe_tokenizer()
    ids_list = []
    for t in texts:
        try:
            toks = tok.encode(t)[:seq_len]
        except ValueError:
            t_clean = re.sub(r'[^\x20-\x7E]', ' ', t)
            toks = tok.encode(t_clean)[:seq_len]
        toks = [min(int(x), vocab_size-1) for x in toks]
        if len(toks) < seq_len:
            toks = toks + [0]*(seq_len-len(toks))
        ids_list.append(toks[:seq_len])
    ids_tensor = torch.tensor(ids_list, dtype=torch.long)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, ids, texts):
            self.ids = ids
            self.texts = texts
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return self.ids[i], self.texts[i]

    dataset = TextDataset(ids_tensor, texts)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MathAwareWeaveFormer(dim=256, num_layers=2, vocab_size=512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2)
    ce_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    ce_loss_raw_fn = torch.nn.CrossEntropyLoss()

    try:
        from infrastructure.music_mapper.music_mapper import log_note
        _music_log = log_note
    except Exception:
        _music_log = None

    lambda_distill_base = 0.02
    temp = 2.0
    step = 0
    for epoch in range(args.epochs):
        for ids_batch, texts_batch in loader:
            step += 1
            if args.steps and step > args.steps:
                break
            lambda_distill = lambda_distill_base * min(1.0, step/200.0)
            targets_batch = torch.roll(ids_batch, shifts=-1, dims=1)
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(ids_batch, texts_batch)
                ce_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), targets_batch.view(-1))
                ce_loss_raw = ce_loss_raw_fn(logits.view(-1, logits.size(-1)), targets_batch.view(-1))
            n_rev = sum(1 for t in texts_batch if 'The answer is' in t)
            n_fwd = len(texts_batch) - n_rev
            distill_loss = torch.tensor(0.0, device=logits.device)
            count = 0
            probs = torch.softmax(logits / temp, dim=-1)
            for i, txt in enumerate(texts_batch):
                parsed = model._parse_math(txt)
                if parsed:
                    val, _, _, _ = parsed
                    token_id = model._result_token_id(val)
                    if token_id is not None:
                        model_prob = probs[i, -1]
                        target = torch.zeros_like(model_prob)
                        target[token_id] = 1.0
                        kl = torch.nn.functional.kl_div(torch.log(model_prob + 1e-9), target, reduction='batchmean') * (temp**2)
                        distill_loss += kl
                        count += 1
            if count > 0:
                distill_loss = distill_loss / count
            loss = ce_loss + lambda_distill * distill_loss

            warmup_steps = 50
            if step <= warmup_steps:
                warmup_factor = step / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = 1e-3 * warmup_factor
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            if _music_log is not None:
                try:
                    conf = max(0.1, min(1.0, 1.0 - float(loss.detach().cpu().item())/5.0))
                    _music_log(query=f"train_step_{step}", response=f"loss_{loss.item():.4f}", confidence=conf, source="training", weight=conf, latency_ms=None, request_id=f"train_{step}")
                except Exception:
                    pass

            if step % 10 == 0:
                blend_val = getattr(model, 'math_blend', 0.3)
                print(f'Step {step} loss {loss:.4f} ce_sm {ce_loss.item():.4f} ce_raw {ce_loss_raw.item():.4f} distill {distill_loss.item():.4f} fwd {n_fwd} rev {n_rev} blend {blend_val:.3f}')
                try:
                    log_path = Path(__file__).resolve().parents[1] / 'runtime' / 'logs' / 'train_log.jsonl'
                    with open(log_path,'a',encoding='utf-8') as f:
                        f.write(json.dumps({'ts':time.time(),'step':step,'loss':float(loss.item()),'ce_raw':float(ce_loss_raw.item()),'distill':float(distill_loss.item()),'math_blend':float(blend_val)})+'\n')
                except Exception:
                    pass
        if args.steps and step >= args.steps:
            break
    print('Done')

if __name__ == '__main__':
    main()
