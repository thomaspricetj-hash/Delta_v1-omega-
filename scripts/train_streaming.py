#!/usr/bin/env python3
"""
Streaming trainer for Translation OS
Tokenizes on the fly, never materializes whole dataset.
"""
import argparse, sys, re, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
from infrastructure.true_hybrid.math_aware_weaveformer import MathAwareWeaveFormer

def load_parquet_paths(data_dir):
    p = Path(data_dir)
    if not p.exists() or not list(p.glob('train_*.parquet')):
        p = Path(__file__).resolve().parents[2] / 'training data'
    files = sorted(p.glob('train_*.parquet'))
    return files

def iter_texts(files):
    for f in files:
        try:
            df = pd.read_parquet(f)
            for col in ['text','content','prompt','response']:
                if col in df.columns:
                    for v in df[col].fillna('').astype(str):
                        yield v
                    return
            for col in df.columns:
                if df[col].dtype == object:
                    for v in df[col].fillna('').astype(str):
                        yield v
                    return
        except Exception:
            continue

class StreamingTextDataset(IterableDataset):
    def __init__(self, files, seq_len=32, vocab_size=512, batch_size=4):
        self.files = files
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        from infrastructure.tokenizer.bpe_tokenizer import get_bpe_tokenizer
        self.tok = get_bpe_tokenizer()

    def __iter__(self):
        batch_ids = []
        batch_texts = []
        for raw in iter_texts(self.files):
            s = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', raw)
            s = re.sub(r'\s+', ' ', s).strip()
            try:
                toks = self.tok.encode(s)[:self.seq_len]
            except Exception:
                toks = self.tok.encode(re.sub(r'[^\x20-\x7E]', ' ', s))[:self.seq_len]
            toks = [min(int(x), self.vocab_size-1) for x in toks]
            if len(toks) < self.seq_len:
                toks = toks + [0]*(self.seq_len-len(toks))
            batch_ids.append(toks)
            batch_texts.append(s)
            if len(batch_ids) >= self.batch_size:
                ids = torch.tensor(batch_ids, dtype=torch.long)
                yield ids, batch_texts
                batch_ids, batch_texts = [], []
        if batch_ids:
            ids = torch.tensor(batch_ids, dtype=torch.long)
            yield ids, batch_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    files = load_parquet_paths(args.data)
    dataset = StreamingTextDataset(files, batch_size=args.batch_size)
    loader = DataLoader(dataset, batch_size=None)

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
            lambda_distill = lambda_distill_base * min(1.0, step/200.0)
            targets = torch.roll(ids_batch, shifts=-1, dims=1)
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(ids_batch, texts_batch)
                ce_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                ce_loss_raw = ce_loss_raw_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            # distillation
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

            if step <= 50:
                warmup = step/50
                for pg in optimizer.param_groups:
                    pg['lr'] = 1e-3 * warmup
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            if _music_log:
                try:
                    conf = max(0.1, min(1.0, 1.0 - float(loss.item())/5.0))
                    _music_log(query=f"train_step_{step}", response=f"loss_{loss.item():.4f}", confidence=conf, source="training", weight=conf, latency_ms=None, request_id=f"train_{step}")
                except Exception:
                    pass

            if step % 10 == 0:
                blend = getattr(model, 'math_blend', 0.3)
                print(f'Step {step} loss {loss:.4f} ce {ce_loss.item():.4f} distill {distill_loss.item():.4f} blend {blend:.3f}')
    print('Done')

if __name__ == '__main__':
    main()
