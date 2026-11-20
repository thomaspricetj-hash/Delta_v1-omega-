"""
Production MBFN inference wrapper for WeaveFormer
"""
from pathlib import Path
import torch
from infrastructure.true_hybrid.weaveformer import WeaveFormer
from infrastructure.tokenizer.bpe_tokenizer import get_bpe_tokenizer
from infrastructure.hscl.hscl_preprocessor import compress as hscl_compress
from infrastructure.mbfn.store import FrameStore
from infrastructure.mbfn.bridge import MBFNBridge

class MBFNGenerator:
    def __init__(self, model_path: Path, hscl_dict_path: Path = None, alpha: float = 0.7):
        self.tok = get_bpe_tokenizer()
        vocab = getattr(self.tok, 'vocab', {})
        vocab_size = min(len(vocab), 8192)
        self.model = WeaveFormer(dim=256, num_layers=36, vocab_size=vocab_size)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.store = FrameStore()
        if hscl_dict_path and Path(hscl_dict_path).exists():
            self.store.build_from_hscl(Path(hscl_dict_path))
        self.bridge = MBFNBridge(self.store, alpha=alpha)

    def _activate_from_prompt(self, prompt: str):
        # HSCL compress then keyword activation
        comp = hscl_compress(prompt)
        # simple trigger activation
        words = set(comp.lower().split())
        triggers = []
        for label in self.store.label_index.keys():
            if any(w in label for w in words):
                triggers.append((label, 1.0))
        self.store.activate_frames(triggers)

    def generate(self, prompt: str, max_new_tokens: int = 64, top_k: int = 10) -> str:
        self._activate_from_prompt(prompt)
        # tokenize with HSCL
        comp_prompt = hscl_compress(prompt)
        ids = self.tok.encode(comp_prompt)
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        # simple greedy generation with MBFN re-ranking every step
        output_ids = list(ids)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(input_ids)
                # get top-k candidates
                last_logits = logits[0, -1]
                top_vals, top_idx = torch.topk(last_logits, k=top_k)
                # decode candidates
                candidates = [self.tok.decode([int(i)]) for i in top_idx.tolist()]
                # map to scores
                scores = top_vals.tolist()
                # MBFN re-rank
                ranked = self.bridge.rank_candidates(candidates, scores)
                # pick top
                chosen_text = ranked[0][0]
                # tokenize chosen text and append
                chosen_ids = self.tok.encode(chosen_text)
                if not chosen_ids:
                    break
                output_ids.extend(chosen_ids)
                # update input
                input_ids = torch.tensor(output_ids, dtype=torch.long).unsqueeze(0)
                # stop on eos if present
                if chosen_ids and chosen_ids[0] == 0:
                    break
        return self.tok.decode(output_ids)

    def explain(self, prompt: str, choice: str):
        self._activate_from_prompt(prompt)
        return self.bridge.explain_choice(choice)
