"""
Production HSCL preprocessor
Supports abbreviation mode and numeric concept ID mode
"""
import json
import re
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent
DICT_PATH = BASE_DIR / "hscl_dictionary_v1.json"
NUMERIC_DICT_PATH = BASE_DIR / "hscl_numeric_dict.json"

class HSCLCompressor:
    def __init__(self, mode="abbr"):
        self.mode = mode
        self.patterns = []
        self.id_map = {}
        self.phrase_to_id = {}
        self.id_to_phrase = {}
        if mode == "abbr":
            self._load_abbr()
        elif mode == "numeric":
            self._load_numeric()
        else:
            raise ValueError("mode must be abbr or numeric")

    def _load_abbr(self):
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        patterns = []
        for long_phrase, short in mapping.items():
            pattern = re.compile(r'\b' + re.escape(long_phrase) + r'\b', flags=re.IGNORECASE)
            patterns.append((pattern, long_phrase, short))
        patterns.sort(key=lambda x: len(x[1]), reverse=True)
        self.patterns = patterns

    def _load_numeric(self):
        if not NUMERIC_DICT_PATH.exists():
            # fallback to empty
            self.patterns = []
            return
        with open(NUMERIC_DICT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        entries = data.get("entries", {})
        self.phrase_to_id = {p: v["id"] for p,v in entries.items()}
        self.id_to_phrase = {v["id"]: p for p,v in entries.items()}
        # build regex patterns
        patterns = []
        for phrase, cid in self.phrase_to_id.items():
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', flags=re.IGNORECASE)
            patterns.append((pattern, phrase, cid))
        patterns.sort(key=lambda x: len(x[1]), reverse=True)
        self.patterns = patterns

    def compress(self, text: str) -> str:
        if self.mode == "abbr":
            result = text
            for pattern, long_phrase, short in self.patterns:
                def repl(m):
                    match_text = m.group(0)
                    if match_text.isupper():
                        return short.upper()
                    elif match_text[0].isupper():
                        return short.capitalize()
                    else:
                        return short
                result = pattern.sub(repl, result)
            return result
        else:
            # numeric mode: replace phrase with <CID> token placeholder
            result = text
            for pattern, phrase, cid in self.patterns:
                # Replace with a special token that tokenizer can map to id
                # Use a placeholder that survives BPE: <HSCL_12345>
                placeholder = f"<HSCL_{cid}>"
                result = pattern.sub(lambda m: placeholder, result)
            return result

    def decompress(self, text: str) -> str:
        if self.mode != "numeric":
            return text
        # replace placeholders back
        def repl(m):
            cid = int(m.group(1))
            phrase = self.id_to_phrase.get(cid, f"<HSCL_{cid}>")
            return phrase
        return re.sub(r"<HSCL_(\d+)>", repl, text)

# singleton for easy import
_compressor = HSCLCompressor(mode="abbr")

def compress(text: str) -> str:
    return _compressor.compress(text)

def set_mode(mode: str):
    global _compressor
    _compressor = HSCLCompressor(mode=mode)
