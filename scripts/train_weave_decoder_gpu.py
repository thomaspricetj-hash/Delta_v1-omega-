"""
GPU-first WeaveFormer trainer with fixed-size tree ledger, Schrödinger annealing,
resonance drafting, and unthrottled CPU pre-processing.
"""
import os
# Unthrottle CPU for data pre-processing / tokenization
os.environ['OMP_NUM_THREADS']='24'
os.environ['MKL_NUM_THREADS']='24'
os.environ['NUMEXPR_NUM_THREADS']='24'
os.environ['CUDA_LAUNCH_BLOCKING']='0'

import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import torch
# Use 24 logical threads for CPU pre-processing
torch.set_num_threads(24)
try:
    torch.set_num_interop_threads(12)
except Exception:
    pass

print(f"PyTorch threads: {torch.get_num_threads()} interop: {torch.get_num_interop_threads()}")
assert torch.cuda.is_available(), "CUDA not available"
device = torch.device('cuda')
print(f"Using device: {device}")
# Performance tuning
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Module-level shutdown flag for signal handling
_shutdown_requested = False

import random, math, copy, subprocess
from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer
from infrastructure.true_hybrid.weaveformer import WeaveFormer
from infrastructure.true_hybrid.dual_weight_layers import DualWeightWeaveFormer
from infrastructure.true_hybrid.codebook_checkpoint import save_model_codebook, load_model_codebook
from infrastructure.true_hybrid.tree_store import TreeStore, TreeNode
from torch.utils.data import DataLoader
from infrastructure.data.codebook_dataset import CodebookTextDataset, collate_codebook_batch
import torch.nn as nn
from infrastructure.optimization.resonance_drafting import ResonantDraftOptimizer
from infrastructure.memory.ai_memory_orchestrator import AIMemoryOrchestrator
# Orchestrator removed from training per user request; only Memory Orchestrator is used
ORCHESTRATOR_ENABLED = False
ORCHESTRATOR = None
# Emergence map hook – read last emergence score from logs, no orchestrator call
EMERGENCE_LOG = BASE / "runtime" / "logs" / "emergence_score_history.jsonl"
EMERGENCE_AVAILABLE = EMERGENCE_LOG.exists()

# Schrödinger quantum annealing helpers
try:
    from infrastructure.quantum.schrodinger_sampler import schrodinger_anneal_schedule, quantum_rotation_prob
    QUANTUM = True
except Exception:
    QUANTUM = False
    schrodinger_anneal_schedule = None
    quantum_rotation_prob = None

# Convergence map & heatmap cold overlay for reward weighting
try:
    from infrastructure.convergence_map.convergence_map import load_map as cm_load_map, add_reward_signal as cm_add_reward, get_hot_terms as cm_get_hot_terms
    CONVERGENCE_AVAILABLE = True
except Exception:
    CONVERGENCE_AVAILABLE = False
    cm_load_map = None
    cm_add_reward = None
    cm_get_hot_terms = None

# Pattern mapper for learning from correct patterns - layered tree with fixed size nodes
try:
    from infrastructure.true_hybrid.layered_pattern_tree import LayeredPatternTree
    PATTERN_TREE_AVAILABLE = True
except Exception:
    PATTERN_TREE_AVAILABLE = False
    LayeredPatternTree = None

# Matrix calc engine with CPU offload and Bloom-filtered cache for pattern replay preprocessing
class MatrixCalcEngine:
    def __init__(self, dim=288, cache_size=65536):
        self.dim = dim
        self.cache = {}
        self.bloom = set()
        self.cache_size = cache_size
    def _key(self, a, b):
        return (tuple(a.tolist()), tuple(b.tolist()))
    def matmul_cpu(self, a, b):
        # Simple CPU low-rank matmul with caching
        key = self._key(a, b)
        if key in self.cache:
            return self.cache[key]
        # Move to CPU for offload
        a_cpu = a.detach().cpu()
        b_cpu = b.detach().cpu()
        out = torch.matmul(a_cpu, b_cpu)
        if len(self.cache) >= self.cache_size:
            # evict oldest
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = out
        return out.to(a.device)
    def maybe_cache_pattern(self, pattern_bytes):
        # Bloom-like quick check
        h = hash(pattern_bytes)
        if h in self.bloom:
            return True
        if len(self.bloom) > self.cache_size:
            self.bloom.clear()
        self.bloom.add(h)
        return False

def load_texts(data_dir):
    import pandas as pd, glob, pickle, pyarrow.parquet as pq, json
    cache_dir = BASE / "runtime" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_idx = int(os.environ.get('CHUNK_INDEX','0'))
    files_per_chunk = int(os.environ.get('FILES_PER_CHUNK','0')) or 0
    overlap_texts = int(os.environ.get('OVERLAP_TEXTS','0'))
    layer_depth = int(os.environ.get('LAYER_DEPTH','1'))
    micro_chunk_rows = int(os.environ.get('MICRO_CHUNK_ROWS','20000'))
    cache_part_size = int(os.environ.get('CACHE_PART_SIZE','10000'))
    # Compute effective chunk size with layer depth
    if layer_depth > 1 and files_per_chunk > 0:
        effective_files_per_chunk = files_per_chunk * layer_depth
    else:
        effective_files_per_chunk = files_per_chunk
    cache_manifest = cache_dir / f"texts_chunk_{chunk_idx}.manifest"
    cache_path = cache_dir / f"texts_chunk_{chunk_idx}.pkl"
    use_cache = os.environ.get('USE_TEXT_CACHE','1') == '1'
    # Try layered chunked cache load
    if use_cache and cache_manifest.exists():
        try:
            with open(cache_manifest, 'r', encoding='utf-8') as mf:
                manifest = json.load(mf)
            parts = manifest.get('parts', [])
            total = manifest.get('total', 0)
            max_texts = int(os.environ.get('MAX_TEXTS','0')) or None
            need = max_texts if max_texts else total
            texts = []
            loaded = 0
            for part_file in parts:
                part_path = cache_dir / part_file
                if not part_path.exists():
                    continue
                with open(part_path, 'rb') as pf:
                    part_texts = pickle.load(pf)
                texts.extend(part_texts)
                loaded += len(part_texts)
                if max_texts and loaded >= need:
                    texts = texts[:need]
                    break
            print(f"[Cache] Loaded {len(texts)} texts from {len(parts)} parts, total≈{total}")
            # Layered overlap
            if overlap_texts > 0 and chunk_idx > 0:
                prev_manifest = cache_dir / f"texts_chunk_{chunk_idx-1}.manifest"
                if prev_manifest.exists():
                    try:
                        with open(prev_manifest, 'r', encoding='utf-8') as pmf:
                            prev_man = json.load(pmf)
                        prev_parts = prev_man.get('parts', [])
                        # Load tail from last part
                        overlap = []
                        needed = overlap_texts
                        for pfname in reversed(prev_parts):
                            pp = cache_dir / pfname
                            if not pp.exists():
                                continue
                            with open(pp, 'rb') as ppf:
                                ptexts = pickle.load(ppf)
                            # take from end
                            take = ptexts[-needed:]
                            overlap = list(take) + overlap
                            needed -= len(take)
                            if needed <= 0:
                                break
                        if overlap:
                            texts = overlap + texts
                            print(f"[Layer] Added {len(overlap)} overlap texts from previous chunk")
                    except Exception:
                        pass
            return texts
        except Exception as e:
            print(f"[Cache] Manifest load failed: {e}, falling back to parquet")
    # Fallback to old pickle cache
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                texts = pickle.load(f)
            print(f"[Cache] Loaded {len(texts)} texts from {cache_path}")
            if overlap_texts > 0 and chunk_idx > 0:
                prev_path = cache_dir / f"texts_chunk_{chunk_idx-1}.pkl"
                if prev_path.exists():
                    try:
                        with open(prev_path,'rb') as pf:
                            prev_texts = pickle.load(pf)
                        overlap = prev_texts[-overlap_texts:] if overlap_texts < len(prev_texts) else prev_texts
                        texts = overlap + texts
                        print(f"[Layer] Added {len(overlap)} overlap texts from previous chunk")
                    except Exception:
                        pass
            max_texts = int(os.environ.get('MAX_TEXTS','0')) or None
            if max_texts and len(texts) > max_texts:
                texts = texts[:max_texts]
            return texts
        except Exception as e:
            print(f"[Cache] Load failed: {e}, falling back to parquet")
    # Gather parquet files
    all_files = sorted(glob.glob(str(Path(data_dir)/"*.parquet")))
    if files_per_chunk > 0:
        start = chunk_idx * files_per_chunk
        window = effective_files_per_chunk if effective_files_per_chunk>0 else files_per_chunk
        end = start + window
        files = all_files[start:end]
        print(f"[Chunk] Chunk {chunk_idx} using files {start}..{end-1} -> {len(files)} files (layer_depth={layer_depth})")
    else:
        files = all_files
        print(f"[Chunk] No chunking, using all {len(files)} files")
    texts = []
    max_texts = int(os.environ.get('MAX_TEXTS','0')) or None
    count = 0
    # Helper to filter out metadata noise
    def is_valid_text(s):
        if len(s) < 30:
            return False
        # Skip URLs, s3 paths, UUIDs, ISO timestamps, pure numbers
        s_low = s.lower()
        if s_low.startswith('http://') or s_low.startswith('https://') or s_low.startswith('s3://'):
            return False
        if s_low.startswith('<urn:uuid:') or 'urn:uuid:' in s_low:
            return False
        # ISO timestamp like 2025-06-23T22:40:14Z
        import re
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?', s.strip()):
            return False
        # Single word very short
        words = s.split()
        if len(words) == 1 and len(s) > 200:
            # likely a long URL or blob
            return False
        # Require at least 3 words
        if len(words) < 3:
            return False
        return True
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            # Stream row groups in micro chunks
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                for col in table.column_names:
                    col_data = table.column(col)
                    for batch in col_data.chunks:
                        values = batch.to_pylist()
                        for v in values:
                            if v is None:
                                continue
                            s = str(v)
                            if not s:
                                continue
                            if not is_valid_text(s):
                                continue
                            texts.append(s)
                            count += 1
                            if max_texts and count >= max_texts:
                                break
                        if max_texts and count >= max_texts:
                            break
                    if max_texts and count >= max_texts:
                        break
                if max_texts and count >= max_texts:
                    break
        except Exception as e:
            print(f"[Parquet] Error reading {f}: {e}")
        if max_texts and count >= max_texts:
            break
    if use_cache:
        try:
            # Save as chunked parts for fast partial loads
            part_size = cache_part_size
            parts = []
            total = len(texts)
            for i in range(0, total, part_size):
                part_texts = texts[i:i+part_size]
                part_name = f"texts_chunk_{chunk_idx}_part_{i//part_size}.pkl"
                part_path = cache_dir / part_name
                with open(part_path, 'wb') as pf:
                    pickle.dump(part_texts, pf)
                parts.append(part_name)
            manifest = {
                'chunk_idx': chunk_idx,
                'part_size': part_size,
                'total': total,
                'parts': parts
            }
            with open(cache_manifest, 'w', encoding='utf-8') as mf:
                json.dump(manifest, mf)
            # Also keep legacy pickle for backward compat
            with open(cache_path, 'wb') as f:
                pickle.dump(texts, f)
            print(f"[Cache] Saved {total} texts to {len(parts)} parts + manifest")
        except Exception as e:
            print(f"[Cache] Save failed: {e}")
    return texts

def clean_text(s):
    return ''.join(ch if ord(ch)<128 else ' ' for ch in s)

def apply_risk_adjustment(model, grad_ema_mean, grad_ema_var, beta=0.99, eps=1e-8):
    # Per-parameter risk-adjusted scaling: pi = delta / sigma
    for p in model.parameters():
        if p.grad is None:
            continue
        pid = id(p)
        g = p.grad.detach()
        # Update EMA of mean and variance
        if pid not in grad_ema_mean:
            grad_ema_mean[pid] = g.clone().detach()
            grad_ema_var[pid] = torch.zeros_like(g)
        else:
            grad_ema_mean[pid].lerp_(g, 1-beta)
            # EMA of variance
            diff = g - grad_ema_mean[pid]
            grad_ema_var[pid].lerp_(diff*diff, 1-beta)
        mean = grad_ema_mean[pid]
        var = grad_ema_var[pid]
        sigma = torch.sqrt(var + eps)
        # delta = mean, sigma = std
        sharpe = mean / (sigma + eps)
        # Soft scaling 0.5..1.5
        scale = torch.sigmoid(sharpe * 0.5) * 0.5 + 0.5
        # Clip extreme scaling
        scale = torch.clamp(scale, 0.3, 1.5)
        p.grad.data.mul_(scale)

def main():
    print('main start')
    global torch
    # Optional Perl live monitor
    if os.environ.get('USE_PERL_MONITOR','0') == '1':
        try:
            log_path = str(BASE / 'runtime' / 'logs' / 'train.log')
            perl_script = str(BASE / 'tools' / 'monitor_training.pl')
            subprocess.Popen(['perl', perl_script, log_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f'[Perl Monitor] Started monitor on {log_path}')
        except Exception as e:
            print(f'[Perl Monitor] Failed to start: {e}')
    # Enable anomaly detection for NaN debugging if requested
    if os.environ.get('DEBUG_NAN','0') == '1':
        torch.autograd.set_detect_anomaly(True)
        print('[DEBUG] torch.autograd.set_detect_anomaly(True) enabled')
    # Config
    DIM = int(os.environ.get('DIM','288'))
    HEADS = int(os.environ.get('HEADS','9'))
    TRAIN_STEPS = int(os.environ.get('TRAIN_STEPS','10000000'))
    SAVE_EVERY = int(os.environ.get('SAVE_EVERY','20000'))
    SFT_MODE = os.environ.get('SFT_MODE','0') == '1'
    SFT_DATA_PATH = os.environ.get('SFT_DATA_PATH','')
    FAST_MODE = os.environ.get('FAST_MODE','0') == '1'
    # Riemann-inspired annealing helpers
    def _zeta_approx(s, terms=2000):
        # Simple real zeta approximation for s>1
        if s <= 1.0:
            s = 1.001
        total = 0.0
        for n in range(1, terms+1):
            total += 1.0 / (n ** s)
        return total
    def _riemann_lr_factor(step, total_steps):
        # Map step to critical line-like oscillation
        t = 0.5 + 0.5 * math.sin(step * 0.0003)
        s = 1.01 + 0.4 * t
        z = _zeta_approx(s, terms=500)
        # Normalize to ~1.0
        z_norm = z / _zeta_approx(2.0, terms=500)
        return 0.8 + 0.4 * (1.0 / max(z_norm, 1e-6))
    OVERFIT_GUARD = True
    TRUST_REGION_CHECK_EVERY = int(os.environ.get('TRUST_REGION_CHECK_EVERY','500'))
    TRUST_REGION_TOL = float(os.environ.get('TRUST_REGION_TOL','1.02'))
    VAL_SAMPLES = int(os.environ.get('VAL_SAMPLES','1000'))
    ETHICAL_DATA_PATH = os.environ.get('ETHICAL_DATA_PATH','')
    ETHICAL_EVERY = int(os.environ.get('ETHICAL_EVERY','5000'))
    SELF_TALK_EVERY = int(os.environ.get('SELF_TALK_EVERY','1000'))
    DISABLE_TALK_CHECK = os.environ.get('DISABLE_TALK_CHECK','0') == '1'
    SELF_TALK_PROMPT = os.environ.get('SELF_TALK_PROMPT','The')
    TALK_GREEDY = os.environ.get('TALK_GREEDY','0') == '1'
    SFT_WARMUP_STEPS = int(os.environ.get('SFT_WARMUP_STEPS','50000'))
    ROT_PROB = float(os.environ.get('ROT_PROB','0.62'))
    WARMUP_STEPS = int(os.environ.get('WARMUP_STEPS','500'))
    INITIAL_SEQ = int(os.environ.get('INITIAL_SEQ','512'))
    INITIAL_BS = int(os.environ.get('INITIAL_BS','4'))
    BASE_LR = float(os.environ.get('BASE_LR','3e-4'))
    MAX_GRAD_NORM = float(os.environ.get('MAX_GRAD_NORM','1.0'))
    HELPER_WARMUP_STEPS = int(os.environ.get('HELPER_WARMUP_STEPS','10000'))
    HELPER_LR_SCALE = float(os.environ.get('HELPER_LR_SCALE','0.1'))
    tok = get_codebook_tokenizer()
    # Ensure codebook mappings are built before any encoding to avoid first-encode stall
    try:
        tok._ensure_codebook_vocab()
        print("[Tokenizer] Codebook vocab ensured")
    except Exception as e:
        print(f"[Tokenizer] Codebook ensure failed: {e}")
    data_dir = BASE / "trainingData"
    # Build codebook from training texts (parquet) rather than empty .txt files
    # Skip rebuilding codebook if resuming from checkpoint to preserve vocab size
    models_dir = BASE / "runtime" / "models"
    resume_exists = any(models_dir.glob("weave_decoder_step*.pt")) or (models_dir / "weave_decoder_full.pt").exists()
    if hasattr(tok,'codebook') and not resume_exists:
        try:
            print("Pre-building hierarchical codebook from sample parquet data...")
            # Load a small sample for fast codebook init
            import pandas as pd, glob
            sample_texts=[]
            for f in glob.glob(str(data_dir/"*.parquet")):
                try:
                    df=pd.read_parquet(f)
                    for col in df.columns:
                        sample_texts.extend(df[col].dropna().astype(str).tolist()[:100])
                        if len(sample_texts)>=500:
                            break
                except Exception:
                    pass
                if len(sample_texts)>=500:
                    break
            if sample_texts:
                tok.build_codebook(sample_texts[:500], min_count=2)
                print(f"Hierarchical token codebook pre-built with {len(tok.codebook.text_to_ref)} entries")
            else:
                print("Token codebook pre-build skipped: no sample texts found")
        except Exception as e:
            print(f"Token codebook pre-build failed: {e}")
    vocab = getattr(tok,'bpe').vocab
    bpe_size=len(vocab)
    codebook_entries=len(getattr(tok.codebook,'text_to_ref',{}))
    dict_size=len(getattr(tok,'_word_dict',{}))
    vocab_size = min(1_048_576, max(8192, bpe_size + dict_size + codebook_entries))
    # Ensure DIM divisible by HEADS
    if DIM % HEADS != 0:
        # bump DIM to next multiple
        new_dim = ((DIM // HEADS) + 1) * HEADS
        print(f"[Config] DIM {DIM} not divisible by HEADS {HEADS}, bumping to {new_dim}")
        DIM = new_dim
    model = DualWeightWeaveFormer(dim=DIM,num_layers=32,vocab_size=vocab_size,use_dual_weights=True,nhead=HEADS)
    print("Using DualWeightWeaveFormer with master+delta weights")
    model.to(device)
    # Production: compile model if available - disabled by default due to Triton requirement
    # FAST_MODE compile is disabled to avoid Triton backend errors
    print("[Compile] Skipped: torch.compile disabled for stability (Triton not available)")
    # Memory Orchestrator init
    try:
        memory_orch = AIMemoryOrchestrator(dim=DIM, memory_root=BASE/"runtime"/"memory")
        print("AIMemoryOrchestrator initialized")
    except Exception as e:
        print(f"AIMemoryOrchestrator init failed: {e}")
        memory_orch = None
    # Matrix calc engine with CPU offload + Bloom cache
    try:
        matrix_engine = MatrixCalcEngine(dim=DIM, cache_size=65536)
        print("MatrixCalcEngine initialized with CPU offload + Bloom cache")
    except Exception as e:
        print(f"MatrixCalcEngine init failed: {e}")
        matrix_engine = None
    # Ensure cold overlay has non-zero data to drive zoning
    try:
        import json, numpy as np
        cold_path = BASE / "runtime" / "memory" / "cold_overlay.json"
        if cold_path.exists():
            with open(cold_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cold_map = data.get('cold_map', [])
            # Check if all zeros
            all_zero = True
            for plane in cold_map:
                for row in plane:
                    if any(v != 0 for v in row):
                        all_zero = False
                        break
                if not all_zero:
                    break
            if all_zero or not cold_map:
                # Initialize with small random coldness
                # Assume 4 planes of 64x64 for demo
                planes = 4
                size = 64
                new_map = []
                rng = np.random.default_rng(42)
                for _ in range(planes):
                    plane = rng.uniform(0.0, 0.3, size=(size, size)).tolist()
                    new_map.append(plane)
                data['cold_map'] = new_map
                with open(cold_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                print("[ColdOverlay] Initialized with random coldness")
    except Exception:
        pass
    # Pattern mapper init - layered tree with fixed size nodes
    pattern_tree = None
    pattern_weights = {}
    if PATTERN_TREE_AVAILABLE:
        try:
            pattern_dir = BASE/"runtime"/"memory"/"pattern_tree"
            pattern_tree = LayeredPatternTree(capacity_per_node=2000, max_nodes_per_layer=256, dir_path=str(pattern_dir), name="patterns")
            print(f"LayeredPatternTree initialized: {pattern_tree.stats()}")
            # Sync in-memory weights with persisted tree weights
            pattern_weights.update(pattern_tree.weights)
            # Auto-fill from existing saved pattern_weights.json if present
            try:
                import json
                pw_path = BASE / "runtime" / "memory" / "pattern_tree" / "patterns" / "pattern_weights.json"
                if pw_path.exists():
                    with open(pw_path, 'r', encoding='utf-8') as f:
                        saved = json.load(f)
                        # Merge, prefer existing in-memory, then saved
                        for k,v in saved.items():
                            pattern_weights.setdefault(k, float(v))
                    print(f"[Pattern] Auto-filled {len(saved)} existing patterns from disk")
            except Exception as e:
                print(f"[Pattern] Auto-fill skipped: {e}")
        except Exception as e:
            print(f"LayeredPatternTree init failed: {e}")
            pattern_tree = None

    # Helper & Micro-helper pattern mappers for accelerated learning
    helper_pattern_tree = None
    micro_pattern_tree = None
    helper_weights = {}
    micro_weights = {}
    helper_loss_bias = 1.0
    micro_loss_bias = 1.0
    if PATTERN_TREE_AVAILABLE:
        try:
            helper_dir = BASE/"runtime"/"memory"/"pattern_tree"/"helper"
            helper_pattern_tree = LayeredPatternTree(capacity_per_node=512, max_nodes_per_layer=128, dir_path=str(helper_dir), name="helper_patterns")
            helper_weights.update(helper_pattern_tree.weights)
            # Auto-fill helper weights from disk
            try:
                import json
                hw_path = BASE / "runtime" / "memory" / "pattern_tree" / "helper" / "helper_patterns" / "pattern_weights.json"
                if hw_path.exists():
                    with open(hw_path, 'r', encoding='utf-8') as f:
                        saved = json.load(f)
                        for k,v in saved.items():
                            helper_weights.setdefault(k, float(v))
                    print(f"[Helper] Auto-filled {len(saved)} existing patterns from disk")
                else:
                    # Seed helper from main pattern weights if empty
                    main_pw = BASE / "runtime" / "memory" / "pattern_tree" / "patterns" / "pattern_weights.json"
                    if main_pw.exists() and not helper_weights:
                        with open(main_pw, 'r', encoding='utf-8') as f:
                            main_saved = json.load(f)
                            # Copy top 5000 patterns
                            for k,v in list(main_saved.items())[:5000]:
                                helper_weights[k] = float(v)
                        print(f"[Helper] Seeded {len(helper_weights)} patterns from main tree")
            except Exception:
                pass
            print(f"HelperPatternTree initialized: {helper_pattern_tree.stats()}")
        except Exception as e:
            print(f"HelperPatternTree init failed: {e}")
            helper_pattern_tree = None
        try:
            micro_dir = BASE/"runtime"/"memory"/"pattern_tree"/"micro"
            micro_pattern_tree = LayeredPatternTree(capacity_per_node=256, max_nodes_per_layer=64, dir_path=str(micro_dir), name="micro_patterns")
            micro_weights.update(micro_pattern_tree.weights)
            # Auto-fill micro weights from disk
            try:
                import json
                mw_path = BASE / "runtime" / "memory" / "pattern_tree" / "micro" / "micro_patterns" / "pattern_weights.json"
                if mw_path.exists():
                    with open(mw_path, 'r', encoding='utf-8') as f:
                        saved = json.load(f)
                        for k,v in saved.items():
                            micro_weights.setdefault(k, float(v))
                    print(f"[Micro] Auto-filled {len(saved)} existing patterns from disk")
                else:
                    # Seed micro from main pattern weights if empty
                    main_pw = BASE / "runtime" / "memory" / "pattern_tree" / "patterns" / "pattern_weights.json"
                    if main_pw.exists() and not micro_weights:
                        with open(main_pw, 'r', encoding='utf-8') as f:
                            main_saved = json.load(f)
                            for k,v in list(main_saved.items())[:2000]:
                                micro_weights[k] = float(v)
                        print(f"[Micro] Seeded {len(micro_weights)} patterns from main tree")
            except Exception:
                pass
            print(f"MicroPatternTree initialized: {micro_pattern_tree.stats()}")
        except Exception as e:
            print(f"MicroPatternTree init failed: {e}")
            micro_pattern_tree = None
    # EMA
    ema_decay = 0.999
    ema_model = None
    try:
        import copy
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()
        print("EMA model initialized")
    except Exception as e:
        print(f"EMA init skipped: {e}")
    # Split params for main vs helper to allow different LRs
    main_params = [p for n,p in model.named_parameters() if not ('helper' in n or 'helper_wave' in n)]
    helper_params = [p for n,p in model.named_parameters() if ('helper' in n or 'helper_wave' in n)]
    optimizer = torch.optim.Adam([
        {'params': main_params, 'lr': BASE_LR},
        {'params': helper_params, 'lr': BASE_LR * HELPER_LR_SCALE}
    ], weight_decay=0.001)
    scaler = torch.amp.GradScaler('cuda')
    base_lr = BASE_LR
    label_smooth = 0.05
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    # Resonant drafting wrapper
    res_drafter = ResonantDraftOptimizer(base_lr=BASE_LR, zeta=0.15, draft_strength=0.5)
    # Load texts
    texts = load_texts(data_dir)
    # SFT mode: mixed supervised examples
    sft_pool = []
    if SFT_MODE and SFT_DATA_PATH and os.path.exists(SFT_DATA_PATH):
        try:
            import json
            sft_examples = []
            with open(SFT_DATA_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    inp = obj.get('input','') or obj.get('prompt','')
                    out = obj.get('output','') or obj.get('response','') or obj.get('assistant','') or obj.get('answer','')
                    if inp and out:
                        # Simple instruction format
                        sft_examples.append(f"User: {inp}\nAssistant: {out}")
            if sft_examples:
                sft_pool = sft_examples
                print(f"[SFT] Loaded {len(sft_examples)} supervised examples from {SFT_DATA_PATH}")
                # Light prepend for initial bias, main mixing happens per-batch
                texts = sft_examples * 2 + texts
        except Exception as e:
            print(f"[SFT] Failed to load SFT data: {e}")
    # sanitize simple
    texts = [t for t in texts if len(t)>10]
    random.shuffle(texts)
    val_size = max(100, len(texts)//100)
    val_texts = texts[:val_size]
    train_texts = texts[val_size:]
    print(f"Loaded {len(texts)} texts, validation {len(val_texts)}")
    if os.environ.get('SKIP_WARMUP','0') == '1':
        print("[Init] Warmup skipped via SKIP_WARMUP")
    else:
        print("[Init] Starting tokenizer warmup...")
        # Pre-warm tokenizer cache with a small encode to prime codebook
        # Avoid ThreadPoolExecutor deadlock on Windows; use single-threaded warmup
        def _encode_one(t):
            try:
                return tok.encode(t, chunk_size=512, overlap=64)
            except Exception as e:
                print(f"[Warmup] encode error: {e}")
                return []
        try:
            warm_samples = train_texts[:min(50, len(train_texts))]
            print(f"[Warmup] encoding first sample: {warm_samples[0][:80]!r}")
            _encode_one(warm_samples[0])
            print("[Warmup] first sample ok")
            for i, t in enumerate(warm_samples[1:], start=1):
                print(f"[Warmup] encoding sample {i}, len={len(t)}, words={len(t.split())}, preview={t[:60]!r}")
                _encode_one(t)
                if i % 10 == 0:
                    print(f"[Warmup] progress {i}/{len(warm_samples)-1}")
            print(f"[CPU Warmup] Tokenizer primed on {len(warm_samples)} samples")
        except Exception as e:
            print(f"[CPU Warmup] skipped: {e}")
    # DataLoader GPU friendly - start with short sequences for fast learning
    max_seq_len = INITIAL_SEQ
    base_batch = INITIAL_BS
    batch_size = base_batch
    train_dataset = CodebookTextDataset(train_texts, tokenizer=tok)
    # Use multiple CPU workers for tokenization pre-processing
    import platform
    # Windows spawn cannot pickle local lambda collate; keep num_workers=0 on Windows
    num_workers = 0 if platform.system() == 'Windows' else 4
    def make_loader(bs, seq_len):
        # Randomize sequence length per batch for length generalization
        def collate_random(b):
            # Sample length between 64 and seq_len for variety
            import random
            min_len = 64
            rand_len = random.randint(min_len, seq_len)
            return collate_codebook_batch(b, max_seq_len=rand_len)
        return DataLoader(train_dataset, batch_size=bs, shuffle=True,
                          num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0,
                          collate_fn=collate_random)
    train_loader = make_loader(batch_size, max_seq_len)
    print(f"[Data] DataLoader created bs={batch_size} seq={max_seq_len}")
    train_iter = iter(train_loader)
    # Ensure runtime directories exist
    (BASE / "runtime" / "models").mkdir(parents=True, exist_ok=True)
    (BASE / "runtime" / "models" / "codebook_checkpoints").mkdir(parents=True, exist_ok=True)
    # Tree ledger for window samples to avoid 4GB monolith
    ledger_dir = BASE / "runtime" / "memory" / "tree_ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    tree_ledger = TreeStore(capacity_per_node=10000, dir_path=str(ledger_dir), name="ledger")
    # Runtime health check
    try:
        models_dir = BASE / "runtime" / "models"
        step_ckpts = sorted(models_dir.glob("weave_decoder_step*.pt"))
        latest_ckpt = step_ckpts[-1] if step_ckpts else None
        latest_step = -1
        if latest_ckpt:
            try:
                latest_step = int(latest_ckpt.stem.split('_')[-1])
            except Exception:
                pass
        # Count ledger entries - skip expensive walk on Windows
        ledger_entries = 0
        try:
            # Quick estimate without full walk to avoid hangs
            ledger_entries = 0
        except Exception:
            pass
        # Memory vector count
        mem_vectors = 0
        try:
            mem_root = BASE / "runtime" / "memory"
            # AIMemoryOrchestrator will load on init, but we can approximate from vectors.json
            vec_file = mem_root / "vecmem_heatmap" / "vectors.json"
            if vec_file.exists():
                import json
                with open(vec_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    mem_vectors = len(data) if isinstance(data, dict) else 0
        except Exception:
            pass
        print(f"[Runtime Health] latest checkpoint step={latest_step}, ledger entries≈{ledger_entries}, memory vectors≈{mem_vectors}")
    except Exception as e:
        print(f"[Runtime Health] check skipped: {e}")
    print("[Train] Starting training loop...", flush=True)
    # Training loop
    print("[Train] Reading config...", flush=True)
    steps = int(os.environ.get('TRAIN_STEPS','10000000'))
    print(f"[Train] steps={steps}", flush=True)
    SAVE_EVERY = int(os.environ.get('SAVE_EVERY','20000'))
    print(f"[Train] SAVE_EVERY={SAVE_EVERY}", flush=True)
    OVERFIT_GUARD = True
    VAL_SAMPLES = int(os.environ.get('VAL_SAMPLES','1000'))
    VAL_INTERVAL = int(os.environ.get('VAL_INTERVAL','100'))
    print(f"[Train] VAL_SAMPLES={VAL_SAMPLES}", flush=True)
    print(f"[Train] VAL_INTERVAL={VAL_INTERVAL}", flush=True)
    # Kid-learning simulation – always on
    KID_MODE = True
    KID_EASY_SAMPLES = 500
    KID_MASTERY_THRESHOLD = 4.0
    KID_REPLAY_FACTOR = 2.0
    KID_EASY_GATE = True
    print(f"[Kid] Mode ON, easy samples={KID_EASY_SAMPLES}, mastery threshold={KID_MASTERY_THRESHOLD}", flush=True)
    print("[Train] Config read complete", flush=True)
    print("[Train] Creating optimizer...", flush=True)
    if FAST_MODE:
        VAL_SAMPLES = min(VAL_SAMPLES, 500)
    ROT_PROB = float(os.environ.get('ROT_PROB','0.62'))
    WARMUP_STEPS = int(os.environ.get('WARMUP_STEPS','2000'))
    INITIAL_SEQ = int(os.environ.get('INITIAL_SEQ','512'))
    INITIAL_BS = int(os.environ.get('INITIAL_BS','4'))
    DIM = int(os.environ.get('DIM','288'))
    HEADS = int(os.environ.get('HEADS','9'))
    GRAD_ACCUM = int(os.environ.get('GRAD_ACCUM','4'))
    if FAST_MODE:
        GRAD_ACCUM = max(GRAD_ACCUM, 8)
    MULTI_PASS_K = int(os.environ.get('MULTI_PASS_K','2'))
    RL_REPLAY_PROB = float(os.environ.get('RL_REPLAY_PROB','0.15'))
    REINFORCE_EPOCHS = int(os.environ.get('REINFORCE_EPOCHS','2'))
    REINFORCE_BATCH_SIZE = int(os.environ.get('REINFORCE_BATCH_SIZE','32'))
    REINFORCE_UPDATE_STEPS = int(os.environ.get('REINFORCE_UPDATE_STEPS','500'))
    RIEMANN_ANN = os.environ.get('RIEMANN_ANN','1') == '1'
    USE_BRACHISTOCHRONE = os.environ.get('USE_BRACHISTOCHRONE','1') == '1'
    COLD_MIRROR = os.environ.get('COLD_MIRROR','1') == '1'
    PATTERN_WEIGHTED_SAMPLING = os.environ.get('PATTERN_WEIGHTED_SAMPLING','1') == '1'
    VALIDATION_STRATIFIED = os.environ.get('VALIDATION_STRATIFIED','1') == '1'
    TOKEN_DROPOUT_ANNEAL = os.environ.get('TOKEN_DROPOUT_ANNEAL','1') == '1'
    LABEL_SMOOTH_ANNEAL = os.environ.get('LABEL_SMOOTH_ANNEAL','1') == '1'
    CHECKPOINT_KEEP = int(os.environ.get('CHECKPOINT_KEEP','5'))
    LOG_TO_FILE = os.environ.get('LOG_TO_FILE','1') == '1'
    start_step = 0
    ema_loss = None
    loss_target = 3.5
    # Default biases, will be overwritten by checkpoint if present
    pattern_loss_bias = 1.0
    zone_bias = 1.0
    helper_loss_bias = 1.0
    micro_loss_bias = 1.0
    print("[Train] Setting up signal handlers...", flush=True)
    # Production: graceful shutdown handler
    import signal
    def _signal_handler(sig, frame):
        global _shutdown_requested
        print(f"\n[Signal] Received interrupt, saving checkpoint and exiting...")
        _shutdown_requested = True
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    print("[Train] Signal handlers set up", flush=True)
    print("[Train] Checking for existing checkpoints...", flush=True)
    # Safe resume attempt - prefer latest step checkpoint
    models_dir = BASE / "runtime" / "models"
    # Resume from codebook checkpoint
    cb_dir = models_dir / "codebook_checkpoints"
    if cb_dir.exists():
        metas = sorted(cb_dir.glob("weave_decoder_cb_step*_meta.pt"))
        if metas:
            try:
                latest = metas[-1]
                base_name = latest.stem.replace("_meta","")
                print(f"Attempting codebook resume from {latest}")
                _, extra_state = load_model_codebook(model, cb_dir, base_name=base_name)
                start_step = extra_state.get('step',0) if extra_state else 0
                print(f"Resumed at step {start_step} from codebook")
            except Exception as e:
                print(f"Codebook resume failed: {e}")
    # Find latest step checkpoint with natural numeric sort
    def _step_from_path(p):
        import re
        m = re.search(r'(\d+)$', p.stem)
        try:
            return int(m.group(1)) if m else -1
        except Exception:
            return -1
    step_ckpts = list(models_dir.glob("weave_decoder_step*.pt"))
    ckpt_path = None
    if step_ckpts:
        step_ckpts_sorted = sorted(step_ckpts, key=_step_from_path, reverse=True)
        ckpt_path = step_ckpts_sorted[0]
        try:
            step_num = _step_from_path(ckpt_path)
            start_step = max(start_step, step_num)
        except Exception:
            pass
    # Fallback to best / full / main checkpoint
    if ckpt_path is None:
        main_ckpt = models_dir / "weave_decoder_main.pt"
        best_ckpt = models_dir / "weave_decoder_best.pt"
        full_ckpt = models_dir / "weave_decoder_full.pt"
        if main_ckpt.exists():
            ckpt_path = main_ckpt
        elif best_ckpt.exists():
            ckpt_path = best_ckpt
        elif full_ckpt.exists():
            ckpt_path = full_ckpt
    if ckpt_path is not None and ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            # Allow size mismatches when DIM/HEADS changed
            state_dict = ckpt['model']
            # Strip _orig_mod. prefix if present from old checkpoints
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                state_dict = {k.replace('_orig_mod.',''):v for k,v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            start_step = ckpt.get('step', start_step)
            ema_loss = ckpt.get('ema_loss', ema_loss)
            # Restore curriculum state if saved
            if 'max_seq_len' in ckpt:
                max_seq_len = ckpt['max_seq_len']
            if 'batch_size' in ckpt:
                batch_size = ckpt['batch_size']
            if 'loss_target' in ckpt:
                loss_target = ckpt['loss_target']
            if 'pattern_loss_bias' in ckpt:
                pattern_loss_bias = float(ckpt['pattern_loss_bias'])
            if 'zone_bias' in ckpt:
                zone_bias = float(ckpt['zone_bias'])
            if 'helper_loss_bias' in ckpt:
                helper_loss_bias = float(ckpt['helper_loss_bias'])
            if 'micro_loss_bias' in ckpt:
                micro_loss_bias = float(ckpt['micro_loss_bias'])
            # Load optimizer/scaler state if present
            if 'optimizer' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    print(f"Resumed optimizer state from {ckpt_path}")
                except Exception as e:
                    print(f"Optimizer state resume skipped due to param mismatch: {e}")
            if 'scaler' in ckpt:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                    print(f"Resumed scaler state from {ckpt_path}")
                except Exception as e:
                    print(f"Scaler state resume skipped: {e}")
            print(f"Resumed model weights from {ckpt_path} at step {start_step}, seq {max_seq_len} bs {batch_size}")
            # Recreate DataLoader to match restored curriculum
            train_loader = make_loader(batch_size, max_seq_len)
            train_iter = iter(train_loader)
            # EMA
            ema_ckpt_name = ckpt_path.stem.replace('weave_decoder','weave_decoder_ema') + ckpt_path.suffix
            ema_ckpt_path = models_dir / ema_ckpt_name
            if ema_model is not None and ema_ckpt_path.exists():
                ema_ckpt = torch.load(ema_ckpt_path, map_location=device)
                ema_state = ema_ckpt['model']
                if any(k.startswith('_orig_mod.') for k in ema_state.keys()):
                    ema_state = {k.replace('_orig_mod.',''):v for k,v in ema_state.items()}
                ema_model.load_state_dict(ema_state, strict=False)
                print(f"Resumed EMA weights from {ema_ckpt_path}")
            else:
                # fallback to generic ema file
                ema_generic = models_dir / "weave_decoder_ema.pt"
                if ema_generic.exists():
                    ema_ckpt = torch.load(ema_generic, map_location=device)
                    ema_state = ema_ckpt['model']
                    if any(k.startswith('_orig_mod.') for k in ema_state.keys()):
                        ema_state = {k.replace('_orig_mod.',''):v for k,v in ema_state.items()}
                    ema_model.load_state_dict(ema_state, strict=False)
                    print(f"Resumed EMA weights from {ema_generic}")
        except Exception as e:
            print(f"Torch checkpoint resume failed: {e}")
    print("[Train] Checkpoint loading complete", flush=True)
    alpha = 0.995
    prev_loss_val = None
    rotation_explore_bias = 0.0
    # pattern_loss_bias, zone_bias, helper_loss_bias, micro_loss_bias already initialized above
    zone_focus_idx = 0
    zone_focus_steps = 0
    val_loss_ema = None
    val_regress_counter = 0
    restore_cooldown_steps = 0
    pattern_replay = []
    # REINFORCE buffer for multi-pass RL
    import collections
    reinforce_buffer = collections.deque(maxlen=2000)
    # Curriculum - advance when EMA loss < threshold
    curriculum_levels = [(128,16,6.0),(256,8,5.0),(512,4,4.2),(1024,2,3.5)]
    # Risk-adjusted optimizer state
    grad_ema_mean = {}
    grad_ema_var = {}
    RISK_BETA = 0.99
    RISK_EPS = 1e-8
    # Marker credit assignment state
    marker_ema = {}
    MARKER_BETA = 0.99
    MARKER_SCALE = float(os.environ.get('MARKER_SCALE','0.5'))
    # Overfitting guard / best checkpoint tracking
    best_val_loss = float('inf')
    best_val_step = -1
    patience = int(os.environ.get('PATIENCE','20000'))
    no_improve_steps = 0
    lr_reductions = 0
    val_rise_counter = 0
    best_ckpt_path = BASE / "runtime" / "models" / "weave_decoder_best.pt"
    print(f"[Train] Entering training loop from step {start_step} to {steps}", flush=True)
    print(f"[Train] Dataset size {len(train_dataset)} batch_size {batch_size} max_seq_len {max_seq_len}", flush=True)
    for step in range(start_step, steps):
        if _shutdown_requested:
            print(f"[Signal] Shutdown requested, breaking training loop at step {step}")
            break
        # Helper warmup freeze: keep helper params frozen until warmup
        if step < HELPER_WARMUP_STEPS:
            for name, p in model.named_parameters():
                if 'helper' in name or 'helper_wave' in name:
                    p.requires_grad_(False)
        else:
            for name, p in model.named_parameters():
                if 'helper' in name or 'helper_wave' in name:
                    p.requires_grad_(True)
        # curriculum advance
        curriculum_changed = False
        kid_gate_ok = True
        if KID_MODE and KID_EASY_GATE:
            try:
                kid_gate_ok = globals().get('kid_easy_loss_ema', float('inf')) <= KID_MASTERY_THRESHOLD
            except Exception:
                kid_gate_ok = True
        if ema_loss is not None and ema_loss < loss_target*1.05 and kid_gate_ok:
            for i,(seq,bs,target) in enumerate(curriculum_levels):
                if max_seq_len==seq and batch_size==bs:
                    if i+1<len(curriculum_levels):
                        new_seq,new_bs,new_target = curriculum_levels[i+1]
                        if new_seq != max_seq_len or new_bs != batch_size:
                            max_seq_len,batch_size,loss_target = new_seq,new_bs,new_target
                            print(f"Curriculum advance: seq {max_seq_len} bs {batch_size}")
                            # Recreate DataLoader to match new batch size / seq len
                            train_loader = make_loader(batch_size, max_seq_len)
                            train_iter = iter(train_loader)
                            curriculum_changed = True
                    break
        # Quantum rotation annealing
        if QUANTUM and schrodinger_anneal_schedule and quantum_rotation_prob:
            temp, beta = schrodinger_anneal_schedule(step, total_steps=steps, t0=1.0, t_min=0.01)
            # loss velocity proxy
            loss_velocity = 0.0
            if prev_loss_val is not None and ema_loss is not None:
                loss_velocity = abs(ema_loss - prev_loss_val)
            rotation_prob = quantum_rotation_prob(float(ROT_PROB), float(loss_velocity), float(temp))
        else:
            rotation_prob = ROT_PROB
        # Riemann inspired rotation modulation
        if RIEMANN_ANN:
            riemann_factor = _riemann_lr_factor(step, steps)
            rotation_prob = min(0.9, max(0.3, rotation_prob * riemann_factor))
        # Rotation annealing enabled; keep ROT_PROB as baseline, allow annealing
        # rotation_prob = float(ROT_PROB)  # disabled to allow annealing
        # Cosine LR schedule with warmup and warm-restart floor
        # Keep base_lr as peak, compute current lr
        warmup_factor = 1.0
        if step < WARMUP_STEPS:
            warmup_factor = max(1e-6, step / max(1, WARMUP_STEPS))
        # cosine progress
        progress = min(1.0, max(0.0, (step - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)))
        if USE_BRACHISTOCHRONE:
            theta = math.pi * progress
            y_norm = (1 - math.cos(theta)) / 2.0
            # Brachistochrone-inspired fast descent: 1 - sqrt(y)
            brach_factor = 1.0 - math.sqrt(max(0.0, y_norm))
            cosine_factor = max(1e-4, brach_factor)
        else:
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = base_lr * warmup_factor * cosine_factor
        # Riemann hypothesis inspired modulation
        if RIEMANN_ANN:
            riemann_factor = _riemann_lr_factor(step, steps)
            lr = lr * riemann_factor
        # Periodic LR kick to escape plateaus, but respect cosine
        if step > 0 and step % 50000 == 0:
            base_lr = min(5e-4, base_lr * 1.5)
            print(f"[LR Kick] Step {step} base LR boosted to {base_lr:.6f}")
        # Heatmap zoning with sequential cold-zone focus
        if step % 5000 == 0 and step > 0:
            try:
                import json, numpy as np
                cold_path = BASE / "runtime" / "memory" / "cold_overlay.json"
                if cold_path.exists():
                    with open(cold_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    cold_map = data.get('cold_map', [])
                    # Mirror cold map for symmetry when enabled
                    if COLD_MIRROR and cold_map:
                        try:
                            new_map = []
                            for plane in cold_map:
                                arr = np.array(plane, dtype=float)
                                mirrored = np.fliplr(arr)
                                blended = 0.9 * arr + 0.1 * mirrored
                                new_map.append(blended.tolist())
                            data['cold_map'] = new_map
                            with open(cold_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f)
                            cold_map = new_map
                        except Exception:
                            pass
                    if cold_map:
                        # Compute mean coldness per plane/zone
                        zone_means = []
                        for plane in cold_map:
                            vals = []
                            for row in plane:
                                vals.extend(row)
                            if vals:
                                zone_means.append(float(np.mean(vals)))
                            else:
                                zone_means.append(0.0)
                        # Focus on coldest zone first, stay until it warms up
                        if zone_means:
                            # Find coldest zone index
                            coldest_idx = int(np.argmax(zone_means))
                            coldest_val = zone_means[coldest_idx]
                            # Stay on current zone until it warms up, then move to next coldest
                            current_mean = zone_means[zone_focus_idx] if zone_focus_idx < len(zone_means) else 0.0
                            min_stay_steps = 50000
                            warm_threshold = 0.02
                            if zone_focus_idx >= len(zone_means):
                                zone_focus_idx = coldest_idx
                                zone_focus_steps = 0
                            elif current_mean < warm_threshold and zone_focus_steps >= min_stay_steps:
                                # current zone warmed up sufficiently, switch to coldest
                                zone_focus_idx = coldest_idx
                                zone_focus_steps = 0
                            elif coldest_idx != zone_focus_idx and coldest_val > current_mean + 0.08:
                                # coldest is significantly colder, switch now
                                zone_focus_idx = coldest_idx
                                zone_focus_steps = 0
                            zone_focus_steps += 5000
                            # Use only the focused zone's coldness for bias
                            target_cold = zone_means[zone_focus_idx]
                            if COLD_MIRROR:
                                # Mirror coldness around mean to avoid runaway heating
                                mean_cold = float(np.mean(zone_means)) if zone_means else 0.0
                                # Symmetric push: colder than mean -> slight up bias, hotter -> slight down
                                delta = mean_cold - target_cold
                                # tanh limits the push to ~±0.2
                                target_bias = 1.0 + 0.2 * float(np.tanh(delta * 2.0))
                            else:
                                target_bias = 1.0 + 0.4 * min(1.0, max(0.0, target_cold))
                            zone_bias = 0.9*zone_bias + 0.1*target_bias
                            zone_bias = max(1.0, min(1.5, zone_bias))
                            # Optional log
                            # print(f"[Zone] Focus zone {zone_focus_idx} cold={target_cold:.3f} bias={zone_bias:.3f}")
            except Exception:
                pass
        # Pattern replay removed per user request
        input_padded = None
        target_padded = None
        # Mixed SFT sampling for teacher-student learning with decay
        try:
            if SFT_MODE and sft_pool:
                sft_prob = max(0.05, 0.20 - step / 500000.0)
                if random.random() < sft_prob:
                    txt = random.choice(sft_pool)
                    ids = tok.encode(txt)
                    if len(ids) >= 2:
                        if len(ids) > max_seq_len:
                            ids = ids[:max_seq_len]
                        inp_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                        tgt_ids = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
                        input_padded = inp_ids
                        target_padded = tgt_ids
        except Exception:
            pass
        if input_padded is None:
            print(f"[Train] Step {step}: fetching batch...", flush=True)
            # get batch from DataLoader
            while True:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                if batch is None:
                    continue
                input_padded, target_padded = batch
                print(f"[Train] Step {step}: batch fetched shape {input_padded.shape}", flush=True)
                break
        input_padded = input_padded.to(device, non_blocking=True)
        target_padded = target_padded.to(device, non_blocking=True)
        # clamp ids and sanity checks
        if input_padded.max().item() >= vocab_size:
            input_padded.clamp_(max=vocab_size-1)
        if target_padded.max().item() >= vocab_size:
            target_padded.clamp_(max=vocab_size-1)
        # Data sanity: shapes match, no NaNs
        assert input_padded.shape == target_padded.shape, "Input/target shape mismatch"
        assert torch.isfinite(input_padded).all(), "NaN in inputs"
        assert torch.isfinite(target_padded).all(), "NaN in targets"
        mask = target_padded != 0
        # Light token dropout for regularisation with annealing
        if step > WARMUP_STEPS and step % 2 == 0:
            if TOKEN_DROPOUT_ANNEAL:
                # anneal from 0.07 down to 0.01 over training
                progress = min(1.0, step / max(1, steps))
                dropout_p = 0.07 * (1.0 - 0.85 * progress) + 0.01
            else:
                dropout_p = 0.07
            # randomly mask a fraction of input tokens to pad value
            mask_drop = torch.rand_like(input_padded.float()) < dropout_p
            input_padded = torch.where(mask_drop, torch.zeros_like(input_padded), input_padded)
        model.train()
        # Anneal label smoothing
        if LABEL_SMOOTH_ANNEAL:
            progress = min(1.0, step / max(1, steps))
            # anneal from 0.05 down to 0.01
            new_smooth = 0.05 * (1.0 - 0.8 * progress) + 0.01
            if abs(new_smooth - label_smooth) > 1e-4:
                label_smooth = new_smooth
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(input_padded)
            if isinstance(logits,tuple):
                logits=logits[0]
            # Stress inoculation: small logit noise early in training
            if os.environ.get('STRESS_INOCULATION','1') == '1' and step < WARMUP_STEPS*2:
                noise_scale = 0.02 * max(0.0, 1.0 - step/(WARMUP_STEPS*2))
                logits = logits + torch.randn_like(logits) * noise_scale
            # Clamp logits to prevent softmax overflow in loss
            logits = torch.clamp(logits, -10.0, 10.0)
            # Guard against NaNs from model with diagnostics
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[NaN Guard] logits contain NaN/Inf at step {step}, skipping")
                # Diagnostic dump
                try:
                    print(f"  logits stats: min={logits.min().item():.4e} max={logits.max().item():.4e} mean={logits.mean().item():.4e} std={logits.std().item():.4e}")
                    print(f"  input stats: min={input_padded.min().item()} max={input_padded.max().item()} unique={len(torch.unique(input_padded))}")
                    # Parameter health
                    nan_params = []
                    inf_params = []
                    for n,p in model.named_parameters():
                        if torch.isnan(p).any(): nan_params.append(n)
                        if torch.isinf(p).any(): inf_params.append(n)
                    if nan_params:
                        print(f"  NaN params: {nan_params[:5]}")
                    if inf_params:
                        print(f"  Inf params: {inf_params[:5]}")
                    # Gradient norm if available
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.norm().item()**2
                    total_norm = total_norm**0.5
                    print(f"  grad norm approx: {total_norm:.4e}")
                except Exception as e:
                    print(f"  NaN debug failed: {e}")
                optimizer.zero_grad(set_to_none=True)
                continue
            # Consistent loss calculation with flatten + mask
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_padded.view(-1)
            mask_flat = mask.view(-1)
            logits_masked = logits_flat[mask_flat]
            targets_masked = targets_flat[mask_flat]
            loss = criterion(logits_masked, targets_masked)
            # Entropy bonus for exploration when plateauing - delayed to avoid early stall
            if step > max(WARMUP_STEPS, 10000) and no_improve_steps > 2000:
                try:
                    # encourage higher entropy to break stale points
                    probs = torch.softmax(logits_masked, dim=-1)
                    # compute entropy in chunks to avoid OOM
                    # chunk size ~ 1M elements
                    chunk_size = 1024 * 1024
                    n_elements = probs.numel()
                    if n_elements > chunk_size:
                        # process in chunks along first dim
                        num_chunks = (probs.size(0) + chunk_size // probs.size(1) - 1) // (chunk_size // probs.size(1))
                        ent_sum = 0.0
                        count = 0
                        for i in range(0, probs.size(0), max(1, chunk_size // probs.size(1))):
                            chunk = probs[i:i+chunk_size//probs.size(1)]
                            ent_chunk = -(chunk * torch.log(chunk + 1e-12)).sum(dim=-1)
                            ent_sum += ent_chunk.sum().item()
                            count += ent_chunk.numel()
                        entropy = ent_sum / count if count > 0 else torch.tensor(0.0, device=logits_masked.device)
                    else:
                        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
                    ent_weight = 0.02 * min(1.0, no_improve_steps / 5000.0)
                    loss = loss - ent_weight * entropy
                except RuntimeError as e:
                    # skip entropy bonus on OOM
                    if 'out of memory' in str(e).lower():
                        pass
                    else:
                        raise
            # NaN guard with diagnostics
            if not torch.isfinite(loss):
                print(f"[NaN] Skipping step {step} due to non-finite loss")
                try:
                    print(f"  loss value: {loss.item()}")
                    print(f"  logits_masked stats: min={logits_masked.min().item():.4e} max={logits_masked.max().item():.4e} mean={logits_masked.mean().item():.4e}")
                    # Find first NaN source in model outputs
                    nan_layers = []
                    for name, module in model.named_modules():
                        # quick check for buffers/params
                        for n,p in module.named_parameters(recurse=False):
                            if torch.isnan(p).any():
                                nan_layers.append(f"{name}.{n}")
                    if nan_layers:
                        print(f"  NaN in params: {nan_layers[:5]}")
                except Exception as e:
                    print(f"  NaN debug failed: {e}")
                optimizer.zero_grad(set_to_none=True)
                continue
            raw_loss_val = loss.item()
            # Store logits for REINFORCE after weights are known
            logits_for_rl = logits_masked.detach()
            targets_for_rl = targets_masked.detach()
        # Pattern/zone/helper/micro-helper weighting active with soft-start for child learning
        # Attenuate weighting early to avoid over-steering
        if step < WARMUP_STEPS * 2:
            soft = 0.2 + 0.8 * (step / (WARMUP_STEPS * 2))
            pat_w = 1.0 + soft * (pattern_loss_bias - 1.0)
            zone_w = 1.0 + soft * (zone_bias - 1.0)
            helper_w = 1.0 + soft * (helper_loss_bias - 1.0)
            micro_w = 1.0 + soft * (micro_loss_bias - 1.0)
        else:
            pat_w = pattern_loss_bias
            zone_w = zone_bias
            helper_w = helper_loss_bias
            micro_w = micro_loss_bias
        # Pattern-aware modulation is handled via loss weighting and gates.
        # Avoid in-place param multiplication to prevent NaNs.
        # The prediction_bias and logic weights are learned normally and steered by gates.
        # REINFORCE logging after weights are known
        try:
            log_probs = torch.nn.functional.log_softmax(logits_for_rl, dim=-1)
            target_log_probs = log_probs.gather(1, targets_for_rl.unsqueeze(1)).squeeze(1)
            reward_scale = float(pat_w * zone_w)
            reward = reward_scale * max(0.0, 1.0 - loss.item()/10.0)
            reinforce_buffer.append({
                'log_probs': target_log_probs.detach().cpu(),
                'reward': float(reward),
                'step': step
            })
        except Exception:
            pass
        loss = loss * pat_w * zone_w * helper_w * micro_w
        # Temporal coherence regulariser on hidden window memories
        try:
            if hasattr(model, 'layers') and len(model.layers) > 0:
                mem = model.layers[0].hidden_window_mem
                if mem is not None and mem.size(0) > 1:
                    diff = mem[0] - mem[1]
                    coherence_penalty = 0.001 * torch.mean(diff * diff)
                    loss = loss + coherence_penalty
        except Exception:
            pass
        # MoE load-balance encouragement
        try:
            moe_penalty = 0.0
            for layer in model.layers:
                if hasattr(layer, 'moe_gate'):
                    moe_penalty = moe_penalty + 0.0001 * (layer.moe_gate**2).sum()
            if moe_penalty > 0:
                loss = loss + moe_penalty
        except Exception:
            pass
        # Gradient accumulation scaling
        loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        # Determine lead/follower for drafting
        is_lead = ((step // 2) % 2 == 0)
        # Step optimizer every GRAD_ACCUM steps
        if (step + 1) % GRAD_ACCUM == 0:
            # Apply resonant drafting to grads
            res_drafter.prepare_grads(model, is_lead=is_lead)
            # Resonant LR modulation combined with cosine schedule
            lr_scale = res_drafter.get_lr_scale(loss.item()*GRAD_ACCUM)
            lr_scale = max(1e-4, float(lr_scale))  # floor to avoid zero LR
            for pg in optimizer.param_groups:
                pg['lr'] = lr * lr_scale
            scaler.unscale_(optimizer)
            # Marker credit assignment: EMA of |grad| and scale gradients
            for p in model.parameters():
                if p.grad is None:
                    continue
                pid = id(p)
                g_abs = p.grad.detach().abs()
                if pid not in marker_ema:
                    marker_ema[pid] = g_abs.clone().detach()
                else:
                    marker_ema[pid].lerp_(g_abs, 1.0 - MARKER_BETA)
                # Scale grad by marker strength: strong markers get boost
                marker_strength = torch.tanh(marker_ema[pid] * MARKER_SCALE)
                # Boost up to ~2x for high-marker params
                scale = 1.0 + marker_strength
                p.grad.data.mul_(scale)
            # Risk-adjusted gradient scaling pi = delta / sigma
            apply_risk_adjustment(model, grad_ema_mean, grad_ema_var, beta=RISK_BETA, eps=RISK_EPS)
            # Gradient noise injection to break stale plateaus
            if no_improve_steps > patience // 4:
                noise_scale = 1e-4 * min(1.0, no_improve_steps / 10000.0)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * noise_scale)
            # Per-param gradient clip for wave params to prevent NaNs
            for name, p in model.named_parameters():
                if p.grad is not None and ('wave_amp' in name or 'wave_freq' in name or 'wave_phase' in name):
                    p.grad.data.clamp_(-0.1, 0.1)
            # Extra clip for unstable params
            for name, p in model.named_parameters():
                if p.grad is not None and ('layer_idx_embedding' in name or 'layer_idx_gate' in name or 'conv_gates' in name or 'channel_gates' in name or 'helper_global_context' in name):
                    p.grad.data.clamp_(-1e-3, 1e-3)
            # Scale gradients for shared params that get 32x accumulation
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if 'shared_' in name or 'layer_idx_embedding' in name or 'pattern_bank' in name or 'structural_memory' in name:
                        # 32 layers reuse -> scale down
                        p.grad.data.mul_(1.0/32.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # Log marker stats periodically
            if step % 1000 == 0 and marker_ema:
                avg_marker = sum(m.mean().item() for m in marker_ema.values()) / len(marker_ema)
                print(f"[Marker] step {step} avg marker strength {avg_marker:.4f}")
            # SFT warm-up for early chatting bootstrap
            if SFT_MODE and step < SFT_WARMUP_STEPS and step % 100 == 0:
                try:
                    # Use teacher dataset if available
                    if SFT_DATA_PATH:
                        import json
                        sft_path = Path(SFT_DATA_PATH)
                        if sft_path.exists():
                            with open(sft_path, 'r', encoding='utf-8') as f:
                                sft_examples = [json.loads(line) for line in f if line.strip()]
                            if sft_examples:
                                for _ in range(2):
                                    ex = random.choice(sft_examples)
                                    txt = ex.get('text') or ex.get('prompt') or ''
                                    if not txt:
                                        continue
                                    ids = tok.encode(txt)
                                    if len(ids) < 2 or len(ids) > max_seq_len:
                                        continue
                                    inp_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                                    tgt_ids = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
                                    mask = tgt_ids != 0
                                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                                        logits = model(inp_ids)
                                        if isinstance(logits, tuple):
                                            logits = logits[0]
                                        logits_flat = logits.view(-1, logits.size(-1))
                                        targets_flat = tgt_ids.view(-1)
                                        mask_flat = mask.view(-1)
                                        loss_sft = criterion(logits_flat[mask_flat], targets_flat[mask_flat])
                                    # Boost SFT loss early
                                    loss_sft = loss_sft * 2.0
                                    loss_sft = loss_sft / GRAD_ACCUM
                                    scaler.scale(loss_sft).backward()
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
            # Ethical training step
            if ETHICAL_DATA_PATH and step > 0 and step % ETHICAL_EVERY == 0:
                try:
                    import json
                    ethical_path = Path(ETHICAL_DATA_PATH)
                    if ethical_path.exists():
                        with open(ethical_path, 'r', encoding='utf-8') as f:
                            examples = [json.loads(line) for line in f if line.strip()]
                        if examples:
                            # Contrastive ethical training with REINFORCE/PPO style reward
                            for _ in range(4):
                                ex = random.choice(examples)
                                # Support both single examples and preferred/dispreferred pairs
                                pref_txt = ex.get('preferred') or ex.get('text') or ex.get('prompt')
                                disp_txt = ex.get('dispreferred')
                                if not pref_txt:
                                    continue
                                def compute_loss(txt):
                                    ids = tok.encode(txt)
                                    if len(ids) < 2 or len(ids) > max_seq_len:
                                        return None, 0.0
                                    inp_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                                    tgt_ids = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
                                    mask = tgt_ids != 0
                                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                                        logits = model(inp_ids)
                                        if isinstance(logits, tuple):
                                            logits = logits[0]
                                        logits_flat = logits.view(-1, logits.size(-1))
                                        targets_flat = tgt_ids.view(-1)
                                        mask_flat = mask.view(-1)
                                        loss = criterion(logits_flat[mask_flat], targets_flat[mask_flat])
                                    return loss, mask.sum().item()
                                loss_pref, n_pref = compute_loss(pref_txt)
                                if loss_pref is None:
                                    continue
                                # Ethical gate reward proxy: higher loss on dispreferred is good
                                if disp_txt:
                                    loss_disp, n_disp = compute_loss(disp_txt)
                                    if loss_disp is not None:
                                        # Contrastive objective: minimize pref loss, maximize disp loss
                                        # Use margin-based ranking loss
                                        margin = 0.2
                                        # Normalize by token count
                                        loss_pref_norm = loss_pref / max(1.0, n_pref)
                                        loss_disp_norm = loss_disp / max(1.0, n_disp)
                                        # Ranking loss
                                        rank_loss = torch.relu(margin - (loss_disp_norm - loss_pref_norm))
                                        # Combine with standard pref loss
                                        loss_eth = loss_pref_norm + 0.5 * rank_loss
                                        # REINFORCE style reward from ethical gate activation
                                        # Approximate ethical signal by average hidden norm
                                        # Here we use a simple scaling
                                        loss_eth = loss_eth * 1.5
                                    else:
                                        loss_eth = loss_pref * 1.5
                                else:
                                    loss_eth = loss_pref * 1.5
                                loss_eth = loss_eth / GRAD_ACCUM
                                scaler.scale(loss_eth).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            print(f"[Ethical] Step {step} contrastive fine-tune on {len(examples)} examples")
                except Exception as e:
                    print(f"[Ethical] Skipped due to error: {e}")
            # Multi-pass reinforcement learning on high-reward patterns - removed per user request
            # Pattern replay removed, so RL replay is disabled
            # REINFORCE / PPO style update from buffer
            if step > 0 and step % REINFORCE_UPDATE_STEPS == 0 and len(reinforce_buffer) > REINFORCE_BATCH_SIZE:
                try:
                    # Sample mini-batches from buffer
                    buffer_list = list(reinforce_buffer)
                    for _ in range(REINFORCE_EPOCHS):
                        random.shuffle(buffer_list)
                        for i in range(0, len(buffer_list), REINFORCE_BATCH_SIZE):
                            batch = buffer_list[i:i+REINFORCE_BATCH_SIZE]
                            rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32, device=device)
                            # Baseline = running mean reward
                            baseline = rewards.mean()
                            advantages = rewards - baseline
                            # Aggregate log_probs
                            log_probs_batch = []
                            for b in batch:
                                lp = b['log_probs']
                                if len(lp) > 0:
                                    log_probs_batch.append(lp.mean().item())
                            if not log_probs_batch:
                                continue
                            log_probs_tensor = torch.tensor(log_probs_batch, dtype=torch.float32, device=device)
                            # REINFORCE loss with baseline
                            reinforce_loss = -(log_probs_tensor * advantages).mean()
                            # PPO clip placeholder - can be expanded
                            # Backprop reinforce loss
                            scaler.scale(reinforce_loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
            # Dual-weight delta decay
            if hasattr(model, 'apply_delta_decay'):
                try:
                    model.apply_delta_decay()
                except Exception:
                    pass
            # Periodic delta consolidation
            if step % 10000 == 0 and hasattr(model, 'consolidate_delta'):
                try:
                    model.consolidate_delta()
                    print(f"[DualWeight] Consolidated deltas at step {step}")
                except Exception:
                    pass
            # EMA update with warmup
            if ema_model is not None:
                # Warmup EMA decay from 0.99 to 0.999 over first 50k steps
                warm_ema_decay = 0.99 + 0.009 * min(1.0, step / 50000.0)
                effective_decay = warm_ema_decay if step < 50000 else ema_decay
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.mul_(effective_decay).add_(p, alpha=1-effective_decay)
            # Memory orchestrator store experience
            if memory_orch is not None and step % 1000 == 0:
                try:
                    with torch.no_grad():
                        emb = model.embed(input_padded)
                        vec = emb.mean(dim=(0,1)).detach().cpu().numpy()
                        vid = f"step_{step}"
                        token_ids = input_padded[0].tolist()
                        # loss_val not yet defined here, compute directly
                        cur_loss = loss.item()*GRAD_ACCUM
                        memory_orch.store_experience(vid, vec, token_ids, context=f"loss {cur_loss:.4f}")
                except Exception:
                    pass
        loss_val = loss.item()*GRAD_ACCUM
        if ema_loss is None:
            ema_loss = loss_val
        else:
            ema_loss = alpha*ema_loss + (1-alpha)*loss_val
        # logging
        log_interval = 200 if FAST_MODE else 100
        if step % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            raw_loss_print = raw_loss_val * GRAD_ACCUM if 'raw_loss_val' in locals() else loss_val
            print(f"Step {step} loss {loss_val:.4f} raw {raw_loss_print:.4f} ema {ema_loss:.4f} lr {lr:.6f} seq {max_seq_len} bs {batch_size} rot {rotation_prob:.2f} pat_bias {pattern_loss_bias:.3f} zone_bias {zone_bias:.3f} helper_bias {helper_loss_bias:.3f} micro_bias {micro_loss_bias:.3f}")
            # Self-talk probe
            if step > 0 and step % SELF_TALK_EVERY == 0:
                try:
                    model.eval()
                    prompt = SELF_TALK_PROMPT
                    try:
                        eos_id = tok.bpe.inverse_vocab.get('<|endoftext|>', None)
                    except Exception:
                        eos_id = None
                    ids = tok.encode(prompt)
                    if len(ids) > max_seq_len:
                        ids = ids[-max_seq_len:]
                    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                    generated = ids.copy()
                    with torch.no_grad():
                        for _ in range(64):
                            logits = model(input_ids)
                            if isinstance(logits, tuple):
                                logits = logits[0]
                            logits = torch.clamp(logits, -20.0, 20.0)
                            logits[0, -1, 0] = -1e9
                            if TALK_GREEDY:
                                next_id = torch.argmax(logits[0,-1]).item()
                            else:
                                temp = 0.8
                                logits_temp = logits[0,-1] / temp
                                probs = torch.softmax(logits_temp, dim=-1)
                                top_k = 40
                                top_vals, top_idx = torch.topk(probs, k=min(top_k, probs.size(-1)))
                                # repetition penalty
                                for i in range(top_vals.size(0)):
                                    tok_id = top_idx[i].item()
                                    if tok_id in generated[-10:]:
                                        top_vals[i] = top_vals[i] * 0.7
                                probs_top = top_vals / top_vals.sum()
                                next_token = torch.multinomial(probs_top, 1).item()
                                next_id = top_idx[next_token].item()
                            generated.append(next_id)
                            if next_id == 0 or (eos_id is not None and next_id == eos_id):
                                break
                            input_ids = torch.tensor(generated[-max_seq_len:], dtype=torch.long, device=device).unsqueeze(0)
                    text = tok.decode(generated)
                    text = ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in text)
                    # Simple coherence heuristic
                    # Count unique 3-grams vs total
                    tokens = generated
                    if len(tokens) >= 3:
                        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
                        uniq_ratio = len(set(trigrams)) / max(1, len(trigrams))
                    else:
                        uniq_ratio = 0.0
                    print(f"[SelfTalk] Step {step} probe prompt='{prompt}' uniq3={uniq_ratio:.3f} sample: {text[:200]}")
                    if uniq_ratio > 0.6 and ema_loss < 6.0:
                        print(f"[SelfTalk] *** Model appears to be speaking on its own at step {step} ***")
                    model.train()
                except Exception as e:
                    print(f"[SelfTalk] Skipped due to error: {e}")
                    model.train()
            # Tree ledger logging
            try:
                tree_ledger.root.entries.append(float(loss_val))
                if len(tree_ledger.root.entries) >= tree_ledger.capacity:
                    tree_ledger._split_node(tree_ledger.root)
                tree_ledger.root.save()
            except Exception:
                pass
            # validation
            if step>0 and step%VAL_INTERVAL==0:
                model.eval()
                val_items=[]
                # Stratified / random validation sampling
                if VALIDATION_STRATIFIED and len(val_texts) > VAL_SAMPLES:
                    import random as _rnd
                    val_subset = _rnd.sample(val_texts, VAL_SAMPLES)
                else:
                    val_subset = val_texts[:VAL_SAMPLES]
                with torch.inference_mode():
                    for vtxt in val_subset:
                        if _shutdown_requested:
                            break
                        txt = clean_text(vtxt)
                        try:
                            ids = tok.encode(txt)
                        except:
                            continue
                        ids = [i if i<vocab_size else vocab_size-1 for i in ids]
                        if len(ids)<2: continue
                        if len(ids)>max_seq_len: ids=ids[:max_seq_len]
                        inp = torch.tensor(ids[:-1],dtype=torch.long,device=device).unsqueeze(0)
                        tgt = torch.tensor(ids[1:],dtype=torch.long,device=device).unsqueeze(0)
                        out = ema_model(inp) if ema_model else model(inp)
                        logits_v = out[0] if isinstance(out,tuple) else out
                        logits_flat = logits_v.view(-1, logits_v.size(-1))
                        targets_flat = tgt.view(-1)
                        mask_flat = targets_flat != 0
                        if mask_flat.any():
                            loss_val_item = criterion(logits_flat[mask_flat], targets_flat[mask_flat]).item()
                        else:
                            loss_val_item = 0.0
                        val_items.append((txt, loss_val_item))
                if val_items:
                    val_losses = [l for _,l in val_items]
                    val_loss_avg = sum(val_losses)/len(val_losses)
                    # Apply same weighting as training for comparable numbers
                    # Compute current weighting factors
                    if step < WARMUP_STEPS * 2:
                        soft = 0.2 + 0.8 * (step / (WARMUP_STEPS * 2))
                        pat_w_v = 1.0 + soft * (pattern_loss_bias - 1.0)
                        zone_w_v = 1.0 + soft * (zone_bias - 1.0)
                        helper_w_v = 1.0 + soft * (helper_loss_bias - 1.0)
                        micro_w_v = 1.0 + soft * (micro_loss_bias - 1.0)
                    else:
                        pat_w_v = pattern_loss_bias
                        zone_w_v = zone_bias
                        helper_w_v = helper_loss_bias
                        micro_w_v = micro_loss_bias
                    weight_factor = pat_w_v * zone_w_v * helper_w_v * micro_w_v
                    val_loss_avg_weighted = val_loss_avg * weight_factor
                    # Smooth validation loss with EMA for stability
                    if val_loss_ema is None:
                        val_loss_ema = val_loss_avg_weighted
                    else:
                        val_loss_ema = 0.9*val_loss_ema + 0.1*val_loss_avg_weighted
                    print(f"  Val loss {val_loss_avg_weighted:.4f} ema {val_loss_ema:.4f} on {len(val_items)} samples")
                    # Kid-learning easy set gating
                    if KID_MODE:
                        kid_n = min(KID_EASY_SAMPLES, len(val_items))
                        kid_losses = [l for _,l in val_items[:kid_n]]
                        kid_easy_loss = sum(kid_losses)/len(kid_losses) if kid_losses else float('inf')
                        # Store for curriculum gating
                        prev_ema = globals().get('kid_easy_loss_ema', None)
                        if prev_ema is None:
                            globals()['kid_easy_loss_ema'] = kid_easy_loss
                        else:
                            globals()['kid_easy_loss_ema'] = 0.9*prev_ema + 0.1*kid_easy_loss
                        print(f"  [Kid] Easy loss {kid_easy_loss:.4f} ema {globals()['kid_easy_loss_ema']:.4f} on {kid_n} samples")
                    # Real-time talking check after validation
                    try:
                        model.eval()
                        prompt = SELF_TALK_PROMPT
                        # Try to get EOS id for clean stopping
                        try:
                            eos_id = tok.bpe.inverse_vocab.get('<|endoftext|>', None)
                        except Exception:
                            eos_id = None
                        ids = tok.encode(prompt)
                        if len(ids) > max_seq_len:
                            ids = ids[-max_seq_len:]
                        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                        generated = ids.copy()
                        with torch.no_grad():
                            for _ in range(48):
                                inp = torch.tensor(generated[-max_seq_len:], dtype=torch.long, device=device).unsqueeze(0)
                                logits = model(inp)
                                if isinstance(logits, tuple):
                                    logits = logits[0]
                                logits = torch.clamp(logits, -20.0, 20.0)
                                # Mask PAD token 0 to avoid space domination
                                logits[0, -1, 0] = -1e9
                                if TALK_GREEDY:
                                    next_id = torch.argmax(logits[0,-1]).item()
                                else:
                                    temp = 0.8
                                    logits_temp = logits[0,-1] / temp
                                    probs = torch.softmax(logits_temp, dim=-1)
                                    top_k = 40
                                    k = min(top_k, probs.size(-1))
                                    vals, idx = torch.topk(probs, k)
                                    # Simple repetition penalty
                                    for i in range(k):
                                        tok_id = idx[i].item()
                                        if tok_id in generated[-10:]:
                                            vals[i] = vals[i] * 0.7
                                    vals = vals / vals.sum()
                                    next_token = torch.multinomial(vals, 1).item()
                                    next_id = idx[next_token].item()
                                generated.append(next_id)
                                if next_id == 0 or (eos_id is not None and next_id == eos_id):
                                    break
                        text = tok.decode(generated)
                        # Sanitize output to printable ASCII for readability
                        text_raw = text
                        text = ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in text)
                        # Diagnostics
                        # Show repr to reveal hidden whitespace/pad
                        repr_text = repr(text_raw[:160])
                        # Simple coherence heuristic
                        tokens = generated
                        if len(tokens) >= 3:
                            trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
                            uniq_ratio = len(set(trigrams)) / max(1, len(trigrams))
                        else:
                            uniq_ratio = 0.0
                        # Print diagnostics
                        print(f"[TalkCheck] step {step} prompt='{prompt}' uniq3={uniq_ratio:.3f} -> {text[:160]}")
                        print(f"[TalkCheck] repr={repr_text}")
                        if uniq_ratio > 0.6 and val_loss_ema < 6.0:
                            print(f"[TalkCheck] *** Model appears to be speaking at step {step} ***")
                    finally:
                        model.train()
                    print(f"[Train] Step {step} TalkCheck complete, continuing...", flush=True)
                    # Pattern-based predictor & convergence predictor
                    # Compute average pattern strength on validation set
                    if pattern_weights:
                        def pattern_score(txt):
                            words = txt.split()
                            if len(words) < 2:
                                return 0.0
                            s = 0.0
                            c = 0
                            for n in (2,3,4):
                                if len(words) < n: continue
                                for i in range(len(words)-n+1):
                                    key = f"{n}:{' '.join(words[i:i+n])}"
                                    w = pattern_weights.get(key, 0.0)
                                    s += abs(w)
                                    c += 1
                            return s / max(1,c)
                        scores = [pattern_score(txt) for txt,_ in val_items]
                        avg_pat_score = sum(scores)/len(scores) if scores else 0.0
                        # Convergence predictor: slope of recent val_loss_ema
                        # Simple EMA of slope
                        if not hasattr(main, 'val_ema_hist'):
                            main.val_ema_hist = []
                        main.val_ema_hist.append(val_loss_ema)
                        if len(main.val_ema_hist) > 5:
                            main.val_ema_hist.pop(0)
                        if len(main.val_ema_hist) >= 2:
                            slope = main.val_ema_hist[-1] - main.val_ema_hist[0]
                            # If loss is decreasing, allow slightly higher pattern bias
                            if slope < -0.01:
                                pattern_loss_bias = min(1.5, pattern_loss_bias + 0.005)
                                helper_loss_bias = min(1.5, helper_loss_bias + 0.003)
                                micro_loss_bias = min(1.5, micro_loss_bias + 0.002)
                            elif slope > 0.01:
                                pattern_loss_bias = max(0.8, pattern_loss_bias - 0.005)
                                helper_loss_bias = max(0.8, helper_loss_bias - 0.003)
                                micro_loss_bias = max(0.8, micro_loss_bias - 0.002)
                        # Modulate bias by pattern strength
                        if avg_pat_score > 0.1:
                            pattern_loss_bias = min(1.5, pattern_loss_bias + 0.01 * avg_pat_score)
                            helper_loss_bias = min(1.5, helper_loss_bias + 0.005 * avg_pat_score)
                            micro_loss_bias = min(1.5, micro_loss_bias + 0.003 * avg_pat_score)
                        elif avg_pat_score < 0.02:
                            pattern_loss_bias = max(0.8, pattern_loss_bias - 0.005)
                            helper_loss_bias = max(0.8, helper_loss_bias - 0.003)
                            micro_loss_bias = max(0.8, micro_loss_bias - 0.002)
                    # Immediate best-weight restore with tolerance and hysteresis
                    # Increased tolerance and evals to avoid noisy rollback
                    # Cooldown to prevent rapid re-trigger after restore
                    if restore_cooldown_steps > 0:
                        restore_cooldown_steps -= 1
                    if best_val_loss != float('inf') and val_loss_ema > best_val_loss + 0.10:
                        val_regress_counter += 1
                        if val_regress_counter >= 5 and restore_cooldown_steps == 0:
                            try:
                                best_ckpt = torch.load(best_ckpt_path, map_location=device)
                                model.load_state_dict(best_ckpt['model'], strict=False)
                                if ema_model is not None and 'ema_model' in best_ckpt and best_ckpt['ema_model'] is not None:
                                    ema_model.load_state_dict(best_ckpt['ema_model'], strict=False)
                                print(f"[Best] Validation regressed for 5 evals, restored best weights from step {best_val_step}. val_loss_ema={val_loss_ema:.4f} best={best_val_loss:.4f}")
                                val_regress_counter = 0
                                restore_cooldown_steps = 5000  # ~50 evals cooldown
                            except Exception:
                                pass
                    else:
                        val_regress_counter = 0
                    # Pattern mapper with layered tree, fixed-size nodes, reward/punishment weights
                    if os.environ.get('DISABLE_PATTERN_UPDATES','1') != '1' and pattern_tree is not None:
                        try:
                            # Reward for low loss, punishment for high loss
                            # Entropy-based sampling: sample more items for diversity
                            k = 10
                            top_items = sorted(val_items, key=lambda x: x[1])[:k]
                            bad_items = sorted(val_items, key=lambda x: x[1], reverse=True)[:k]
                            # Also sample mid-loss items for exploration
                            if len(val_items) > 2*k:
                                mid_items = sorted(val_items, key=lambda x: abs(x[1] - sum(l for _,l in val_items)/len(val_items)))[:k]
                            else:
                                mid_items = []
                            # Pattern replay removed per user request - no replay buffer updates
                            updated_weights = []
                            # Update weights in memory and persist to layered tree
                            # Multi-scale n-grams for richer pattern capture
                            ngram_sizes = [2,3,4]
                            def process_ngrams(txt, loss_v, reward_val, entry_type):
                                nonlocal updated_weights
                                words = txt.split()
                                if len(words) < 2:
                                    return
                                for n in ngram_sizes:
                                    if len(words) < n:
                                        continue
                                    for i in range(len(words)-n+1):
                                        ngram = ' '.join(words[i:i+n])
                                        # Prefix with n-gram size for separate heads
                                        key = f"{n}:{ngram}"
                                        prev = pattern_weights.get(key, 0.0)
                                        new_weight = 0.9*prev + 0.1*reward_val
                                        pattern_weights[key] = new_weight
                                        updated_weights.append(new_weight)
                                        pattern_tree.update_weight(key, reward_val, alpha=0.9)
                                        pattern_tree.add_pattern(key, new_weight, loss_v, step, entry_type=entry_type)
                            for txt, loss_v in top_items:
                                reward = max(-1.0, min(1.0, 1.0 - loss_v/10.0))
                                process_ngrams(txt, loss_v, reward, "reward")
                            # Punishment for bad patterns
                            for txt, loss_v in bad_items:
                                punishment = max(-1.0, min(0.0, -0.5 - loss_v/10.0))
                                process_ngrams(txt, loss_v, punishment, "punish")
                            # Update pattern loss bias for training weighting
                            if updated_weights:
                                import numpy as np
                                mean_w = float(np.mean(updated_weights))
                                # Use positive/negative reinforcement for loss scaling
                                bias_factor = max(-0.5, min(0.5, mean_w))
                                # EMA smooth the bias, allow decrease/increase
                                pattern_loss_bias = 0.9*pattern_loss_bias + 0.1*(1.0 + 0.3*bias_factor)
                                pattern_loss_bias = max(0.8, min(1.5, pattern_loss_bias))
                                # Meta-reasoning: compose patterns from reused ones to build hierarchy
                                if step % 1000 == 0:
                                    # Get top weighted patterns
                                    top_items = sorted(pattern_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
                                    # Create pairwise meta-patterns
                                    for i in range(len(top_items)):
                                        for j in range(i+1, len(top_items)):
                                            k1, w1 = top_items[i]
                                            k2, w2 = top_items[j]
                                            # Skip self combos and very long keys
                                            if k1 == k2:
                                                continue
                                            combo_key = f"meta:{k1}||{k2}"
                                            # Compose weight as blend of parents
                                            combo_w = 0.5 * (abs(w1) + abs(w2)) * 0.9
                                            prev = pattern_weights.get(combo_key, 0.0)
                                            new_w = 0.9*prev + 0.1*combo_w
                                            pattern_weights[combo_key] = new_w
                                            try:
                                                pattern_tree.update_weight(combo_key, combo_w, alpha=0.9)
                                                pattern_tree.add_pattern(combo_key, new_w, loss_v, step, entry_type="meta")
                                            except Exception:
                                                pass
                            # Periodic weight persistence
                            if step % 5000 == 0:
                                pattern_tree.save_weights()
                                # Sync helper/micro trees from main pattern weights every 5k steps
                                try:
                                    import json
                                    main_pw = BASE / "runtime" / "memory" / "pattern_tree" / "patterns" / "pattern_weights.json"
                                    if main_pw.exists():
                                        with open(main_pw, 'r', encoding='utf-8') as f:
                                            main_weights = json.load(f)
                                        # Update helper weights
                                        if helper_pattern_tree is not None:
                                            # Keep top patterns
                                            for k,v in list(main_weights.items())[:5000]:
                                                helper_weights[k] = float(v)
                                                helper_pattern_tree.weights[k] = float(v)
                                            # Persist helper weights via tree
                                            helper_pattern_tree._save_weights()
                                            # Also write dict backup
                                            hw_path = BASE / "runtime" / "memory" / "pattern_tree" / "helper" / "helper_patterns" / "pattern_weights.json"
                                            with open(hw_path, 'w', encoding='utf-8') as f:
                                                json.dump(helper_pattern_tree.weights, f)
                                        # Update micro weights
                                        if micro_pattern_tree is not None:
                                            for k,v in list(main_weights.items())[:2000]:
                                                micro_weights[k] = float(v)
                                                micro_pattern_tree.weights[k] = float(v)
                                            micro_pattern_tree._save_weights()
                                            mw_path = BASE / "runtime" / "memory" / "pattern_tree" / "micro" / "micro_patterns" / "pattern_weights.json"
                                            with open(mw_path, 'w', encoding='utf-8') as f:
                                                json.dump(micro_pattern_tree.weights, f)
                                except Exception:
                                    pass
                        except Exception as e:
                            print(f"[PatternTree] update failed: {e}")
                    # Convergence map reward logging for cold/hot area tracking
                    if CONVERGENCE_AVAILABLE and cm_add_reward:
                        try:
                            # Reward is higher when loss is lower
                            reward = max(-1.0, min(1.0, 1.0 - val_loss_avg/10.0))
                            # Log a few validation examples
                            for vtxt in val_texts[:min(5, VAL_SAMPLES)]:
                                # Use text as both query and answer for simple reward signal
                                cm_add_reward(vtxt[:300], vtxt[-300:], reward)
                        except Exception:
                            pass
                    # Best checkpoint tracking and LR scheduling
                    if OVERFIT_GUARD:
                        improved = False
                        if val_loss_ema < best_val_loss - 1e-4:
                            best_val_loss = val_loss_avg
                            best_val_step = step
                            improved = True
                            no_improve_steps = 0
                            val_rise_counter = 0
                            # Save best checkpoint
                            try:
                                best_data = {'model':model.state_dict(),'ema_model':ema_model.state_dict() if ema_model is not None else None,'step':step,'ema_loss':ema_loss,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'max_seq_len':max_seq_len,'batch_size':batch_size,'loss_target':loss_target}
                                torch.save(best_data, best_ckpt_path)
                                print(f"[Best] New best val loss {best_val_loss:.4f} at step {step}, saved best checkpoint")
                                # Emergence map hook – read last emergence score from logs, no orchestrator
                                if EMERGENCE_AVAILABLE and step % 50000 == 0:
                                    try:
                                        import json
                                        with open(EMERGENCE_LOG, 'r', encoding='utf-8') as f:
                                            # read last non-empty line
                                            lines = f.readlines()
                                            if lines:
                                                last = json.loads(lines[-1].strip())
                                                percent = float(last.get('percent', 0.0))
                                                # Map percent to bias adjustment
                                                if percent > 50:
                                                    delta = 0.02 * (percent / 100.0)
                                                    pattern_loss_bias = min(1.5, pattern_loss_bias + delta)
                                                    print(f"[Emergence] Log score {percent:.1f}%, pattern_bias -> {pattern_loss_bias:.3f}")
                                    except Exception:
                                        pass
                            except Exception as e:
                                print(f"[Best] Save failed: {e}")
                        else:
                            no_improve_steps += 100
                            val_rise_counter += 1
                        # Reduce LR if validation rises for 5 evaluations
                        if val_rise_counter >= 5:
                            # Reduce base LR by 25% with floor for smoother decay, warm restart if too low
                            new_lr = base_lr * 0.75
                            if new_lr < 1e-5:
                                # Warm restart to recover learning with floor
                                restart_lr = 3e-4 * (0.5 ** min(lr_reductions, 10))
                                base_lr = max(1e-5, restart_lr)
                                print(f"[LR] Warm restart triggered, base LR reset to {base_lr:.6f}")
                            else:
                                base_lr = max(new_lr, 1e-5)
                                print(f"[LR] Validation rising for {val_rise_counter} evals, base LR reduced to {base_lr:.6f}")
                            val_rise_counter = 0
                            lr_reductions += 1
                        # Adaptive heads: adjust if plateauing
                        if no_improve_steps >= patience // 2 and HEADS < 16:
                            # Try increase heads if DIM divisible
                            new_heads = HEADS + 1
                            # Find next divisor of DIM >= new_heads
                            candidates = [h for h in range(new_heads, 17) if DIM % h == 0]
                            if candidates:
                                chosen_heads = candidates[0]
                                if chosen_heads != HEADS:
                                    print(f"[AdaptiveHeads] Plateau detected, increasing heads {HEADS} -> {chosen_heads}")
                                    HEADS = chosen_heads
                                    # Rebuild model with new heads, keep weights where possible
                                    try:
                                        old_state = model.state_dict()
                                        model = DualWeightWeaveFormer(dim=DIM,num_layers=32,vocab_size=vocab_size,use_dual_weights=True,nhead=HEADS).to(device)
                                        model.load_state_dict(old_state, strict=False)
                                        if ema_model is not None:
                                            ema_old = ema_model.state_dict()
                                            ema_model = copy.deepcopy(model)
                                            for p in ema_model.parameters():
                                                p.requires_grad_(False)
                                            ema_model.load_state_dict(ema_old, strict=False)
                                            ema_model.eval()
                                        # Reset optimizer to match new params
                                        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.001)
                                        print(f"[AdaptiveHeads] Model rebuilt with {HEADS} heads")
                                        no_improve_steps = 0
                                    except Exception as e:
                                        print(f"[AdaptiveHeads] Rebuild failed: {e}")
                        # Patience stop
                        if no_improve_steps >= patience:
                            print(f"[Overfit Guard] No improvement for {patience} steps, best={best_val_loss:.4f} at step {best_val_step}")
                            # Restore best checkpoint before exit
                            try:
                                if best_ckpt_path.exists():
                                    best_ckpt = torch.load(best_ckpt_path, map_location=device)
                                    model.load_state_dict(best_ckpt['model'], strict=False)
                                    if ema_model is not None and 'ema_model' in best_ckpt and best_ckpt['ema_model'] is not None:
                                        ema_model.load_state_dict(best_ckpt['ema_model'], strict=False)
                                    print(f"[Best] Restored best checkpoint from step {best_val_step}")
                            except Exception as e:
                                print(f"[Best] Restore failed: {e}")
                            break
                model.train()
            # periodic save every SAVE_EVERY steps
            if step % SAVE_EVERY == 0 and step > 0:
                models_dir = BASE / "runtime" / "models"
                ckpt_path = models_dir / "weave_decoder_main.pt"
                step_ckpt_path = models_dir / f"weave_decoder_step{step}.pt"
                ckpt_data = {'model':model.state_dict(),'ema_model':ema_model.state_dict() if ema_model is not None else None,'step':step,'ema_loss':ema_loss,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'max_seq_len':max_seq_len,'batch_size':batch_size,'loss_target':loss_target,'pattern_loss_bias':pattern_loss_bias,'zone_bias':zone_bias,'helper_loss_bias':helper_loss_bias,'micro_loss_bias':micro_loss_bias}
                torch.save(ckpt_data, step_ckpt_path)
                # keep symlink / copy as main
                import shutil
                shutil.copy2(step_ckpt_path, ckpt_path)
                print(f"[Save] Checkpoint saved at step {step} to weave_decoder_step{step}.pt")
                # checkpoint rotation
                if CHECKPOINT_KEEP > 0:
                    try:
                        step_files = sorted(models_dir.glob("weave_decoder_step*.pt"), key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1)) if re.search(r'(\d+)$', p.stem) else 0)
                        # keep newest CHECKPOINT_KEEP
                        for old in step_files[:-CHECKPOINT_KEEP]:
                            try:
                                old.unlink()
                            except Exception:
                                pass
                    except Exception:
                        pass
                # Memory orchestrator periodic save
                if memory_orch is not None:
                    try:
                        memory_orch.save()
                        print(f"[Memory] Saved memory orchestrator at step {step}")
                    except Exception as e:
                        print(f"[Memory] save skipped: {e}")
            # codebook / long-term save - disabled, save only at end
            # if step % 25000 == 0 and step > 0:
            #     try:
            #         from infrastructure.true_hybrid.codebook_checkpoint import save_model_codebook
            #         cb_dir = BASE / "runtime" / "models" / "codebook_checkpoints"
            #         save_model_codebook(model, cb_dir, base_name=f"weave_decoder_cb_step{step}", extra_state={'step':step,'ema_loss':ema_loss})
            #         print(f"[Codebook] Saved codebook checkpoint at step {step}")
            #     except Exception as e:
            #         print(f"[Codebook] save skipped: {e}")
                # Memory orchestrator periodic save - disabled by default for speed
                # Uncomment to enable
                # if memory_orch is not None and step % 25000 == 0:
                #     try:
                #         memory_orch.save()
                #         print(f"[Memory] Saved memory orchestrator at step {step}")
                #     except Exception as e:
                #         print(f"[Memory] save skipped: {e}")
        prev_loss_val = ema_loss
    # Final save at end of training - restore best checkpoint if available
    try:
        if best_val_step >= 0 and best_ckpt_path.exists():
            # Restore best model for final output
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_ckpt['model'], strict=False)
            if ema_model is not None:
                ema_best_path = BASE / "runtime" / "models" / "weave_decoder_ema_best.pt"
                if ema_best_path.exists():
                    ema_ckpt = torch.load(ema_best_path, map_location=device)
                    ema_model.load_state_dict(ema_ckpt['model'], strict=False)
            print(f"[Best] Restored best checkpoint from step {best_val_step} for final save")
        elif best_val_step < 0:
            print(f"[Best] No best checkpoint found in this run, keeping final weights")
        final_ckpt = BASE / "runtime" / "models" / "weave_decoder_main.pt"
        torch.save({'model':model.state_dict(),'ema_model':ema_model.state_dict() if ema_model is not None else None,'step':steps-1,'ema_loss':ema_loss,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'max_seq_len':max_seq_len,'batch_size':batch_size,'loss_target':loss_target}, final_ckpt)
        print(f"Final model saved to {final_ckpt}")
        # Final memory orchestrator save
        if memory_orch is not None:
            try:
                memory_orch.save()
                print(f"[Memory] Final memory orchestrator saved")
            except Exception as e:
                print(f"[Memory] final save skipped: {e}")
    except Exception as e:
        print(f"Final save failed: {e}")
    print("Training complete")

import json
main()
