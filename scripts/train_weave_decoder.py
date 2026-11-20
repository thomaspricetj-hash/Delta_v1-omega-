"""
Train WeaveFormer decoder for autoregressive generation.
Uses training data from trainingData/*.parquet
"""
import os
# GPU-only: limit CPU parallelism
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
import torch
import torch.nn as nn
# GPU-only mode: minimize CPU threads to avoid CPU work
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
print(f"PyTorch threads: {torch.get_num_threads()} interop: {torch.get_num_interop_threads()}")
import torch.nn.utils.rnn as rnn_utils
import random
import math
from infrastructure.true_hybrid.weaveformer import WeaveFormer
try:
    from infrastructure.true_hybrid.dual_weight_layers import DualWeightWeaveFormer
    DUAL_WEIGHT_AVAILABLE = True
except Exception:
    DUAL_WEIGHT_AVAILABLE = False
try:
    from infrastructure.true_hybrid.layered_codebook_dual import CodebookDualWeightCompressor
    from infrastructure.true_hybrid.codebook_checkpoint import save_model_codebook, load_model_codebook
    CODEBOOK_AVAILABLE = True
except Exception:
    CODEBOOK_AVAILABLE = False
    save_model_codebook = None
    load_model_codebook = None
from infrastructure.tokenizer.codebook_bpe_tokenizer import get_codebook_tokenizer
try:
    from infrastructure.hscl.hscl_preprocessor import compress as hscl_compress
except Exception:
    def hscl_compress(t): return t
try:
    from infrastructure.true_hybrid.token_codebook_tokenizer import HierarchicalTokenCodebook
    TOKEN_CODEBOOK_AVAILABLE = True
except Exception:
    TOKEN_CODEBOOK_AVAILABLE = False
    HierarchicalTokenCodebook = None

# BitDrop v2 compressed checkpoint helpers
try:
    import bitdrop_v2 as _bd2
    import io
    _BD2_AVAILABLE = hasattr(_bd2, 'compress_adaptive_py') and hasattr(_bd2, 'decompress_adaptive_py')
    if _BD2_AVAILABLE:
        def save_compressed(path, obj):
            buf = io.BytesIO()
            torch.save(obj, buf)
            data = buf.getvalue()
            comp = _bd2.compress_adaptive_py(data)
            with open(path.with_suffix('.pt.bd2'), 'wb') as f:
                f.write(comp)
        def load_compressed(path):
            data = open(path.with_suffix('.pt.bd2'), 'rb').read()
            dec = _bd2.decompress_adaptive_py(data)
            buf = io.BytesIO(dec)
            return torch.load(buf, map_location='cpu')
    else:
        raise AttributeError("bitdrop_v2 missing compress functions")
except Exception:
    _BD2_AVAILABLE = False
    def save_compressed(path, obj):
        torch.save(obj, path)
    def load_compressed(path):
        return torch.load(path, map_location='cpu')

try:
    from infrastructure.mbfn.store import FrameStore
    from infrastructure.mbfn.bridge import MBFNBridge
    MBFN_AVAILABLE = True
except Exception:
    MBFN_AVAILABLE = False
    FrameStore = None
    MBFNBridge = None

def load_texts(data_dir):
    import pandas as pd, glob
    texts = []
    for f in glob.glob(str(Path(data_dir)/"*.parquet")):
        try:
            df = pd.read_parquet(f)
            for col in df.columns:
                texts.extend(df[col].dropna().astype(str).tolist())
        except Exception:
            pass
    return texts

def train(reward_window=100, reward_interval=100, reward_scale=1.0, force_codebook_resume=False):
    import os
    # GPU-only mode
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available - trainer is GPU-only')
    device = torch.device('cuda')
    # Schrödinger-inspired quantum annealing for exploration
    try:
        from infrastructure.quantum.schrodinger_sampler import (
            schrodinger_anneal_schedule,
            quantum_rotation_prob,
            batch_collapse_sample,
        )
        QUANTUM_AVAILABLE = True
    except Exception:
        QUANTUM_AVAILABLE = False
        schrodinger_anneal_schedule = None
        quantum_rotation_prob = None
        batch_collapse_sample = None
    print(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    tok = get_codebook_tokenizer()
    # Load a small sample to size vocab for codebook before model init
    data_dir = BASE / "trainingData"
    def _quick_load_texts(d):
        import os
        texts = []
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith('.txt'):
                    try:
                        with open(os.path.join(root,f), 'r', encoding='utf-8', errors='ignore') as fh:
                            texts.extend([l.strip() for l in fh if l.strip()])
                    except Exception:
                        pass
        return texts
    quick_texts = _quick_load_texts(data_dir)
    # Build codebook early so vocab size accounts for codebook entries
    if hasattr(tok, 'codebook'):
        try:
            sample = quick_texts[:min(50000, len(quick_texts))]
            if sample:
                tok.build_codebook(sample, min_count=5)
                print(f"Hierarchical token codebook pre-built with {len(tok.codebook.text_to_ref)} entries")
        except Exception as e:
            print(f"Token codebook pre-build failed: {e}")
    vocab = getattr(tok, 'bpe').vocab
    bpe_size = len(vocab)
    codebook_entries = len(getattr(tok.codebook, 'text_to_ref', {}))
    # Reserve space for codebook ids; cap at 1M for safety
    vocab_size = min(1_048_576, max(8192, bpe_size + codebook_entries))
    if DUAL_WEIGHT_AVAILABLE:
        # Use dual-weight 32-layer model for master+delta learning
        model = DualWeightWeaveFormer(dim=256, num_layers=32, vocab_size=vocab_size, use_dual_weights=True)
        print("Using DualWeightWeaveFormer with master+delta weights")
    else:
        model = WeaveFormer(dim=256, num_layers=36, vocab_size=vocab_size)
    model = model.to(device)
    # Disable predictor cache during training to avoid stale gradients; enable for inference
    if hasattr(model, 'disable_predictor_mode'):
        model.disable_predictor_mode()
    # EMA model for stable evaluation and checkpoint averaging
    ema_decay = 0.999
    ema_model = None
    try:
        import copy
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()
        print(f"EMA model initialized with decay {ema_decay}")
    except Exception as e:
        ema_model = None
        print(f"EMA init skipped: {e}")
    # Compile model for faster forward/backward graph execution
    # Disabled on Windows due to missing MSVC compiler (cl) for inductor
    import platform
    if platform.system() != "Windows":
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            print(f"torch.compile not available: {e}")
    else:
        print("torch.compile disabled on Windows (MSVC not available)")
    # Use Adam optimizer for adaptive learning rates and stability
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)  # Adam for production-grade training
    # Optimizer toggle: Adam <-> AdamW to get benefits of both
    optimizer_toggle_interval = 5000  # steps
    use_adamw = False
    next_toggle_step = optimizer_toggle_interval
    # Helper to switch optimizer while preserving momentum buffers
    def _make_optimizer(use_adamw_flag):
        # Preserve param groups lr/weight_decay
        pg = optimizer.param_groups[0]
        lr = pg.get('lr', 3e-4)
        wd = pg.get('weight_decay', 0.01)
        # Save old state
        old_state = optimizer.state_dict()
        # Create new optimizer
        if use_adamw_flag:
            new_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)
        else:
            new_opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)
        # Transfer momentum buffers where possible
        try:
            # Map parameter IDs
            old_id_to_new = {}
            for old_p, new_p in zip(optimizer.param_groups[0]['params'], new_opt.param_groups[0]['params']):
                old_id_to_new[id(old_p)] = new_p
            # Build new state dict
            new_state = new_opt.state_dict()
            new_state['state'] = {}
            for pid, s in old_state['state'].items():
                # Find matching param in new optimizer
                # param ids are not directly comparable across instances, so match by index
                # Simple fallback: copy by order
                pass
            # Fallback: keep optimizer fresh to avoid mismatch
        except Exception:
            pass
        return new_opt
    # Gradient accumulation to simulate larger batch
    grad_accum_steps = 4
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else torch.amp.GradScaler('cpu')
    data_dir = BASE / "trainingData"
    texts = load_texts(data_dir)
    def clean_text(s):
        return ''.join(ch if ord(ch) < 128 else ' ' for ch in s)
    # Sanitize corpus with Z-score + Isolation Forest
    try:
        from infrastructure.ml_extensions.data_sanitizer import DataSanitizer
        sanitizer = DataSanitizer(z_thresh=3.0, iforest_contamination=0.01)
        # Simple length-based feature for sanitization
        feats = [[len(t), len(t.split())] for t in texts]
        import numpy as np
        X = np.array(feats, dtype=np.float32)
        # Fit sanitizer before predicting to avoid StandardScaler not fitted error
        sanitizer.fit(X)
        mask = sanitizer.predict(X)
        texts = [t for t, keep in zip(texts, mask) if keep]
        print(f"Sanitized corpus: {len(texts)} texts retained")
    except Exception as e:
        print(f"Sanitizer skipped: {e}")
    # Shuffle corpus to avoid distribution mismatch between train/val
    random.shuffle(texts)
    val_size = max(100, len(texts) // 100)
    val_texts = texts[:val_size]
    train_texts = texts[val_size:]
    print(f"Loaded {len(texts)} texts, validation {len(val_texts)}")
    # Initialize hierarchical token codebook for input compression via tokenizer
    if hasattr(tok, 'codebook'):
        try:
            sample_size = min(50000, len(texts))
            sample_texts = random.sample(texts, sample_size)
            tok.build_codebook(sample_texts, min_count=5)
            print(f"Hierarchical token codebook initialized with {len(tok.codebook.text_to_ref)} entries")
        except Exception as e:
            print(f"Token codebook init failed: {e}")
    # Initialize window indexer for production sampling
    from infrastructure.indexing.window_indexer import WindowIndexer
    index_base = BASE / "runtime" / "index"
    indexer = WindowIndexer(index_base, scales=[128,256,512,1024])
    # Guard against huge ledger starving RAM/CPU; disable indexer if ledger too large
    # Disable by default for GPU-only DataLoader training – window index building is CPU heavy
    disable_indexer = True
    try:
        ledger_size = indexer.ledger_path.stat().st_size if indexer.ledger_path.exists() else 0
        if ledger_size > 500 * 1024 * 1024:
            print(f"[WARN] Window ledger {ledger_size/1024/1024:.1f} MB too large, disabling indexer to avoid GPU starvation")
            disable_indexer = True
    except Exception:
        pass
    if not disable_indexer and not indexer.ledger_path.exists():
        print("Building window index from corpus...")
        indexer.build_from_texts(texts, tok)
        print("Window index built")
    # Initialize unified AI memory orchestrator
    try:
        from infrastructure.memory.ai_memory_orchestrator import create_orchestrator
        memory_root = BASE / "runtime" / "memory"
        memory_root.mkdir(parents=True, exist_ok=True)
        memory_orch = create_orchestrator(dim=256, memory_root=str(memory_root))
        print(f"AIMemoryOrchestrator initialized with {memory_orch.health_check()['vectors']} vectors")
        # Compatibility aliases for existing code
        vector_mem = memory_orch.vector_mem
        bio_mem = memory_orch.bio_mem
        reasoning_mem = memory_orch.reasoning_mem
        # Inject heatmap into MDWE projector for spatial gating
        try:
            if hasattr(memory_orch, 'heatmap') and memory_orch.heatmap is not None:
                model.set_heatmap(memory_orch.heatmap)
        except Exception:
            pass
        # Inject embedding heatmap for learning modulation
        try:
            from infrastructure.embeddings.embeddings import store as embed_store
            # Build a simple heat vector from embedding heat scores
            # Normalize to [0,1] for MDWE gating
            heat_vals = list(embed_store._heat.values())
            if heat_vals:
                import numpy as np
                heat_arr = np.array(heat_vals, dtype=np.float32)
                # Pad/truncate to model dim for compatibility
                heat_vec = np.zeros(256, dtype=np.float32)
                n = min(len(heat_arr), 256)
                heat_vec[:n] = heat_arr[:n]
                # Update model heatmap with embedding heat
                model.set_heatmap(heat_vec)
        except Exception:
            pass
    except Exception as e:
        vector_mem = None
        bio_mem = None
        reasoning_mem = None
        memory_orch = None
        print(f"Memory init skipped: {e}")
    # Initialize SyntheticMind DAX Tile Engine
    try:
        from infrastructure.syntheticmind_tile import SMTile
        sm_tile = SMTile(sram_size=4096, num_warps=32, warp_size=4)
    except Exception as e_tile:
        sm_tile = None
        print(f"SMTile init skipped: {e_tile}")
    # Initialize Pattern Identifier for training speed-up
    try:
        from infrastructure.pattern.pattern_identifier import PatternIdentifier
        pattern_id = PatternIdentifier(ngram_sizes=(3,4,5,8), decay=0.999)
    except Exception as e_pat:
        pattern_id = None
        print(f"PatternIdentifier init skipped: {e_pat}")
    # Initialize Integrated Cognitive Core - Structural Learning + MDWE + DAX Registry
    try:
        from infrastructure.integration.wire_all import create_integrated_core
        integrated_core = create_integrated_core(base_dir=BASE / "runtime")
        print(f"Integrated Cognitive Core initialized: {integrated_core.health_check()}")
    except Exception as e_core:
        integrated_core = None
        print(f"Integrated Cognitive Core init skipped: {e_core}")
    # Initialize Master Orchestrator for runtime wiring
    try:
        from orchestrator.master_orchestrator_v0 import teach_full, run_orchestrator
        master_orch_available = True
    except Exception as e_mo:
        teach_full = None
        run_orchestrator = None
        master_orch_available = False
        print(f"Master Orchestrator init skipped: {e_mo}")
    # Initialize Foldable Map logging for predictor promotion
    try:
        from infrastructure.foldable_map.multilayered_foldable_map import log_answer as foldable_log
        foldable_logging_available = True
    except Exception as e_fm:
        foldable_log = None
        foldable_logging_available = False
        print(f"Foldable Map init skipped: {e_fm}")
    buffer = train_texts[:500000]
    random.seed(42)
    random.shuffle(buffer)
    steps = 2000000
    # MBFN initialization for explainable reasoning
    mbfn_store = None
    mbfn_bridge = None
    if MBFN_AVAILABLE:
        mbfn_store = FrameStore()
        hscl_dict_path = BASE / "infrastructure" / "hscl" / "hscl_numeric_dict.json"
        if hscl_dict_path.exists():
            mbfn_store.build_from_hscl(hscl_dict_path)
        mbfn_bridge = MBFNBridge(mbfn_store, alpha=0.7)
    # Loss-gated curriculum for sliding-window training
    # Scale batch size up on GPU for faster learning
    base_batch = 8 if device.type == 'cuda' else 2
    max_seq_len = 128
    batch_size = base_batch
    loss_target = 1.0
    # DataLoader for GPU-friendly prefetch with pin_memory
    try:
        from torch.utils.data import DataLoader
        from infrastructure.data.codebook_dataset import CodebookTextDataset, collate_codebook_batch
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 4
        train_dataset = CodebookTextDataset(train_texts, tokenizer=tok)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=lambda batch: collate_codebook_batch(batch, max_seq_len=max_seq_len),
        )
        print(f"DataLoader created with num_workers={num_workers}, pin_memory=True")
        # Initialize persistent iterator for GPU-friendly batching
        try:
            train_loader_iter = iter(train_loader)
        except Exception:
            train_loader_iter = None
    except Exception as e:
        print(f"DataLoader init failed: {e}")
        train_loader = None
        train_loader_iter = None
    # Curriculum levels: (seq_len, batch_size, loss_target)
    curriculum_levels = [
        (128, base_batch, 1.0),
        (256, max(4, base_batch), 0.9),
        (512, max(2, base_batch//2), 0.8),
        (1024, max(1, base_batch//4), 0.7),
    ]
    # LQR-style loss-velocity LR controller with auto-tuner
    ema_loss = None
    alpha = 0.98
    kp = 0.2
    ki = 0.005
    # Reward signal config (from args)
    # reward_window, reward_interval, reward_scale passed as function args
    integral = 0.0
    # Auto-tunable hyperparameters for fastest learning
    mse_weight = 0.001
    reward_lambda = 0.02
    entropy_reg_weight = 0.001
    rotation_explore_bias = 0.0
    grad_clip = 1.0
    target_loss_decay = 0.999
    noise_scale = 0.01
    entanglement_strength = 0.05
    # Auto-tuner state
    loss_history = []
    auto_tune_window = 5
    # LQR error correction for loss trajectory
    error_integral = 0.0
    prev_error = 0.0
    kp_err = 0.02
    ki_err = 0.001
    kd_err = 0.001
    # Speed / velocity tracking for Rubik's cube augmentation
    prev_loss_val = None
    # Hybrid augmentation policy: heuristic + Q-Learning
    try:
        from infrastructure.ml_extensions.augmentation_policy import AugmentationPolicy
        aug_policy = AugmentationPolicy()
        use_q_policy = True
    except Exception:
        aug_policy = None
        use_q_policy = False
    # Full training checkpoint for resume
    ckpt_path = BASE / "runtime" / "models" / "weave_decoder.pt"
    full_ckpt_path = BASE / "runtime" / "models" / "weave_decoder_full.pt"
    start_step = 0
    # Force codebook-only resume
    if force_codebook_resume:
        if CODEBOOK_AVAILABLE and load_model_codebook is not None:
            try:
                cb_dir = BASE / "runtime" / "models" / "codebook_checkpoints"
                if cb_dir.exists():
                    metas = sorted(cb_dir.glob("weave_decoder_cb_step*_meta.pt"))
                    if not metas:
                        metas = sorted(cb_dir.glob("weave_decoder_*_cb_meta.pt"))
                    if metas:
                        latest_meta = metas[-1]
                        base_name = latest_meta.stem.replace("_meta", "")
                        print(f"Force codebook resume from {latest_meta}")
                        _, extra_state = load_model_codebook(model, cb_dir, base_name=base_name)
                        if extra_state is not None:
                            try:
                                optimizer.load_state_dict(extra_state['optimizer'])
                                print("Optimizer state restored from codebook")
                            except Exception as oe:
                                print(f"Optimizer restore from codebook failed: {oe}")
                            start_step = extra_state.get('step', start_step)
                            ema_loss = extra_state.get('ema_loss', ema_loss)
                            integral = extra_state.get('integral', integral)
                            error_integral = extra_state.get('error_integral', error_integral)
                            prev_error = extra_state.get('prev_error', prev_error)
                            max_seq_len = extra_state.get('max_seq_len', max_seq_len)
                            batch_size = extra_state.get('batch_size', batch_size)
                            loss_target = extra_state.get('loss_target', loss_target)
                            print(f"Training state restored from codebook")
                        print(f"Resumed model weights from codebook {base_name}")
            except Exception as ce:
                print(f"Forced codebook resume failed: {ce}")
    elif full_ckpt_path.exists():
        try:
            try:
                ckpt = load_compressed(full_ckpt_path)
            except Exception:
                ckpt = torch.load(full_ckpt_path, map_location='cpu')
            # Filter state dict to avoid shape mismatches when vocab size changed
            ckpt_state = ckpt.get('model', ckpt)
            model_state = model.state_dict()
            filtered_state = {}
            mismatched = []
            for k, v in ckpt_state.items():
                if k in model_state and v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    mismatched.append(k)
            if mismatched:
                print(f"Checkpoint state dict filtered, mismatched keys: {mismatched}")
            model.load_state_dict(filtered_state, strict=False)
            # Recreate optimizer after model load to ensure param groups match current model
            # Preserve current lr/weight_decay
            pg = optimizer.param_groups[0]
            lr = pg.get('lr', 3e-4)
            wd = pg.get('weight_decay', 0.01)
            # Re-create optimizer with fresh param groups
            if use_adamw:
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9,0.999), eps=1e-8)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9,0.999), eps=1e-8)
            # Optimizer state may be from a different optimizer type; load if compatible
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                print(f"Optimizer state loaded")
            except Exception as oe:
                print(f"Optimizer state load skipped: {oe}")
            start_step = ckpt.get('step', 0)
            ema_loss = ckpt.get('ema_loss', None)
            integral = ckpt.get('integral', 0.0)
            error_integral = ckpt.get('error_integral', 0.0)
            prev_error = ckpt.get('prev_error', 0.0)
            max_seq_len = ckpt.get('max_seq_len', max_seq_len)
            batch_size = ckpt.get('batch_size', batch_size)
            loss_target = ckpt.get('loss_target', loss_target)
            print(f"Resumed from full checkpoint {full_ckpt_path} at step {start_step}")
        except Exception as e:
            print(f"Full checkpoint load failed {e}, trying model-only checkpoint")
            if ckpt_path.exists():
                try:
                    try:
                        state = load_compressed(ckpt_path)
                    except Exception:
                        state = torch.load(ckpt_path, map_location='cpu')
                    # Filter for shape mismatches
                    model_state = model.state_dict()
                    filtered_state = {}
                    for k, v in state.items():
                        if k in model_state and v.shape == model_state[k].shape:
                            filtered_state[k] = v
                    model.load_state_dict(filtered_state, strict=False)
                    print(f"Loaded model weights from {ckpt_path}")
                except Exception as e2:
                    print("Model-only checkpoint load failed", e2)
            # Try codebook resume as fallback
            if CODEBOOK_AVAILABLE and load_model_codebook is not None:
                try:
                    cb_dir = BASE / "runtime" / "models" / "codebook_checkpoints"
                    # Find latest codebook meta
                    if cb_dir.exists():
                        metas = sorted(cb_dir.glob("weave_decoder_cb_step*_meta.pt"))
                        if not metas:
                            metas = sorted(cb_dir.glob("weave_decoder_*_cb_meta.pt"))
                        if metas:
                            latest_meta = metas[-1]
                            base_name = latest_meta.stem.replace("_meta", "")
                            print(f"Attempting codebook resume from {latest_meta}")
                            _, extra_state = load_model_codebook(model, cb_dir, base_name=base_name)
                            if extra_state is not None:
                                # Restore optimizer and training state
                                try:
                                    optimizer.load_state_dict(extra_state['optimizer'])
                                    print("Optimizer state restored from codebook")
                                except Exception as oe:
                                    print(f"Optimizer restore from codebook failed: {oe}")
                                start_step = extra_state.get('step', start_step)
                                ema_loss = extra_state.get('ema_loss', ema_loss)
                                integral = extra_state.get('integral', integral)
                                error_integral = extra_state.get('error_integral', error_integral)
                                prev_error = extra_state.get('prev_error', prev_error)
                                max_seq_len = extra_state.get('max_seq_len', max_seq_len)
                                batch_size = extra_state.get('batch_size', batch_size)
                                loss_target = extra_state.get('loss_target', loss_target)
                                print(f"Training state restored from codebook")
                            # Fallback step inference
                            import re
                            m = re.search(r"step(\d+)", base_name)
                            if m and start_step == 0:
                                start_step = int(m.group(1))
                                print(f"Codebook resume step inferred as {start_step}")
                            print(f"Resumed model weights from codebook {base_name}")
                except Exception as ce:
                    print(f"Codebook resume failed: {ce}")
    for step in range(start_step, steps):
        # Optimizer toggle: Adam <-> AdamW
        if step >= next_toggle_step:
            use_adamw = not use_adamw
            # Preserve current lr / weight_decay
            pg = optimizer.param_groups[0]
            lr = pg.get('lr', 3e-4)
            wd = pg.get('weight_decay', 0.01)
            # Create new optimizer with same hyperparams
            if use_adamw:
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)
                print(f"[Optimizer Toggle] Switched to AdamW at step {step}")
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)
                print(f"[Optimizer Toggle] Switched to Adam at step {step}")
            next_toggle_step = step + optimizer_toggle_interval
        # Loss-gated curriculum advance - use EMA with a little hysteresis to avoid premature jumps
        if ema_loss is not None and ema_loss < loss_target * 1.05:
            # advance to next level if exists
            for i, (seq, bs, target) in enumerate(curriculum_levels):
                if max_seq_len == seq and batch_size == bs:
                    if i + 1 < len(curriculum_levels):
                        max_seq_len, batch_size, loss_target = curriculum_levels[i + 1]
                        print(f"Curriculum advance: seq {max_seq_len} bs {batch_size} target loss {loss_target}")
                    break
        # Enable gradient checkpointing for larger windows to save memory
        if hasattr(model, 'use_checkpoint'):
            model.use_checkpoint = max_seq_len >= 512
        # Adaptive rotation probability: heuristic + Q-Learning hybrid
        if ema_loss is None:
            rotation_prob_heuristic = 0.3
            speed_factor_heuristic = 1.0
        else:
            loss_ratio = ema_loss / (loss_target + 1e-8)
            rotation_prob_heuristic = max(0.1, min(0.8, 0.3 + 0.5 * (loss_ratio - 1.0)))
            if prev_loss_val is not None:
                velocity = abs(ema_loss - prev_loss_val)
                speed_factor_heuristic = 1.0 + min(1.0, velocity * 5.0)
            else:
                speed_factor_heuristic = 1.0
        # Q-Learning policy suggestion
        if use_q_policy and aug_policy is not None and ema_loss is not None:
            rot_q, speed_q, action_q, reward_q = aug_policy.step(ema_loss, loss_target, prev_loss_val)
            # Hybrid blend: 0.6 heuristic, 0.4 policy, clamp to valid ranges
            rotation_prob = 0.6 * rotation_prob_heuristic + 0.4 * rot_q
            rotation_prob = max(0.1, min(0.8, rotation_prob))
            speed_factor = 0.6 * speed_factor_heuristic + 0.4 * speed_q
        else:
            rotation_prob = rotation_prob_heuristic
            speed_factor = speed_factor_heuristic
        # Heatmap-guided exploration: increase rotation in cold regions
        if memory_orch is not None and hasattr(memory_orch, 'heatmap') and memory_orch.heatmap is not None:
            try:
                hm = memory_orch.heatmap.maps[0].heat
                avg_heat = float(hm.mean())
                heat_factor = 1.0 + max(0.0, 0.5 - avg_heat * 10.0)
                rotation_prob = min(0.8, rotation_prob * heat_factor)
            except Exception:
                pass
        # Auto-tuned exploration bias for faster learning
        rotation_prob = min(0.8, max(0.1, rotation_prob + rotation_explore_bias))
        # Schrödinger-inspired quantum annealing for exploration
        if QUANTUM_AVAILABLE and schrodinger_anneal_schedule is not None and quantum_rotation_prob is not None:
            temp, beta = schrodinger_anneal_schedule(step, total_steps=2000000)
            # Use loss velocity as kinetic energy proxy
            loss_velocity = abs(ema_loss - prev_loss_val) if prev_loss_val is not None else 0.0
            rotation_prob = quantum_rotation_prob(rotation_prob, loss_velocity, temp)
        stride = max_seq_len // 2
        # GPU-friendly DataLoader fast path
        skip_batch_gen = False
        input_padded = None
        target_padded = None
        if train_loader_iter is not None:
            try:
                batch = next(train_loader_iter)
                if batch is not None:
                    input_padded, target_padded = batch
                    skip_batch_gen = True
            except StopIteration:
                # reset iterator
                try:
                    train_loader_iter = iter(train_loader)
                    batch = next(train_loader_iter)
                    if batch is not None:
                        input_padded, target_padded = batch
                        skip_batch_gen = True
                except Exception:
                    pass
            except Exception:
                pass
        batch_input_ids = []
        batch_target_ids = []
        batch_texts = []
        # MTD-indexer sampling with fallback to text buffer
        use_indexer = False
        try:
            if not disable_indexer:
                windows = indexer.sample_windows(max_seq_len, batch_size)
            else:
                windows = []
            if windows:
                use_indexer = True
                # Import numpy for vector features
                import numpy as np
                for w_idx, w in enumerate(windows):
                    ids = list(w)
                    if len(ids) < max_seq_len:
                        ids = ids + [0] * (max_seq_len - len(ids))
                    window_ids_orig = ids[:max_seq_len]
                    # Clamp window ids to vocab size to avoid out-of-bounds embedding lookup
                    window_ids_orig = [i if i < vocab_size else vocab_size - 1 for i in window_ids_orig]
                    # Pattern-weighted sampling for training speed-up
                    if pattern_id is not None:
                        wgt = pattern_id.get_weight(window_ids_orig)
                        # Oversample rare patterns, undersample very common ones
                        # Accept with probability proportional to weight
                        if random.random() > min(1.0, wgt / 1.5):
                            continue
                    # Store window embedding in delta vector memory and bio-memory
                    if vector_mem is not None:
                        arr = np.asarray(window_ids_orig, dtype=np.float32)
                        feat = np.zeros(256, dtype=np.float32)
                        n = min(len(arr), 256)
                        feat[:n] = arr[:n] / 100000.0
                        feat[0] = float(len(arr))
                        feat[1] = float(arr.mean()) if n>0 else 0.0
                        feat[2] = float(arr.std()) if n>1 else 0.0
                        vid = f"win_{step}_{w_idx}"
                        if vid not in vector_mem.store.vectors:
                            vector_mem.add_memory(vid, feat)
                        else:
                            delta = np.random.randn(256).astype(np.float32) * 1e-4
                            vector_mem.update_memory(vid, delta)
                        if w_idx > 0:
                            prev_vid = f"win_{step}_{w_idx-1}"
                            vector_mem.link_memories(prev_vid, vid, weight=0.9, layer=2)
                    # Bio-memory: Heart rhythmic + Muscle procedural consolidation
                    if bio_mem is not None:
                        arr = np.asarray(window_ids_orig, dtype=np.float32)
                        feat = np.zeros(256, dtype=np.float32)
                        n = min(len(arr), 256)
                        feat[:n] = arr[:n] / 100000.0
                        bio_vid = f"win_{step}_{w_idx}"
                        bio_mem.store_experience(bio_vid, feat, list(window_ids_orig), context="training_window")
                        # Pattern Identifier update for training speed-up
                        if pattern_id is not None:
                            pattern_id.update(window_ids_orig, step)
                        # SyntheticMind DAX Tile processing for deterministic compute & drift monitoring
                        if sm_tile is not None:
                            arr_tile = np.asarray(window_ids_orig, dtype=np.float32)
                            feat_tile = np.zeros(4096, dtype=np.float32)
                            n_t = min(len(arr_tile), 4096)
                            feat_tile[:n_t] = arr_tile[:n_t] / 100000.0
                            sm_tile.write_sram(feat_tile)
                            delta_input = np.random.randn(4096).astype(np.float32) * 1e-4
                            region_id = f"win_region_{step}_{w_idx % 4}"
                            _tile_metrics = sm_tile.step(region_id, delta_input)
                    variants = [window_ids_orig, window_ids_orig[::-1]]
                    if random.random() < rotation_prob:
                        rot_range = max_seq_len * speed_factor
                        rot = random.randint(0, int(rot_range) - 1) % max_seq_len
                        rot_window = window_ids_orig[rot:] + window_ids_orig[:rot]
                        variants += [rot_window, rot_window[::-1]]
                    for v in variants:
                        batch_input_ids.append(v[:-1])
                        batch_target_ids.append(v[1:])
        except Exception:
            use_indexer = False

        if not use_indexer:
            batch_texts = random.sample(buffer, batch_size)
        # Log token-count savings from hierarchical codebook
        if hasattr(tok, 'codebook') and step % 100 == 0:
            bpe_counts = []
            cb_counts = []
            for t in batch_texts:
                t_clean = clean_text(t)
                t_clean = hscl_compress(t_clean)
                try:
                    # For comparison, measure raw BPE length
                    bpe_len = len(tok.bpe.encode(t_clean))
                except Exception:
                    bpe_len = 0
                cb_len = tok.codebook.token_count(t_clean)
                bpe_counts.append(bpe_len)
                cb_counts.append(cb_len)
            if bpe_counts:
                avg_bpe = sum(bpe_counts)/len(bpe_counts)
                avg_cb = sum(cb_counts)/len(cb_counts)
                saving = (avg_bpe - avg_cb) / max(1, avg_bpe) * 100
                print(f"  [TokenCodebook] BPE avg {avg_bpe:.1f} tokens, CB avg {avg_cb:.1f} tokens, saving {saving:.1f}%")
        for text in batch_texts:
            text = clean_text(text)
            text = hscl_compress(text)
            try:
                ids = tok.encode(text)
            except ValueError:
                continue
            # Clamp token ids to vocab size to prevent out-of-bounds embedding lookup
            ids = [i if i < vocab_size else vocab_size - 1 for i in ids]
            # Chunking: cap token length to avoid memory blow-up
            MAX_TOKENS_PER_TEXT = 8192
            if len(ids) > MAX_TOKENS_PER_TEXT:
                ids = ids[:MAX_TOKENS_PER_TEXT]
            if len(ids) < 2:
                continue
            # Sliding window sampling for layered context with drag consistency
            if len(ids) < max_seq_len:
                window_ids = ids
                # Rubik's cube variants for short texts
                variants_short = [window_ids, window_ids[::-1]]
                if random.random() < rotation_prob:
                    rot_range_s = len(window_ids) * speed_factor
                    rot_s = random.randint(0, int(rot_range_s) - 1) % len(window_ids)
                    rot_window_s = window_ids[rot_s:] + window_ids[:rot_s]
                    variants_short += [rot_window_s, rot_window_s[::-1]]
                # 2D grid rotation if window is perfect square
                sqrt_n_s = int(math.isqrt(len(window_ids)))
                if sqrt_n_s * sqrt_n_s == len(window_ids):
                    try:
                        grid_s = [window_ids[i*sqrt_n_s:(i+1)*sqrt_n_s] for i in range(sqrt_n_s)]
                        cur_grid_s = grid_s
                        for _ in range(3):
                            cur_grid_s = [list(reversed(col)) for col in zip(*cur_grid_s)]
                            rot_flat_s = [tok for row in cur_grid_s for tok in row]
                            variants_short += [rot_flat_s, rot_flat_s[::-1]]
                    except Exception:
                        pass
                for v in variants_short:
                    batch_input_ids.append(v[:-1])
                    batch_target_ids.append(v[1:])
                continue
            n_windows = max(1, (len(ids) - max_seq_len) // stride + 1)
            idx = random.randint(0, n_windows - 1)
            start = idx * stride
            window_ids_orig = ids[start:start + max_seq_len]
            # Store window in memory even on fallback path
            if memory_orch is not None:
                import numpy as np
                arr = np.asarray(window_ids_orig, dtype=np.float32)
                feat = np.zeros(256, dtype=np.float32)
                n = min(len(arr), 256)
                feat[:n] = arr[:n] / 100000.0
                feat[0] = float(len(arr))
                feat[1] = float(arr.mean()) if n>0 else 0.0
                feat[2] = float(arr.std()) if n>1 else 0.0
                vid = f"win_fallback_{step}_{idx}"
                try:
                    memory_orch.store_experience(vid, feat, list(window_ids_orig), context="fallback_window")
                except Exception:
                    pass
            # Pattern Identifier update for fallback path
            if pattern_id is not None:
                pattern_id.update(window_ids_orig, step)
            # SyntheticMind DAX Tile processing for fallback path
            if sm_tile is not None:
                import numpy as np
                arr_tile = np.asarray(window_ids_orig, dtype=np.float32)
                feat_tile = np.zeros(4096, dtype=np.float32)
                n_t = min(len(arr_tile), 4096)
                feat_tile[:n_t] = arr_tile[:n_t] / 100000.0
                sm_tile.write_sram(feat_tile)
                delta_input = np.random.randn(4096).astype(np.float32) * 1e-4
                region_id = f"win_fallback_region_{step}_{idx % 4}"
                _ = sm_tile.step(region_id, delta_input)
            # Rubik's cube context: generate multiple views of the same window
            variants = [window_ids_orig, window_ids_orig[::-1]]
            # rotation augmentation with speed/velocity factor
            if random.random() < rotation_prob:
                # speed_factor increases effective rotation range when loss is volatile
                rot_range = max_seq_len * speed_factor
                rot = random.randint(0, int(rot_range) - 1) % max_seq_len
                rot_window = window_ids_orig[rot:] + window_ids_orig[:rot]
                variants += [rot_window, rot_window[::-1]]
            # 2D grid rotation for perfect square windows - third axis
            sqrt_n = int(math.isqrt(max_seq_len))
            if sqrt_n * sqrt_n == max_seq_len:
                try:
                    grid = [window_ids_orig[i*sqrt_n:(i+1)*sqrt_n] for i in range(sqrt_n)]
                    # generate 90°, 180°, 270° rotations
                    cur_grid = grid
                    for _ in range(3):
                        # rotate 90° clockwise
                        cur_grid = [list(reversed(col)) for col in zip(*cur_grid)]
                        rot_flat = [tok for row in cur_grid for tok in row]
                        variants += [rot_flat, rot_flat[::-1]]
                except Exception:
                    pass
            for v in variants:
                batch_input_ids.append(v[:-1])
                batch_target_ids.append(v[1:])
            # drag/overlap window: second window shifted by stride for consistency
            start2 = start + stride
            if start2 + max_seq_len <= len(ids):
                window2_orig = ids[start2:start2 + max_seq_len]
                # Rubik's cube variants for drag window
                variants2 = [window2_orig, window2_orig[::-1]]
                if random.random() < rotation_prob:
                    rot_range2 = max_seq_len * speed_factor
                    rot2 = random.randint(0, int(rot_range2) - 1) % max_seq_len
                    rot_window2 = window2_orig[rot2:] + window2_orig[:rot2]
                    variants2 += [rot_window2, rot_window2[::-1]]
                # 2D grid rotation for perfect square windows
                sqrt_n = int(math.isqrt(max_seq_len))
                if sqrt_n * sqrt_n == max_seq_len:
                    try:
                        grid2 = [window2_orig[i*sqrt_n:(i+1)*sqrt_n] for i in range(sqrt_n)]
                        cur_grid2 = grid2
                        for _ in range(3):
                            cur_grid2 = [list(reversed(col)) for col in zip(*cur_grid2)]
                            rot_flat2 = [tok for row in cur_grid2 for tok in row]
                            variants2 += [rot_flat2, rot_flat2[::-1]]
                    except Exception:
                        pass
                for v2 in variants2:
                    batch_input_ids.append(v2[:-1])
                    batch_target_ids.append(v2[1:])
        if not skip_batch_gen and not batch_input_ids:
            continue
        # Sanity check for token id bounds
        if not skip_batch_gen:
            flat_ids = [i for seq in batch_input_ids for i in seq]
            if flat_ids:
                max_id = max(flat_ids)
                if max_id >= vocab_size:
                    print(f"[WARN] token id {max_id} >= vocab_size {vocab_size}, clamping batch")
                    # Clamp entire batch to avoid CUDA assert
                    batch_input_ids = [[i if i < vocab_size else vocab_size - 1 for i in seq] for seq in batch_input_ids]
                    batch_target_ids = [[i if i < vocab_size else vocab_size - 1 for i in seq] for seq in batch_target_ids]
            num_variants = len(batch_input_ids)
            max_len = max(len(x) for x in batch_input_ids)
            input_padded = rnn_utils.pad_sequence([torch.tensor(x, dtype=torch.long) for x in batch_input_ids], batch_first=True, padding_value=0)
            target_padded = rnn_utils.pad_sequence([torch.tensor(x, dtype=torch.long) for x in batch_target_ids], batch_first=True, padding_value=0)
            input_padded = input_padded.to(device, non_blocking=True)
            target_padded = target_padded.to(device, non_blocking=True)
        else:
            # DataLoader fast path: ensure tensors are on device
            if input_padded is None or target_padded is None:
                # Fallback to empty batch
                input_padded = torch.empty((0, max_seq_len), dtype=torch.long, device=device)
                target_padded = torch.empty((0, max_seq_len), dtype=torch.long, device=device)
            else:
                input_padded = input_padded.to(device, non_blocking=True)
                target_padded = target_padded.to(device, non_blocking=True)
            num_variants = input_padded.size(0) if input_padded.numel() > 0 else 0
        # Ensure mask is defined for both paths
        mask = target_padded != 0
        # Autocast disabled for stability on Windows
        model_out = model(input_padded)
        if isinstance(model_out, tuple):
            logits, awareness_gates = model_out
        else:
            logits = model_out
        loss = criterion(logits[mask], target_padded[mask])
        # Orchestrator reward signal consumption for reinforcement learning
        try:
            from pathlib import Path
            import json
            reward_log_path = BASE / "runtime" / "logs" / "reward_signal.jsonl"
            if reward_log_path.exists() and step % reward_interval == 0:
                rewards = []
                # Read last reward_window entries
                with open(reward_log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-reward_window:]
                for line in lines:
                    try:
                        entry = json.loads(line)
                        r = float(entry.get('reward', 1.0))
                        rewards.append(r)
                    except Exception:
                        pass
                if rewards:
                    avg_reward = sum(rewards) / len(rewards)
                    # Reward shaping: scale loss down when orchestrator reward is high
                    reward_factor = 0.5 + reward_scale * max(0.0, min(1.0, avg_reward))
                    loss = loss * (1.0 / reward_factor)
        except Exception:
            pass
        # Predictor reward weighting / monitoring
        from infrastructure.true_hybrid.weaveformer import PredictiveLinear
        try:
            from infrastructure.true_hybrid.dual_weight_layers import DualWeightLinear
        except Exception:
            DualWeightLinear = None
        hit_rates = []
        for name, module in model.named_modules():
            if isinstance(module, PredictiveLinear):
                hr = module.hit_rate()
                hit_rates.append(hr)
            elif DualWeightLinear is not None and isinstance(module, DualWeightLinear):
                hr = module.hit_rate()
                hit_rates.append(hr)
        avg_hit = sum(hit_rates)/len(hit_rates) if hit_rates else 0.0
        # Small reward shaping: encourage cache hits by slightly reducing loss when hit rate is high
        loss = loss * (1.0 - reward_lambda * avg_hit)
        # Hybrid CE + MSE loss with awareness gate stability regularization
        if 'awareness_gates' in locals() and awareness_gates is not None:
            mse_loss_fn = nn.MSELoss()
            target_gates = torch.full_like(awareness_gates, 0.5)
            mse_reg = mse_loss_fn(awareness_gates, target_gates)
            loss = loss + mse_weight * mse_reg
        # LQR error correction for loss trajectory
        loss_val_raw = loss.item()
        if ema_loss is None:
            ema_loss = loss_val_raw
            target_loss = loss_val_raw
        else:
            # slowly decreasing target
            target_loss = ema_loss * target_loss_decay
            error = target_loss - loss_val_raw
            d_error = error - prev_error
            prev_error = error
            error_integral += error
            # PID correction on loss
            correction = kp_err * error + ki_err * error_integral - kd_err * d_error
            # apply correction as loss shaping, clamp to avoid explosion
            correction = max(-0.1, min(0.1, correction))
            loss = loss + correction
        # Heatmap entropy regularizer to prevent collapse to single hot spot
        if memory_orch is not None and hasattr(memory_orch, 'heatmap') and memory_orch.heatmap is not None:
            try:
                import numpy as np
                hm = memory_orch.heatmap.maps[0].heat
                # Normalize to probability distribution
                h_sum = hm.sum() + 1e-9
                p = hm / h_sum
                # Entropy = - sum(p log p)
                entropy = -float((p * np.log(p + 1e-12)).sum())
                # Target entropy for 64x64 grid ~ log(4096) ~ 8.3, normalize to ~1
                ent_norm = entropy / np.log(hm.size + 1e-9)
                # Encourage higher entropy -> subtract small term
                loss = loss - entropy_reg_weight * ent_norm
            except Exception:
                pass
        # LQR-style loss-velocity LR controller
        loss_val = loss.item()
        loss_history.append(loss_val)
        if len(loss_history) > auto_tune_window:
            loss_history.pop(0)
        # Auto-tune kp/ki based on recent variance
        if len(loss_history) == auto_tune_window:
            mean_l = sum(loss_history) / len(loss_history)
            var_l = sum((x - mean_l) ** 2 for x in loss_history) / len(loss_history)
            # high variance -> reduce kp
            if var_l > 0.02:
                kp = max(0.05, kp * 0.95)
                ki = max(0.001, ki * 0.95)
                # Increase stability regularization when loss is noisy
                mse_weight = min(0.005, mse_weight * 1.05)
                entropy_reg_weight = min(0.005, entropy_reg_weight * 1.05)
                grad_clip = max(0.5, grad_clip * 0.98)
            elif var_l < 0.005:
                kp = min(0.4, kp * 1.02)
                ki = min(0.02, ki * 1.02)
                # Relax regularization when stable
                mse_weight = max(0.0001, mse_weight * 0.98)
                entropy_reg_weight = max(0.0001, entropy_reg_weight * 0.98)
                grad_clip = min(2.0, grad_clip * 1.02)
            # Auto-tune predictor reward based on hit rate
            if 'avg_hit' in locals():
                if avg_hit < 0.1:
                    reward_lambda = min(0.05, reward_lambda * 1.03)
                elif avg_hit > 0.5:
                    reward_lambda = max(0.005, reward_lambda * 0.97)
            # Auto-tune exploration bias based on loss plateau
            if ema_loss is not None and prev_loss_val is not None:
                loss_delta = abs(ema_loss - prev_loss_val)
                if loss_delta < 0.001:
                    rotation_explore_bias = min(0.2, rotation_explore_bias + 0.001)
                else:
                    rotation_explore_bias = max(-0.1, rotation_explore_bias - 0.0005)
            # Auto-tune noise scale: increase when loss is noisy, decrease when stable
            if var_l > 0.02:
                noise_scale = min(0.05, noise_scale * 1.02)
            elif var_l < 0.005:
                noise_scale = max(0.001, noise_scale * 0.98)
            # Auto-tune entanglement strength: reduce when noisy, increase when stable to exploit braid interactions
            if var_l > 0.02:
                entanglement_strength = max(0.01, entanglement_strength * 0.98)
            elif var_l < 0.005:
                entanglement_strength = min(0.2, entanglement_strength * 1.02)
            # Push noise scale to model
            try:
                model.noise_scale = noise_scale
            except Exception:
                pass
            # Push entanglement strength to model
            try:
                if hasattr(model, 'weave_lattice') and model.weave_lattice is not None:
                    # Update parameter safely without in-place graph mutation
                    with torch.no_grad():
                        es = model.weave_lattice.entanglement_strength
                        if es.data.shape == torch.Size([]):
                            es.copy_(torch.tensor(entanglement_strength, device=es.device))
                        else:
                            es.fill_(entanglement_strength)
            except Exception:
                pass
        if ema_loss is None:
            ema_loss = loss_val
            velocity = 0.0
        else:
            ema_loss = alpha * ema_loss + (1 - alpha) * loss_val
            velocity = loss_val - ema_loss
            integral += velocity
            # Adaptive target loss decay: speed up target when loss is dropping fast, slow down when plateauing
            if abs(velocity) > 0.01:
                target_loss_decay = min(0.9999, max(0.995, target_loss_decay + 0.0001 * (1 if velocity < 0 else -1)))
            else:
                target_loss_decay = min(0.9999, max(0.995, target_loss_decay + 0.00001))
            delta_log_lr = -kp * velocity - ki * integral
            delta_log_lr = max(-0.05, min(0.05, delta_log_lr))
            for pg in optimizer.param_groups:
                new_lr = pg['lr'] * math.exp(delta_log_lr)
                new_lr = min(max(new_lr, 1e-5), 1e-2)
                pg['lr'] = new_lr
        # Gradient accumulation
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()  # retain_graph removed to avoid memory leak
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # Dual-weight delta decay
            if hasattr(model, 'apply_delta_decay'):
                model.apply_delta_decay()
            # EMA update for stable model
            if ema_model is not None:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
        # Update Q-Learning policy with reward
        if use_q_policy and aug_policy is not None and 'action_q' in locals() and action_q is not None:
            aug_policy.update(ema_loss, loss_target, prev_loss_val, action_q, reward_q, ema_loss)
        # Bio-memory consolidation
        if bio_mem is not None and step % 500 == 0:
            bio_mem.heart.consolidate()
        # Dual-weight consolidation
        if hasattr(model, 'consolidate_delta') and step % 5000 == 0:
            try:
                model.consolidate_delta()
                print(f"  [DualWeight] Delta consolidated into master at step {step}")
            except Exception as e:
                print(f"  [DualWeight] Consolidation failed: {e}")
        # Master Orchestrator teaching for runtime wiring
        if master_orch_available and teach_full is not None and step % 500 == 0:
            try:
                # Teach a small sample from the current buffer to keep orchestrator in sync
                sample_texts = random.sample(buffer, min(5, len(buffer)))
                for txt in sample_texts:
                    teach_full(txt)
            except Exception as e:
                # Non-fatal: orchestrator teaching should not break training
                pass
        # Integrated Cognitive Core processing - Structural + MDWE + DAX
        if integrated_core is not None and step % 100 == 0:
            try:
                # Sample a text for integrated processing
                sample_text = random.choice(buffer)
                # Generate dummy activations from model embeddings
                import numpy as np
                # Use token IDs as proxy activations
                try:
                    ids = tok.encode(sample_text[:200])
                    ids = [i if i < vocab_size else vocab_size - 1 for i in ids]
                    act = np.array(ids[:256], dtype=np.float32)
                    if act.size < 256:
                        act = np.pad(act, (0, 256-act.size))
                    else:
                        act = act[:256]
                    # Normalize
                    act = act / (np.linalg.norm(act) + 1e-9)
                    tokens = sample_text.split()[:32]
                    result = integrated_core.process_episode(tokens, act, tags=['training'])
                    if step % 1000 == 0:
                        print(f"  [IntegratedCore] step {step} active={result['mdwe_active']} out_norm={result['mdwe_out_norm']:.3f} structural_v={result['structural_version']}")
                except Exception:
                    pass
            except Exception:
                pass
        # Update speed/velocity tracking for next step
        prev_loss_val = ema_loss
        # Clear predictor caches periodically to bound memory
        if step % 500 == 0:
            for name, module in model.named_modules():
                if hasattr(module, 'cache'):
                    module.cache.clear()
                    module.hit_count = 0
                    module.miss_count = 0
        if step % 100 == 0:
            hit_str = f" hit {avg_hit:.3f}" if 'avg_hit' in locals() else ""
            lr = optimizer.param_groups[0]['lr']
            rot_str = f" rot {rotation_prob:.2f}"
            var_str = f" vars {num_variants}"
            vel_str = f" vel {velocity:.4f}" if 'velocity' in locals() else ""
            print(f"Step {step} loss {loss.item():.4f}{hit_str} lr {lr:.6f} seq {max_seq_len} bs {batch_size}{rot_str}{var_str}{vel_str}")
            if memory_orch is not None and step % 1000 == 0:
                hc = memory_orch.health_check()
                print(f"  [Health] vectors={hc['vectors']} index={hc['index_entries']} graph_edges={hc['graph_edges']} family_nodes={hc.get('family_nodes',0)}")
                # Periodic persistence to disk to avoid loss on crash
                try:
                    if step % 2000 == 0:
                        memory_orch.save()
                        print(f"  [Memory] Periodic save at step {step}")
                except Exception as e:
                    print(f"  [Memory] Periodic save failed: {e}")
                # Auto-tune diagnostics
                # Lattice strength for Weave Lattice bridges
                lattice_strength = 0.1
                entanglement_strength = 0.05
                try:
                    if hasattr(model, 'weave_lattice') and model.weave_lattice is not None:
                        ls = model.weave_lattice.lattice_strength
                        lattice_strength = float(ls.item() if isinstance(ls, torch.nn.Parameter) else ls)
                        es = model.weave_lattice.entanglement_strength
                        entanglement_strength = float(es.item() if isinstance(es, torch.nn.Parameter) else es)
                except Exception:
                    pass
                print(f"  [AutoTune] lr_cap=0.01 mse_w={mse_weight:.4f} reward_l={reward_lambda:.4f} ent_w={entropy_reg_weight:.4f} grad_clip={grad_clip:.2f} rot_bias={rotation_explore_bias:.3f} noise={noise_scale:.4f} kp={kp:.3f} ki={ki:.4f} tgt_decay={target_loss_decay:.4f} lattice_str={lattice_strength:.4f} ent_str={entanglement_strength:.4f}")
                if pattern_id is not None:
                    pstats = pattern_id.stats()
                    top = pattern_id.top_patterns(3)
                    print(f"  [Pattern] patterns={pstats['patterns']} recent={pstats['recent_buffer']} top={top}")
            if sm_tile is not None and step % 100 == 0:
                drift = sm_tile.drift.drift_score()
                hot = sm_tile.hot_overlays(3)
                stats = sm_tile.scheduler.stats
                print(f"  [SMTile] drift={drift:.4f} drifting={sm_tile.drift.is_drifting()} hot_overlays={len(hot)} fibers_executed={stats['executed']} warps={stats['warps']}")
            # Live heatmap visualisation dump every 10k steps
            if memory_orch is not None and step % 10000 == 0 and step > 0:
                try:
                    import numpy as np
                    from PIL import Image
                    hm = memory_orch.heatmap
                    if hm is not None:
                        heat = hm.maps[0].heat
                        # Bell-curve / Gaussian normalization for better contrast
                        mean = float(heat.mean())
                        std = float(heat.std()) + 1e-9
                        z = (heat - mean) / std
                        # Sigmoid bell curve centered at mean
                        norm = 1.0 / (1.0 + np.exp(-z / 0.5))
                        norm = norm.astype(np.float32)
                        # Hot-cold RGB overlay: red = hot, blue = cold
                        r = norm.clip(0,1)
                        b = 1.0 - norm
                        g = np.zeros_like(norm)
                        rgb = np.stack([r, g, b], axis=-1)
                        rgb_uint8 = (rgb * 255).astype(np.uint8)
                        img = Image.fromarray(rgb_uint8, mode='RGB')
                        out_dir = BASE / "runtime" / "memory" / "heatmaps"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"heatmap_step_{step}.png"
                        img.save(out_path)
                        print(f"  [Heatmap] saved {out_path}")
                except Exception as e:
                    print(f"  [Heatmap] dump failed: {e}")
            # Validation eval every 100 steps
            if step % 100 == 0 and step > 0:
                # Disable predictor during validation to avoid cache collisions
                for name, module in model.named_modules():
                    if hasattr(module, 'use_predictor'):
                        module.use_predictor = False
                model.eval()
                val_losses = []
                eval_model = ema_model if ema_model is not None else model
                with torch.no_grad():
                    for vtxt in val_texts[:50]:
                        vtxt = clean_text(vtxt)
                        vtxt = hscl_compress(vtxt)
                        try:
                            ids = tok.encode(vtxt)
                        except:
                            continue
                        ids = [i if i < vocab_size else vocab_size - 1 for i in ids]
                        if len(ids) < 2:
                            continue
                        if len(ids) > max_seq_len:
                            ids = ids[:max_seq_len]
                        input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                        target_ids = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
                        model_out = eval_model(input_ids)
                        if isinstance(model_out, tuple):
                            logits = model_out[0]
                        else:
                            logits = model_out
                        # Flatten for CrossEntropy to avoid shape issues
                        loss_v = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                        val_losses.append(loss_v.item())
                if val_losses:
                    val_loss = sum(val_losses)/len(val_losses)
                    print(f"  Val loss {val_loss:.4f} on {len(val_losses)} samples")
                    # Log train loss in eval mode on same distribution to quantify gap
                    train_eval_losses = []
                    with torch.no_grad():
                        for ttxt in train_texts[:50]:
                            ttxt = clean_text(ttxt)
                            ttxt = hscl_compress(ttxt)
                            try:
                                ids = tok.encode(ttxt)
                            except:
                                continue
                            ids = [i if i < vocab_size else vocab_size - 1 for i in ids]
                            if len(ids) < 2:
                                continue
                            if len(ids) > max_seq_len:
                                ids = ids[:max_seq_len]
                            input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                            target_ids = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
                            model_out = eval_model(input_ids)
                            logits = model_out[0] if isinstance(model_out, tuple) else model_out
                            loss_t = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                            train_eval_losses.append(loss_t.item())
                    if train_eval_losses:
                        train_eval_loss = sum(train_eval_losses)/len(train_eval_losses)
                        print(f"  Train eval loss {train_eval_loss:.4f} on {len(train_eval_losses)} samples (eval mode)")
                    # Foldable Map logging for predictor promotion
                    if foldable_logging_available and foldable_log is not None and step % 500 == 0:
                        try:
                            # Log a few validation samples to populate predictor promotion ledger
                            sample_count = min(3, len(val_texts))
                            for i in range(sample_count):
                                txt = val_texts[i][:200]
                                # Simple source tag for predictor promotion tracking
                                source_tag = f"predictor_promoted:weaveformer_conf_{max(0.5, 1.0 - val_loss/10.0):.2f}"
                                foldable_log(txt, txt[:50], True, source=source_tag)
                        except Exception:
                            pass
                    # MBFN explainable reasoning demo
                    if mbfn_bridge is not None:
                        demo_text = val_texts[0][:200]
                        demo_text = hscl_compress(demo_text)
                        candidates = ["Paris", "London", "Berlin"]
                        weave_scores = [0.8, 0.5, 0.3]
                        ranked = mbfn_bridge.rank_candidates(candidates, weave_scores)
                        top = ranked[0][0]
                        expl = mbfn_bridge.explain_choice(top)
                        print(f"  MBFN top candidate {top} with explanation: {expl[:3]}")
                    # Reasoning memory demo
                    if reasoning_mem is not None and vector_mem is not None:
                        # Build a simple query vector from first val sample
                        try:
                            import numpy as np
                            demo_ids = tok.encode(hscl_compress(clean_text(val_texts[0][:200])))
                            demo_ids = [i if i < vocab_size else vocab_size - 1 for i in demo_ids]
                            query_vec = np.zeros(256, dtype=np.float32)
                            n = min(len(demo_ids), 256)
                            query_vec[:n] = np.array(demo_ids[:n], dtype=np.float32) / 100000.0
                            hits = reasoning_mem.recall(query_vec, k=3)
                            print("  ReasoningMemory top hits:")
                            for h in hits:
                                print(f"    {h.vid} score={h.score:.3f} heart={h.heart_score:.3f} muscle={h.muscle_score:.3f} graph={h.graph_score:.3f}")
                        except Exception as e:
                            print(f"  ReasoningMemory demo skipped: {e}")
                model.train()
                # Re-enable predictor for training
                for name, module in model.named_modules():
                    if hasattr(module, 'use_predictor'):
                        module.use_predictor = True
                # Periodic memory maintenance with PID+LQR control
                if step % 500 == 0 and step > 0:
                    if memory_orch is not None:
                        try:
                            metrics = memory_orch.maintenance_step()
                            print(f"  [Memory] PID+LQR maintenance at step {step} | avg_chain={metrics['avg_chain_len']:.2f} recall={metrics['recall_score']:.3f} compress={metrics['compress_aggressiveness']:.3f} decay={metrics['heart_decay']:.4f}")
                        except Exception as e:
                            print(f"  [Memory] Maintenance failed: {e}")
                    elif vector_mem is not None:
                        try:
                            vector_mem.compress()
                            if bio_mem is not None:
                                bio_mem.heart.consolidate()
                            print(f"  [Memory] Compressed vector memory and consolidated bio-memory at step {step}")
                        except Exception as e:
                            print(f"  [Memory] Compression/consolidation failed: {e}")
            # checkpoint - model only, save every 1000 steps to avoid I/O overload
            if step % 1000 == 0:
                save_compressed(ckpt_path, model.state_dict())
                if ema_model is not None:
                    ema_ckpt_path = BASE / "runtime" / "models" / "weave_decoder_ema.pt"
                    save_compressed(ema_ckpt_path, ema_model.state_dict())
                if CODEBOOK_AVAILABLE and save_model_codebook is not None:
                    try:
                        cb_dir = BASE / "runtime" / "models" / "codebook_checkpoints"
                        extra_state = {
                            'optimizer': optimizer.state_dict(),
                            'step': step,
                            'ema_loss': ema_loss,
                            'integral': integral,
                            'error_integral': error_integral,
                            'prev_error': prev_error,
                            'max_seq_len': max_seq_len,
                            'batch_size': batch_size,
                            'loss_target': loss_target,
                        }
                        save_model_codebook(model, cb_dir, base_name=f"weave_decoder_cb_step{step}", extra_state=extra_state)
                        print(f"Codebook checkpoint saved at step {step}")
                    except Exception as e:
                        print(f"Codebook save failed: {e}")
                save_compressed(full_ckpt_path, {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'ema_loss': ema_loss,
                    'integral': integral,
                    'error_integral': error_integral,
                    'prev_error': prev_error,
                    'max_seq_len': max_seq_len,
                    'batch_size': batch_size,
                    'loss_target': loss_target,
                })
    # final saves
    save_compressed(ckpt_path, model.state_dict())
    if ema_model is not None:
        ema_ckpt_path = BASE / "runtime" / "models" / "weave_decoder_ema.pt"
        save_compressed(ema_ckpt_path, ema_model.state_dict())
    if CODEBOOK_AVAILABLE and save_model_codebook is not None:
        try:
            cb_dir = BASE / "runtime" / "models" / "codebook_checkpoints"
            extra_state = {
                'optimizer': optimizer.state_dict(),
                'step': steps,
                'ema_loss': ema_loss,
                'integral': integral,
                'error_integral': error_integral,
                'prev_error': prev_error,
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
                'loss_target': loss_target,
            }
            save_model_codebook(model, cb_dir, base_name="weave_decoder_final_cb", extra_state=extra_state)
            print("Final codebook checkpoint saved")
        except Exception as e:
            print(f"Final codebook save failed: {e}")
    save_compressed(full_ckpt_path, {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': steps,
        'ema_loss': ema_loss,
        'integral': integral,
        'error_integral': error_integral,
        'prev_error': prev_error,
        'max_seq_len': max_seq_len,
        'batch_size': batch_size,
        'loss_target': loss_target,
    }, full_ckpt_path)
    # Persist vector memory
    if memory_orch is not None:
        try:
            memory_orch.save()
            hc = memory_orch.health_check()
            print(f"AIMemory saved: {hc['vectors']} vectors, {hc['graph_edges']} edges")
        except Exception as e:
            print(f"AIMemory save failed: {e}")
    elif vector_mem is not None:
        try:
            mem_path = BASE / "runtime" / "memory" / "vecmem"
            mem_path.mkdir(parents=True, exist_ok=True)
            vector_mem.save(mem_path)
            print(f"Vector memory saved with {len(vector_mem.store.vectors)} vectors")
        except Exception as e:
            print(f"Vector memory save failed: {e}")
    # Save integrated cognitive core checkpoint
    if integrated_core is not None:
        try:
            ckpt_path_core = BASE / "runtime" / "models" / "integrated_core"
            integrated_core.save_checkpoint(ckpt_path_core)
            print(f"Integrated Cognitive Core saved: {integrated_core.health_check()}")
        except Exception as e:
            print(f"Integrated Core save failed: {e}")
    print("Saved model")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train WeaveFormer decoder")
    parser.add_argument('--reward-window', type=int, default=100, help='Reward signal window size')
    parser.add_argument('--reward-interval', type=int, default=100, help='Reward signal read interval steps')
    parser.add_argument('--reward-scale', type=float, default=1.0, help='Reward signal scale')
    parser.add_argument('--force-codebook-resume', action='store_true', help='Force resume from layered codebook checkpoint only')
    args = parser.parse_args()
    train(reward_window=args.reward_window, reward_interval=args.reward_interval, reward_scale=args.reward_scale, force_codebook_resume=args.force_codebook_resume)
