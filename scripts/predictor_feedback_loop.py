"""
Predictor feedback loop: uses foldable map promotions/reworks to adjust predictor thresholds and Third Brain weights
"""
import json
from pathlib import Path
import sys

BASE = Path('C:/Users/thoma/.lmstudio/apps/bionic/projects/d49037d8-47f4-5808-9028-c707de117f8f/workspace/TRANSLATION_OS_V1')
LOG_DIR = BASE / 'runtime' / 'logs'
LEDGER = LOG_DIR / 'foldable_answer_ledger.jsonl'
SUMMARY = LOG_DIR / 'predictor_promotion_summary.json'

def load_ledger():
    entries = []
    try:
        # Try BitDrop compressed reader
        sys.path.insert(0, str(BASE))
        from infrastructure.logging.bitdrop_log_writer import get_writer
        writer = get_writer()
        for line in writer.read_lines(LEDGER):
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
        if entries:
            return entries
    except Exception:
        pass
    # Fallback to plain
    if LEDGER.exists():
        with open(LEDGER, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries

def compute_stats(entries):
    stats = {}
    for e in entries:
        src = e.get('source','')
        # Parse source tags
        # predictor_promoted:xxx or meta_evaluator_needs_rework:xxx
        if 'predictor_promoted' in src:
            # extract predictor name
            parts = src.split(':')
            if len(parts) >= 2:
                pred = parts[1].split('_conf')[0]
                key = f"promoted:{pred}"
                stats.setdefault(key, {'promotions':0, 'reworks':0, 'conf_sum':0.0, 'cnt':0})
                stats[key]['promotions'] += 1
                # extract conf if present
                if '_conf' in src:
                    try:
                        conf = float(src.split('_conf')[1].split('_')[0])
                        stats[key]['conf_sum'] += conf
                        stats[key]['cnt'] += 1
                    except Exception:
                        pass
        elif 'meta_evaluator_needs_rework' in src:
            parts = src.split(':')
            if len(parts) >= 2:
                pred = parts[1].split('_conf')[0]
                key = f"rework:{pred}"
                stats.setdefault(key, {'promotions':0, 'reworks':0})
                stats[key]['reworks'] += 1
    # compute reliability
    summary = {}
    for k,v in stats.items():
        if k.startswith('promoted:'):
            pred = k.split(':',1)[1]
            prom = v.get('promotions',0)
            rework_key = f"rework:{pred}"
            reworks = stats.get(rework_key, {}).get('reworks',0)
            cnt = v.get('cnt',1)
            avg_conf = v.get('conf_sum',0)/cnt if cnt else 0
            reliability = prom / (prom + reworks + 1e-9)
            summary[pred] = {
                'promotions': prom,
                'reworks': reworks,
                'avg_conf': avg_conf,
                'reliability': reliability
            }
    return summary

def load_bias_variance_reward():
    reward_log = LOG_DIR / 'bias_variance_reward.jsonl'
    rewards = []
    if reward_log.exists():
        try:
            with open(reward_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rewards.append(json.loads(line).get('reward', 0.0))
                    except Exception:
                        pass
        except Exception:
            pass
    if rewards:
        avg_reward = sum(rewards) / len(rewards)
    else:
        avg_reward = 0.0
    return avg_reward

def load_avg_weave_weight():
    diag_log = LOG_DIR / 'orchestrator_diagnostics.jsonl'
    weights = []
    if diag_log.exists():
        try:
            with open(diag_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        w = entry.get('bias_variance_weave_weight')
                        if isinstance(w, (int, float)):
                            weights.append(w)
                    except Exception:
                        pass
        except Exception:
            pass
    if weights:
        return sum(weights) / len(weights)
    return 1.0

def update_predictor_tuner(summary):
    # Incorporate Bias-Variance reward and weave lattice weight into tuning
    avg_bv_reward = load_bias_variance_reward()
    avg_weave_weight = load_avg_weave_weight()
    # Mini-batch gradient descent adjustment
    try:
        sys.path.insert(0, str(BASE / 'infrastructure' / 'learning'))
        from mini_batch_gd import update_weights
        update_weights(learning_rate=0.01)
    except Exception:
        pass
    # Linear regression calibration with normal equation
    try:
        sys.path.insert(0, str(BASE / 'infrastructure' / 'learning'))
        from linear_regression import calibrate_predictors
        calibrate_predictors()
    except Exception:
        pass
    # Least squares routine
    try:
        sys.path.insert(0, str(BASE / 'infrastructure' / 'learning'))
        from least_squares import apply_least_squares
        apply_least_squares()
    except Exception:
        pass
    # Batch + Stochastic Gradient Descent combined
    try:
        sys.path.insert(0, str(BASE / 'infrastructure' / 'learning'))
        from batch_stochastic_gd import combined_update
        combined_update()
    except Exception:
        pass
    # Simple heuristic: lower threshold for reliable predictors, raise for unreliable
    try:
        sys.path.insert(0, str(BASE / 'infrastructure' / 'predictor_tuning'))
        from predictor_tuner import load_thresholds, save_thresholds
        thresholds = load_thresholds()
        for pred, data in summary.items():
            rel = data.get('reliability',0.5)
            # map reliability 0-1 to threshold 0.85-0.95
            new_thresh = 0.85 + (1-rel)*0.1
            # Incorporate Bias-Variance reward: higher reward -> lower threshold for more aggressive promotion
            reward_factor = max(0.0, min(1.0, avg_bv_reward))  # normalize
            # Reward pushes threshold down by up to 0.05
            new_thresh = new_thresh - reward_factor * 0.05
            # Incorporate weave lattice weight: higher weave weight -> lower threshold
            weave_factor = max(0.0, min(1.0, (avg_weave_weight - 1.0) / 2.0))
            new_thresh = new_thresh - weave_factor * 0.03
            # clamp
            new_thresh = max(0.80, min(0.98, new_thresh))
            # Find matching key
            key = None
            for k in thresholds:
                if pred.lower() in k.lower():
                    key = k
                    break
            if not key:
                # generic mapping
                key = pred
            thresholds[key] = round(new_thresh,3)
        save_thresholds(thresholds)
    except Exception as e:
        print('predictor tuner update failed', e)

def main():
    entries = load_ledger()
    summary = compute_stats(entries)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    # Boruta predictor selection for noise reduction
    try:
        sys.path.insert(0, str(BASE / 'infrastructure' / 'predictor_selection'))
        from boruta_selector import run_boruta
        boruta_res = run_boruta()
        print('Boruta selected predictors:', boruta_res.get('selected_predictors'))
    except Exception as e:
        print('Boruta selection failed:', e)
    # Lottery Ticket pruning for sparse sub-network
    lt_res = None
    try:
        sys.path.insert(0, str(BASE / 'infrastructure' / 'predictor_selection'))
        from lottery_ticket_pruner import run_lottery_ticket
        lt_res = run_lottery_ticket()
        print('Lottery Ticket winners:', lt_res.get('winners'))
    except Exception as e:
        print('Lottery Ticket pruning failed:', e)
    # Auto-apply Lottery Ticket mask to predictor thresholds
    try:
        sys.path.insert(0, str(BASE / 'infrastructure' / 'predictor_tuning'))
        from predictor_tuner import load_thresholds, save_thresholds
        thresholds = load_thresholds()
        if lt_res and lt_res.get('mask'):
            mask = lt_res['mask']
            # Keep only winning tickets, disable others by raising threshold to max
            for k in list(thresholds.keys()):
                # match predictor name case-insensitive
                pred_name = k.lower()
                keep = any(pred_name in w.lower() for w in lt_res.get('winners', []))
                if not keep:
                    thresholds[k] = 0.99  # effectively inactive
        save_thresholds(thresholds)
        print('Lottery Ticket mask applied to thresholds')
    except Exception as e:
        print('Failed to apply Lottery Ticket mask:', e)
    update_predictor_tuner(summary)
    print('Predictor feedback loop complete, summary written to', SUMMARY)

if __name__ == '__main__':
    main()
