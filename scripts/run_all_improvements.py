"""
Run all improvements together for 2h/day training loop
Starts Day Dream, Dream State, Training Governor and Evaluation Harness
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import time
from pathlib import Path
BASE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(BASE))
MAX_WORKERS = os.cpu_count() or 1

def log_stage(name, start):
    import datetime
    elapsed = (datetime.datetime.utcnow() - start).total_seconds()
    print(f"[DEBUG] Stage '{name}' completed in {elapsed:.1f}s", flush=True)

def run_cycle():
    import datetime
    cycle_start = datetime.datetime.utcnow()
    print(f"[Cycle] Starting improvements cycle at {cycle_start.isoformat()}", flush=True)
    # 0. Background Wikidata ingest
    print("→ Wikidata ingest", flush=True)
    import datetime, traceback
    wi_start = datetime.datetime.utcnow()
    try:
        import concurrent.futures
        from learn_wikidata_loop_byteoffset import load_state, process_batch, save_state
        st = load_state()
        off = st.get('byte_offset',0)
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut = ex.submit(process_batch, off, 100000)
            n = fut.result(timeout=60)
        print(f"← Wikidata ingest done: {n} entities", flush=True)
        # Auto-generate Q/A from Wikidata for model-like training
        try:
            sys.path.insert(0, str(BASE.parent / "infrastructure" / "wikidata_qa"))
            from wikidata_qa_generator import generate_qa
            from orchestrator.master_orchestrator_v0 import teach_full
            # Demo items - in real use replace with actual Wikidata items from ingest
            demo_items = [
                ("Paris", "P1376", "France"),
                ("Berlin", "P1376", "Germany"),
                ("Rome", "P1376", "Italy"),
            ]
            qa_count = 0
            for label, prop, val in demo_items:
                q,a = generate_qa(label, prop, val)
                teach_full(q, a, reward=1.0)
                qa_count += 1
            print(f"  Wikidata QA auto-teach: {qa_count} pairs", flush=True)
        except Exception as e:
            print(f"  Wikidata QA auto-teach skipped: {e}", flush=True)
        log_stage("Wikidata ingest", wi_start)
    except Exception as e:
        print(f"← Wikidata ingest skipped/timeout: {e}", flush=True)
        traceback.print_exc()
    # 1. Training governor auto-tune with policy override
    print("→ Governor", flush=True)
    import datetime, traceback
    gov_start = datetime.datetime.utcnow()
    try:
        from training_governor import adjust_params, load_recent_scores, compute_trend
        sys.path.insert(0, str(BASE.parent / "infrastructure" / "policy"))
        from policy_engine import get_governor_overrides, select_mode
        # Load recent policy mode from ledger
        policy_mode = "exploitation"
        try:
            from bitdrop_persistence import load_lines_bitdrop
            from pathlib import Path
            ledger_path = BASE.parent / "runtime" / "logs" / "master_ledger.jsonl"
            lines = load_lines_bitdrop(ledger_path)
            for e in reversed(lines[-50:]):
                if e.get("type") == "policy_decision":
                    policy_mode = e.get("payload", {}).get("mode", "exploitation")
                    break
        except Exception:
            pass
        scores = load_recent_scores()
        trend = compute_trend(scores)
        cfg = adjust_params(trend)
        # Apply policy overrides
        overrides = get_governor_overrides(policy_mode)
        cfg.update(overrides)
        print(f"  Governor trend={trend:.2f} mode={policy_mode} config={cfg}", flush=True)
        log_stage("Governor", gov_start)
        print("← Governor", flush=True)
    except Exception as e:
        print(f"  Governor skipped: {e}", flush=True)
        traceback.print_exc()
        print("← Governor", flush=True)

    # 1.5 Checklist self-audit
    print("→ Checklist", flush=True)
    try:
        sys.path.insert(0, str(BASE.parent / "infrastructure" / "checklist"))
        from checklist_engine import run_checklist
        checklist = run_checklist()
        overall = checklist.get("overall_pass", False)
        print(f"  Checklist overall: {'PASS' if overall else 'FAIL'}")
        for k,v in checklist.get("checks", {}).items():
            print(f"    {k}: {'PASS' if v.get('pass') else 'FAIL'} {v}")
        print("← Checklist", flush=True)
    except Exception as e:
        print(f"  Checklist skipped: {e}", flush=True)
        print("← Checklist", flush=True)

    # 2. Evaluation harness
    print("→ Evaluation", flush=True)
    import datetime, traceback
    eval_start = datetime.datetime.utcnow()
    try:
        from evaluation_harness import run_eval
        eval_res = run_eval()
        print(f"  Evaluation run at {eval_res['timestamp']}, {len(eval_res['results'])} tests", flush=True)
        log_stage("Evaluation", eval_start)
        print("← Evaluation", flush=True)
    except Exception as e:
        print(f"  Evaluation skipped: {e}", flush=True)
        traceback.print_exc()
        print("← Evaluation", flush=True)

    # 3. Planner update
    print("→ Planner", flush=True)
    import datetime, traceback
    planner_start = datetime.datetime.utcnow()
    try:
        from planner import update_progress
        roadmap, prog = update_progress()
        print(f"  Planner updated, next action: {roadmap['next_actions'][0] if roadmap['next_actions'] else 'None'}", flush=True)
        log_stage("Planner", planner_start)
        print("← Planner", flush=True)
    except Exception as e:
        print(f"  Planner skipped: {e}", flush=True)
        traceback.print_exc()
        print("← Planner", flush=True)

    # 4. Nightly consolidation light pass
    print("→ Consolidation", flush=True)
    import datetime, traceback
    cons_start = datetime.datetime.utcnow()
    try:
        from nightly_consolidation import run as consolidate
        cons = consolidate()
        print(f"  Consolidation layers={cons['layers']}", flush=True)
        log_stage("Consolidation", cons_start)
        print("← Consolidation", flush=True)
    except Exception as e:
        print(f"  Consolidation skipped: {e}", flush=True)
        traceback.print_exc()
        print("← Consolidation", flush=True)

    # 5. Day Dream cycle
    print("→ Day Dream", flush=True)
    from day_dream import day_dream_cycle
    import datetime, traceback
    dd_start = datetime.datetime.utcnow()
    try:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut = ex.submit(day_dream_cycle)
            fut.result(timeout=45)
        print("  Day Dream cycle done", flush=True)
        log_stage("Day Dream", dd_start)
        print("← Day Dream", flush=True)
    except Exception as e:
        print(f"  Day Dream cycle skipped/timeout: {e}", flush=True)
        traceback.print_exc()
        print("← Day Dream", flush=True)

    # 5.5 Teacher Sampler - ask model for explanations on low confidence queries
    print("→ Teacher Sampler", flush=True)
    try:
        from uncertainty_sampler import sample_uncertain
        from orchestrator.master_orchestrator_v0 import run_orchestrator
        sys.path.insert(0, str(BASE.parent / "runtime"))
        from model_adapter import model_adapter
        queries = sample_uncertain(k=3)
        taught = 0
        for q in queries:
            # Run orchestrator to get current answer and confidence
            res = run_orchestrator(q)
            conf = res.get("diagnostics", {}).get("confidence_prior", 0.5)
            if conf >= 0.7:
                continue
            # Ask model for explanation
            try:
                explain_prompt = f"User question: {q}\nExplain step-by-step how you would answer this, including reasoning and concepts."
                teach_res = model_adapter.generate(explain_prompt)
                explanation = teach_res.get("answer","")
                # Store explanation for learning
                from scripts.reasoning_trace_store import store_trace
                store_trace(q, res.get("response"), explanation, source="teacher_sampler", confidence=conf)
                taught += 1
            except Exception:
                pass
        print(f"  Teacher Sampler taught {taught} explanations", flush=True)
        print("← Teacher Sampler", flush=True)
    except Exception as e:
        print(f"  Teacher Sampler skipped: {e}", flush=True)
        print("← Teacher Sampler", flush=True)

    # 6. Dream State cycle
    print("→ Dream State", flush=True)
    from dream_state import dream_cycle
    import datetime, traceback
    ds_start = datetime.datetime.utcnow()
    try:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut = ex.submit(dream_cycle, cycles=3)
            fut.result(timeout=60)
        print("  Dream State cycle done", flush=True)
        log_stage("Dream State", ds_start)
        print("← Dream State", flush=True)
    except Exception as e:
        print(f"  Dream State cycle skipped/timeout: {e}", flush=True)
        traceback.print_exc()
        print("← Dream State", flush=True)

    # 7. Micro-cluster dashboard summary
    try:
        from micro_cluster_dashboard import top_micro_clusters, emergence_trend
        print("  Micro-cluster dashboard:")
        top_clusters = []
        for i, size, rep in top_micro_clusters():
            print(f"    #{i+1} size={size} rep={rep}")
            top_clusters.append((size, rep))
        print(f"    Trend: {emergence_trend()}")
        # Synthetic QA variation for top micro-clusters
        print("  Synthetic QA variation for top clusters", flush=True)
        from orchestrator.master_orchestrator_v0 import run_orchestrator, teach_full
        import random, concurrent.futures
        variations_made = 0
        def _process_variation(rep):
            nonlocal variations_made
            templates = [
                f"Explain {rep}",
                f"What is {rep}",
                f"Tell me about {rep}",
                f"Summarize {rep}"
            ]
            local_count = 0
            for t in random.sample(templates, min(2, len(templates))):
                q = t
                try:
                    res = run_orchestrator(q)
                    ans = res.get('response') or ''
                    if ans:
                        teach_full(q, ans)
                        local_count += 1
                except Exception:
                    pass
            return local_count
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(_process_variation, rep) for _, rep in top_clusters[:3]]
            for f in concurrent.futures.as_completed(futures):
                variations_made += f.result()
        print(f"  Synthetic variations taught: {variations_made}", flush=True)
    except Exception as e:
        print(f"  Dashboard skipped: {e}")

    # 8. Foldable multilayer dashboard snapshot
    try:
        from foldable_dashboard import print_dashboard
        print("  Foldable multilayer dashboard snapshot:")
        # Capture output without flooding
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_dashboard()
        out = buf.getvalue().strip().splitlines()
        for line in out[:12]:
            print(f"    {line}")
    except Exception as e:
        print(f"  Foldable dashboard skipped: {e}")

    # 8b. Convergence dashboard
    try:
        sys.path.insert(0, str(BASE.parent / "infrastructure" / "convergence_map"))
        from convergence_map import get_top_convergences, get_hot_terms, get_cluster_summary
        print("  Convergence dashboard:")
        top = get_top_convergences(5)
        for i, entry in enumerate(top):
            q = entry.get("query","")[:60]
            score = entry.get("weighted_score",0)
            print(f"    Top {i+1}: score={score:.2f} query={q}")
        hot = get_hot_terms(5)
        print(f"    Hot terms: {', '.join([t for t,_ in hot])}")
        clusters = get_cluster_summary()
        if clusters:
            print(f"    Top cluster: {clusters[0]['representative']} avg_score={clusters[0]['avg_score']:.2f}")
    except Exception as e:
        print(f"  Convergence dashboard skipped: {e}")

    # 8c. GPS + North Star dashboard
    try:
        sys.path.insert(0, str(BASE.parent / "infrastructure" / "north_star"))
        from north_star import load_state
        ns_state = load_state()
        alignment_hist = ns_state.get("alignment_history", [])
        avg_align = sum(h.get("alignment",0) for h in alignment_hist[-20:]) / max(1, len(alignment_hist[-20:]))
        print(f"  GPS/North Star dashboard:")
        print(f"    Anchor vector: {ns_state.get('anchor_vector')}")
        print(f"    Avg alignment last 20: {avg_align:.2f}")
        print(f"    Core goals: {', '.join(ns_state.get('core_goals',[]))}")
    except Exception as e:
        print(f"  GPS/North Star dashboard skipped: {e}")

    # 8c2. Answer Maps overlay convergence dashboard
    try:
        sys.path.insert(0, str(BASE.parent / "infrastructure" / "answer_maps"))
        from answer_maps import overlay_convergence
        conv = overlay_convergence(top_k=5)
        print(f"  Answer Maps overlay convergence:")
        if conv:
            for i, item in enumerate(conv):
                qc = item.get("query_correct","")[:60]
                qw = item.get("query_wrong","")[:60]
                score = item.get("convergence_score")
                print(f"    #{i+1} score={score} correct={qc} | wrong={qw}")
        else:
            print(f"    No overlap yet")
    except Exception as e:
        print(f"  Answer Maps overlay skipped: {e}")

    # 8d. Synthesizer candidate ranking dashboard
    try:
        from orchestrator.master_orchestrator_v0 import run_orchestrator
        test_q = "What is the capital of France?"
        res = run_orchestrator(test_q)
        diag = res.get("diagnostics", {})
        cands = diag.get("synthesizer_candidates", [])
        print(f"  Synthesizer candidate ranking:")
        for i,c in enumerate(sorted(cands, key=lambda x: x.get("synthesizer_score",0), reverse=True)[:3]):
            src = c.get("source","?")
            score = c.get("synthesizer_score",0)
            resp = (c.get("response") or "")[:80]
            print(f"    #{i+1} score={score:.3f} source={src} resp={resp}...")
    except Exception as e:
        print(f"  Synthesizer dashboard skipped: {e}")

    # 8e. Training progress snapshot
    try:
        from bitdrop_persistence import load_json_bitdrop
        plan_path = BASE.parent / "runtime" / "logs" / "progress_tracker.json"
        prog = load_json_bitdrop(plan_path)
        milestones = prog.get("roadmap_summary", []) if prog else []
        total = sum(m.get("progress",0) for m in milestones)
        avg = total / len(milestones) if milestones else 0
        print(f"  Training Progress Snapshot:")
        for m in milestones:
            print(f"    {m.get('name')}: {int(m.get('progress',0)*100)}% [{m.get('status')}]" )
        print(f"    Overall: {avg*100:.1f}%")
        # Emergence %
        from pathlib import Path
        import json
        hist_path = BASE.parent / "runtime" / "logs" / "emergence_score_history.jsonl"
        if hist_path.exists():
            last_line = hist_path.read_text().strip().splitlines()[-1]
            try:
                last = json.loads(last_line)
                perc = last.get("percent") or last.get("score")
                if perc is not None:
                    print(f"    Emergence tests: {perc}%")
            except:
                pass
    except Exception as e:
        print(f"  Progress snapshot skipped: {e}")

    # 8f. Learning Modalities usage
    try:
        import json
        mod_path = BASE.parent / "runtime" / "logs" / "learning_modalities.jsonl"
        counts = {}
        if mod_path.exists():
            for line in mod_path.read_text().splitlines()[-200:]:
                try:
                    e = json.loads(line)
                    m = e.get("modality")
                    counts[m] = counts.get(m,0)+1
                except:
                    pass
        print(f"  Learning Modalities usage last 200:")
        for k,v in sorted(counts.items(), key=lambda x: -x[1])[:8]:
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"  Modalities dashboard skipped: {e}")

    # 8g. Learning integrity dashboard
    try:
        import json
        int_path = BASE.parent / "runtime" / "logs" / "learning_integrity.jsonl"
        total = 0
        passes = 0
        if int_path.exists():
            lines = int_path.read_text().splitlines()
            total = len(lines)
            for line in lines[-200:]:
                try:
                    e = json.loads(line)
                    if e.get("full_design_pass"):
                        passes += 1
                except:
                    pass
        print(f"  Learning Integrity last 200: {passes}/{min(200,total)} PASS")
        if total:
            print(f"    Total integrity entries: {total}")
    except Exception as e:
        print(f"  Learning integrity dashboard skipped: {e}")

    # 9. Master map -> abbreviation map loop
    try:
        sys.path.insert(0, str(BASE.parent / "infrastructure" / "abbreviation_map"))
        from abbrev_map import learn_from_master_map
        learn_from_master_map(min_freq=3)
        print("  Abbreviation map updated from master map")
    except Exception as e:
        print(f"  Abbreviation map update skipped: {e}")

    # 10. Free-form chat probe
    try:
        print("→ Free-form chat probe", flush=True)
        sys.path.insert(0, str(BASE.parent / "orchestrator"))
        from master_orchestrator_v0 import run_orchestrator
        import random, datetime
        prompts = [
            "Explain the concept of emergence",
            "What is the capital of France?",
            "Summarize the difference between a dog and a cat",
            "Why is the sky blue?",
            "What did you learn from Wikidata today?"
        ]
        q = random.choice(prompts)
        res = run_orchestrator(q)
        ans = res.get('response') or ''
        # Log probe
        from bitdrop_persistence import save_line
        log_path = BASE.parent / "runtime" / "logs" / "freeform_probe.jsonl"
        save_line(log_path, {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "prompt": q,
            "response": ans,
            "source": res.get('response_source'),
            "len": len(ans)
        })
        print(f"  Free-form probe: {q}", flush=True)
        print(f"  → {ans[:500]}", flush=True)
        print(f"  [source={res.get('response_source')}, len={len(ans)}]", flush=True)
        print("← Free-form chat probe", flush=True)
    except Exception as e:
        print(f"  Free-form probe skipped: {e}", flush=True)

    # 10. Auto emergence trend plot - disabled to avoid Tkinter warnings
    try:
        print("  Emergence plot skipped: disabled to avoid Tkinter warnings", flush=True)
    except Exception as e:
        print(f"  Emergence plot skipped: {e}")

    log_stage("Full cycle", cycle_start)

if __name__ == "__main__":
    print("Starting unlimited improvements loop – press Ctrl+C to stop", flush=True)
    # Run until user stops
    while True:
        run_cycle()
        print("→ Idle 60s – background consolidation", flush=True)
        # Use idle time for light background work
        import datetime
        start_idle = datetime.datetime.utcnow()
        print(f"  Idle start at {start_idle.isoformat()}", flush=True)
        for i in range(60):
            if i % 20 == 0 and i > 0:
                try:
                    # Mini Day Dream tick during idle
                    from day_dream import day_dream_cycle
                    print(f"  Idle tick: mini Day Dream", flush=True)
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(day_dream_cycle)
                        fut.result(timeout=20)
                except Exception:
                    pass
                try:
                    # Mini Teacher Sampler tick during idle
                    print(f"  Idle tick: mini Teacher Sampler", flush=True)
                    from uncertainty_sampler import sample_uncertain
                    from orchestrator.master_orchestrator_v0 import run_orchestrator
                    sys.path.insert(0, str(BASE.parent / "runtime"))
                    from model_adapter import model_adapter
                    from scripts.reasoning_trace_store import store_trace
                    queries = sample_uncertain(k=1)
                    for q in queries:
                        res = run_orchestrator(q)
                        conf = res.get("diagnostics", {}).get("confidence_prior", 0.5)
                        if conf >= 0.7:
                            continue
                        explain_prompt = f"User question: {q}\nExplain step-by-step how you would answer this, including reasoning and concepts."
                        teach_res = model_adapter.generate(explain_prompt)
                        explanation = teach_res.get("answer","")
                        store_trace(q, res.get("response"), explanation, source="teacher_idle", confidence=conf)
                except Exception:
                    pass
            time.sleep(1)
            if i % 10 == 0:
                print(f"  ... {60-i}s remaining", flush=True)
        end_idle = datetime.datetime.utcnow()
        print(f"  Idle end at {end_idle.isoformat()} duration {(end_idle-start_idle).total_seconds():.1f}s", flush=True)
        print("← Idle complete", flush=True) 
