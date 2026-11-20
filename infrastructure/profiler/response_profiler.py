"""
Response time profiler for orchestrator.
Tracks time per stage and logs bottlenecks.
"""
import time
from collections import defaultdict

class ResponseProfiler:
    def __init__(self):
        self.times = defaultdict(list)
        self.enabled = True

    def start(self, stage):
        if not self.enabled:
            return time.perf_counter()
        return time.perf_counter()

    def end(self, stage, start):
        if not self.enabled:
            return
        elapsed = time.perf_counter() - start
        self.times[stage].append(elapsed)

    def report(self):
        report = {}
        for stage, vals in self.times.items():
            if vals:
                report[stage] = {
                    'count': len(vals),
                    'total_s': sum(vals),
                    'avg_ms': sum(vals)/len(vals)*1000,
                    'max_ms': max(vals)*1000,
                    'min_ms': min(vals)*1000
                }
        return report

    def log(self, path='runtime/logs/response_profile.jsonl'):
        import json, time
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'ts': time.time(), 'report': self.report()}) + '\n')

_profiler = ResponseProfiler()

def profile_stage(stage):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = _profiler.start(stage)
            res = func(*args, **kwargs)
            _profiler.end(stage, start)
            return res
        return wrapper
    return decorator

def get_profiler():
    return _profiler
