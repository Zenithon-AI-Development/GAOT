# src/utils/timer.py
import time, contextlib, collections, torch

from contextlib import contextmanager

_ACTIVE = None

def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class StatEMA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.v = None
    def add(self, x):
        self.v = x if self.v is None else (1 - self.alpha) * self.v + self.alpha * x
    @property
    def value(self): return 0.0 if self.v is None else self.v

class IterProfiler:
    """
    Low-overhead section timer. Use:
        with prof.section("forward"): ...
    """
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.sections = collections.defaultdict(StatEMA)
        self._t0 = time.perf_counter()
        self._last_fetch_end = None

    @contextlib.contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        _sync_if_cuda(); t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync_if_cuda(); dt = (time.perf_counter() - t0) * 1000.0
            self.sections[name].add(dt)
    
    def add_metric(self, name: str, value: float):
        self.sections[name].add(float(value))

    def record_data_wait(self, t_fetch_start):
        """Call right after 'for batch in loader' yields a batch, passing the timestamp taken before calling 'next()'."""
        if not self.enabled: return
        _sync_if_cuda()
        dt = (time.perf_counter() - t_fetch_start) * 1000.0
        self.sections["data"].add(dt)

    def iter_reset_anchor(self):
        return time.perf_counter()

    def summary_ms(self):
        return {k: v.value for k, v in self.sections.items()}

    # def format_table(self):
    #     s = self.summary_ms()
    #     total = sum(s.values()) or 1.0
    #     keys = ["data","forward","encode","process","decode","loss","backward","step","zero_grad","sched"]
    #     rows = []
    #     for k in keys:
    #         if k in s:
    #             ms = s[k]; pct = 100.0 * ms / total
    #             rows.append(f"{k:10s}: {ms:8.2f} ms  ({pct:5.1f}%)")
    #     rows.append(f"{'total':10s}: {total:8.2f} ms")
    #     return " | ".join(rows)

    def format_table(self, top_k=None):
        s = self.summary_ms()
        total = sum(s.values()) or 1.0
        items = sorted(s.items(), key=lambda kv: -kv[1])
        if top_k is not None:
            items = items[:top_k]
        rows = [f"{k:10s}: {ms:8.2f} ms  ({100*ms/total:5.1f}%)" for k, ms in items]
        rows.append(f"{'total':10s}: {total:8.2f} ms")
        return " | ".join(rows)

