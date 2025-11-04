# src/utils/timing_helpers.py
import contextlib

def section(timer, name: str):
    """Return a context manager that records <name> if timer is not None; else no-op."""
    return timer.section(name) if timer is not None else contextlib.nullcontext()

def add_metric(timer, name: str, value):
    if timer is not None:
        timer.add_metric(name, float(value))

