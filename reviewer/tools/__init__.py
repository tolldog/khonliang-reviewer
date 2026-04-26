"""Operator-facing reviewer tools.

These are not bus skills — they're reproducible CLIs that operators
(or CI) run directly to exercise the reviewer outside its agent
runtime. The first one is :mod:`reviewer.tools.benchmark_sweep`,
which iterates the provider registry and produces a calibration
matrix per (backend, model). Future tools land here so the agent
process stays lean.
"""
