"""Distill-pipeline transforms.

Each transform consumes a ``ReviewResult`` + ``DistillConfig`` and
returns a possibly-shaped ``ReviewResult``. The pipeline composes
them in a fixed order in ``reviewer.distill.run_pipeline``.

See ``specs/MS-B/spec.md`` for the design context — every transform
must pass the "10×-outlier survives unchanged" feature-preservation
test (an outlier concern in a sea of nits never gets dropped).
"""
