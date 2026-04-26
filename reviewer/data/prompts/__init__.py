"""Bundled artifact-review rubric prompts.

The directory is a Python package only so setuptools' ``package-data``
glob can ship the rubric markdown as bundled resources accessible via
``importlib.resources.files("reviewer.data.prompts")`` once the
artifact-review pipeline (``fr_reviewer_19c871ab``) lands. No runtime
imports.
"""
