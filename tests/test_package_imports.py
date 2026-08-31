"""Day 1 smoke test.

Deliberately trivial: it exists so CI is green from the very first commit
rather than red until day 4-5 when the first real logic lands. Delete this
once tests/test_features.py etc. exist and actually exercise something.
"""

import dsp


def test_package_has_version():
    assert dsp.__version__ == "0.1.0"


def test_subpackages_import():
    import dsp.api  # noqa: F401
    import dsp.features  # noqa: F401
    import dsp.ingestion  # noqa: F401
    import dsp.inventory  # noqa: F401
    import dsp.models  # noqa: F401
    import dsp.monitoring  # noqa: F401
    import dsp.orchestration  # noqa: F401
