"""Sanity unit test for project initialization."""

import incident_search


def test_import_and_version():
    assert incident_search.__version__ == "0.1.0"
