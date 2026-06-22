"""Tests for BLME task certification metadata."""

from blme.core import _register_all_tasks
from blme.registry import list_tasks
from blme.task_metadata import (
    TASK_CERTIFICATION,
    VALID_CERTIFICATION_STATUSES,
    validate_certification_coverage,
)


def test_certification_metadata_covers_registered_tasks():
    _register_all_tasks()
    missing, extra = validate_certification_coverage(list_tasks())

    assert not missing, f"Tasks missing certification metadata: {missing}"
    assert not extra, f"Certification metadata for unregistered tasks: {extra}"


def test_certification_status_values_are_known():
    for task_name, meta in TASK_CERTIFICATION.items():
        assert meta.status in VALID_CERTIFICATION_STATUSES, task_name
        assert meta.papers, f"{task_name} missing paper/provenance description"

