"""Basic tests for project data models."""

from student.models import MinimalSource


def test_minimal_source() -> None:
    """MinimalSource stores evaluator metadata."""
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/README.md",
        first_character_index=0,
        last_character_index=42,
    )

    assert source.file_path.endswith("README.md")
    assert source.first_character_index == 0
    assert source.last_character_index == 42
