import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.citation_utils import extract_citations


def test_bullet_citations():
    text = """
    - resnet_38
    - faster_rcnn_12
    """

    result = extract_citations(text)

    assert result == ["resnet_38", "faster_rcnn_12"]


def test_star_citations():
    text = """
    * unet_5
    * yolo_20
    """

    result = extract_citations(text)

    assert result == ["unet_5", "yolo_20"]


def test_bare_citations():
    text = """
    resnet_10
    vit_25
    """

    result = extract_citations(text)

    assert result == ["resnet_10", "vit_25"]


def test_invalid_citations_are_rejected():
    text = """
    invalid_2
    unknown_10
    resnet_invalid
    """

    result = extract_citations(text)

    assert result == []


def test_duplicate_citations_are_removed():
    text = """
    resnet_38
    resnet_38
    - resnet_38
    """

    result = extract_citations(text)

    assert result == ["resnet_38"]