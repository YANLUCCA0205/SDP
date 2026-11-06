"""Testa preparo das capturas sem alterar originais ou criar labels falsas."""
import cv2
import numpy as np
import pytest

from training.prepare_captures import prepare


def test_prepare_preserves_images_without_inventing_labels(tmp_path):
    source = tmp_path / "received"
    source.mkdir()
    ok, encoded = cv2.imencode(".jpg", np.full((40, 60, 3), 128, dtype=np.uint8))
    assert ok
    (source / "a.jpg").write_bytes(encoded.tobytes())
    (source / "b.jpg").write_bytes(encoded.tobytes())
    (source / "c.jpg").write_bytes(b"bad")
    output = tmp_path / "to_annotate"
    result = prepare(source, output)
    assert result["found"] == 3
    assert result["copied"] == 1
    assert result["duplicates"] == 1
    assert result["invalid"] == 1
    assert (output / "images" / "a.jpg").read_bytes() == (source / "a.jpg").read_bytes()
    assert not list(output.rglob("*.txt"))
    assert "NOT_ANNOTATED" in (output / "manifest.csv").read_text(encoding="utf-8-sig")
    assert len(list(source.glob("*.jpg"))) == 3


def test_prepare_requires_existing_images(tmp_path):
    with pytest.raises(ValueError, match="nao encontrada"):
        prepare(tmp_path / "missing", tmp_path / "out")
