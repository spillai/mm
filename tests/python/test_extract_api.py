"""Tests for the public extraction API (``mm.cat`` / ``mm.cat_many``)."""

from __future__ import annotations

from pathlib import Path

import pytest

import mm
from mm.cat_utils.base_utils import CatOpts
from mm.extract import Extraction


class TestCatOpts:
    """CatOpts must be constructible without every field, for library callers."""

    def test_defaults_fill_unset_fields(self):
        opts = CatOpts(mode="accurate")
        assert opts.mode == "accurate"
        assert opts.dry_run is False
        assert opts.verbose is False
        assert opts.pipelines == {}

    def test_mutable_defaults_are_not_shared(self):
        a, b = CatOpts(), CatOpts()
        a.encode_overrides["strategy"] = "tile"
        assert b.encode_overrides == {}

    def test_unknown_field_rejected(self):
        with pytest.raises(TypeError, match="bogus"):
            CatOpts(bogus=True)

    def test_cli_still_passes_every_field(self):
        opts = CatOpts(**{k: None for k in CatOpts.__slots__})
        assert opts.mode is None  # explicit values win over defaults


class TestCat:
    """``mm.cat`` on a text file — no LLM, no network."""

    @pytest.fixture
    def sample(self, tmp_path: Path) -> Path:
        p = tmp_path / "notes.txt"
        p.write_text("\n".join(f"line {i}" for i in range(1, 11)))
        return p

    def test_returns_extraction(self, sample: Path):
        result = mm.cat(sample)
        assert isinstance(result, Extraction)
        assert result.kind == "text"
        assert result.mode == "fast"
        assert "line 1" in result.content

    def test_stringifies_to_content(self, sample: Path):
        result = mm.cat(sample)
        assert str(result) == result.content
        assert f"{result}".startswith("line 1")

    def test_head_limit(self, sample: Path):
        assert mm.cat(sample, n=3).content.splitlines() == ["line 1", "line 2", "line 3"]

    def test_tail_limit(self, sample: Path):
        assert mm.cat(sample, n=-2).content.splitlines() == ["line 9", "line 10"]

    def test_expands_user_paths(self, sample: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HOME", str(sample.parent))
        assert "line 1" in mm.cat(f"~/{sample.name}").content

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            mm.cat(tmp_path / "nope.txt")


class TestCatMany:
    @pytest.fixture
    def samples(self, tmp_path: Path) -> list[Path]:
        paths = []
        for i in range(4):
            p = tmp_path / f"f{i}.txt"
            p.write_text(f"content {i}")
            paths.append(p)
        return paths

    def test_preserves_input_order(self, samples: list[Path]):
        results = mm.cat_many(samples)
        assert [r.path.name for r in results] == [p.name for p in samples]
        assert [r.content for r in results] == [f"content {i}" for i in range(4)]

    def test_empty_input(self):
        assert mm.cat_many([]) == []

    def test_forwards_kwargs(self, samples: list[Path]):
        results = mm.cat_many(samples, n=1)
        assert all(len(r.content.splitlines()) <= 1 for r in results)


class TestPublicSurface:
    def test_exported_from_package_root(self):
        for name in ("cat", "cat_many", "Extraction"):
            assert name in mm.__all__
            assert getattr(mm, name) is not None

    def test_cli_and_api_share_one_core(self):
        """The CLI must delegate to mm.extract, not carry its own copy."""
        from mm.commands import cat as cat_cmd

        assert cat_cmd.mm_extract is mm.extract
