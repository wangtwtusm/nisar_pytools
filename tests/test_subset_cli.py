"""CLI contract for persistent remote GSLC subset downloads."""

import json

import pytest
import requests

from nisar_pytools.cli import main


def test_subset_cli_passes_asymmetric_bbox_without_reordering(tmp_path, monkeypatch, capsys):
    observed = {}

    def subset(url, output, bbox, **options):
        observed.update(url=url, output=output, bbox=bbox, **options)
        options["progress"]("Validated cached range")
        return {"derived_subset": True, "path": str(output), "bbox_wgs84_w_s_e_n": bbox}

    monkeypatch.setattr("nisar_pytools.io.subset.download_gslc_subset", subset)
    output = tmp_path / "scene_derived_subset.h5"
    cache = tmp_path / "cache"
    prefix = tmp_path / "full_downloads"
    result = main(
        [
            "download-gslc-subset",
            "https://example.com/scene.h5",
            "--output",
            str(output),
            "--bbox-wgs84",
            "92.9",
            "31.6",
            "94.7",
            "33.2",
            "--cache-dir",
            str(cache),
            "--reuse-prefix-dir",
            str(prefix),
            "--freq",
            "frequencyB",
            "--pol",
            "HV",
            "--max-workers",
            "3",
            "--expected-size",
            "123456789",
            "--plan-only",
        ]
    )
    assert result == 0
    assert observed["bbox"] == (92.9, 31.6, 94.7, 33.2)
    assert observed["output"] == output
    assert observed["cache_dir"] == cache
    assert observed["reuse_prefix_dir"] == prefix
    assert observed["frequency"] == "B"
    assert observed["polarization"] == "HV"
    assert observed["expected_size"] == 123456789
    assert observed["max_workers"] == 3
    assert observed["plan_only"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["bbox_wgs84_w_s_e_n"] == [92.9, 31.6, 94.7, 33.2]
    assert payload["derived_subset"] is True


def test_subset_cli_has_persistent_default_cache(tmp_path, monkeypatch, capsys):
    observed = {}

    def subset(url, output, bbox, **options):
        observed.update(options)
        return {"path": str(output)}

    monkeypatch.setattr("nisar_pytools.io.subset.download_gslc_subset", subset)
    main(
        [
            "download-gslc-subset",
            "https://example.com/scene.h5",
            "--output",
            str(tmp_path / "subset.h5"),
            "--bbox-wgs84",
            "92.9",
            "31.6",
            "94.7",
            "33.2",
        ]
    )
    assert observed["cache_dir"] == tmp_path / ".nisar_subset_cache"
    assert observed["frequency"] == "A"
    assert observed["polarization"] == "HH"
    assert observed["max_workers"] == 2
    assert observed["plan_only"] is False
    json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    "bbox",
    [
        [92.9, 94.7, 31.6, 33.2],  # A WEST EAST SOUTH NORTH tuple must not silently work.
        [94.7, 31.6, 92.9, 33.2],
        [92.9, 33.2, 94.7, 31.6],
        [92.9, float("nan"), 94.7, 33.2],
    ],
)
def test_subset_cli_rejects_invalid_bbox_before_network(tmp_path, monkeypatch, bbox):
    def should_not_run(*args, **kwargs):
        pytest.fail("Invalid bounding box reached the network API")

    monkeypatch.setattr("nisar_pytools.io.subset.download_gslc_subset", should_not_run)
    with pytest.raises(SystemExit, match="WEST SOUTH EAST NORTH"):
        main(
            [
                "download-gslc-subset",
                "https://example.com/scene.h5",
                "--output",
                str(tmp_path / "subset.h5"),
                "--bbox-wgs84",
                *map(str, bbox),
            ]
        )


def test_subset_cli_error_does_not_expose_signed_url(tmp_path, monkeypatch, capsys):
    def failure(*args, **kwargs):
        raise requests.RequestException("Request failed: https://host/file?token=SECRET")

    monkeypatch.setattr("nisar_pytools.io.subset.download_gslc_subset", failure)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "download-gslc-subset",
                "https://example.com/scene.h5?token=SECRET",
                "--output",
                str(tmp_path / "subset.h5"),
                "--bbox-wgs84",
                "92.9",
                "31.6",
                "94.7",
                "33.2",
            ]
        )
    assert "RequestException" in str(exc.value)
    assert "SECRET" not in str(exc.value)
    assert "SECRET" not in capsys.readouterr().out
    assert exc.value.__suppress_context__ is True


def test_subset_cli_preserves_safe_resource_change_error(tmp_path, monkeypatch):
    from nisar_pytools.io.subset import SourceChangedError

    def changed(*args, **kwargs):
        raise SourceChangedError("Remote object changed while refreshing access")

    monkeypatch.setattr("nisar_pytools.io.subset.download_gslc_subset", changed)
    with pytest.raises(SystemExit, match="Remote object changed"):
        main(
            [
                "download-gslc-subset",
                "https://example.com/scene.h5",
                "--output",
                str(tmp_path / "subset.h5"),
                "--bbox-wgs84",
                "92.9",
                "31.6",
                "94.7",
                "33.2",
            ]
        )


def test_subset_cli_help_labels_derived_output_and_bbox_order(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["download-gslc-subset", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "derived subset" in help_text
    assert "WEST SOUTH EAST NORTH" in help_text
    assert "--plan-only" in help_text
