"""Synthetic byte-range/HDF5 fixtures; no remote access or observed SAR data."""

import io
import json
import re

import h5py
import numpy as np
import pytest

from nisar_pytools.io import subset as m

URL = "https://example.invalid/source.h5"
ETAG = '"synthetic-object-v1"'


class Response:
    def __init__(self, payload, headers, status=206):
        self.raw = io.BytesIO(payload)
        self.headers, self.status_code = headers, status
        self.url = "https://signed.invalid/source?secret=never-persist"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Server:
    def __init__(self, data, mode=None):
        self.data, self.mode, self.requests = data, mode, []

    def factory(self):
        parent = self

        class Session:
            def get(self, url, headers, **kwargs):
                start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
                parent.requests.append((start, end, headers))
                h = {
                    "Content-Range": f"bytes {start}-{end}/{len(parent.data)}",
                    "ETag": ETAG,
                    "Content-Length": str(end - start + 1),
                }
                body, status = parent.data[start : end + 1], 206
                if start != 0 or end != 15:
                    if parent.mode == "etag":
                        h["ETag"] = '"synthetic-object-v2"'
                    if parent.mode == "total":
                        h["Content-Range"] = f"bytes {start}-{end}/{len(parent.data) + 1}"
                    if parent.mode == "bounds":
                        h["Content-Range"] = f"bytes {start + 1}-{end}/{len(parent.data)}"
                    if parent.mode == "short":
                        body = body[:-1]
                    if parent.mode == "gzip":
                        h["Content-Encoding"] = "gzip"
                    if parent.mode == "ignored":
                        status = 200
                return Response(body, h, status)

            def close(self):
                pass

        return Session()


def fixture(path, compound=False):
    shape, chunks = (13, 17), (4, 5)
    # EPSG:4326 permits exact bbox/window tests without projecting corners.
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    z = (values / 4 + 1j * (values % 7)).astype(np.complex64)
    with h5py.File(path, "w") as f:
        ident = f.require_group(m.IDENT)
        for key, val in {
            "granuleId": b"SYNTHETIC",
            "productType": b"GSLC",
            "boundingPolygon": b"FULL SOURCE POLYGON",
            "listOfFrequencies": [b"A", b"B"],
            "isFullFrame": b"True",
            "trackNumber": 112,
        }.items():
            ident[key] = val
        g = f.require_group(m.GSLC + "/grids/frequencyA")
        for key, val in {
            "centerFrequency": 1.229e9,
            "rangeBandwidth": 20e6,
            "azimuthBandwidth": 1261.0,
            "xCoordinateSpacing": 0.01,
            "yCoordinateSpacing": -0.01,
            "projection": 4326,
        }.items():
            g[key] = val
        g["xCoordinates"] = 93 + np.arange(shape[1]) * 0.01
        g["yCoordinates"] = 33 - np.arange(shape[0]) * 0.01
        g["xCoordinates"].make_scale("xCoordinates")
        g["yCoordinates"].make_scale("yCoordinates")
        g["listOfPolarizations"] = [b"HH", b"HV"]
        if compound:
            packed = np.zeros(shape, dtype=[("r", "f2"), ("i", "f2")])
            packed["r"], packed["i"] = z.real, z.imag
            data = packed
        else:
            data = z
        ds = g.create_dataset(
            "HH", shape=shape, dtype=data.dtype, chunks=chunks, compression="gzip", shuffle=True
        )
        # Last bottom-right chunk remains unallocated and must preserve zero fill.
        ds[:12, :] = data[:12, :]
        ds[12:, :15] = data[12:, :15]
        ds.attrs["units"] = "1"
        ds.attrs["min_real_value"] = -99
        ds.attrs["valid_min"] = 0
        ds.dims[0].attach_scale(g["yCoordinates"])
        ds.dims[1].attach_scale(g["xCoordinates"])
        mask = np.ones(shape, np.uint8)
        mask[3, 4] = 255
        mask[6, 8] = 0
        ds = g.create_dataset("mask", data=mask, chunks=chunks, compression="gzip")
        ds.attrs["description"] = "0 invalid, 1..254 valid, 255 fill"
        exc = np.zeros(shape, np.uint16)
        exc[5, 6] = 2
        ds = g.create_dataset("inputDataExceptionMask", data=exc, chunks=chunks, compression="gzip")
        ds.attrs["description"] = "0: no anomaly, 2: sample slip"
        p = f.require_group(m.GSLC + "/metadata/processingInformation/parameters")
        p["topographicFlatteningApplied"] = b"True"
    z[12:, 15:] = 0
    return path, z, mask, exc


@pytest.mark.parametrize("mode", ["etag", "total", "bounds", "short", "gzip", "ignored"])
def test_range_protocol_fail_closed(tmp_path, monkeypatch, mode):
    path, *_ = fixture(tmp_path / "original.h5")
    server = Server(path.read_bytes(), mode)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    with m.RangeSource(URL, session_factory=server.factory) as source:
        cache = m.ChunkCache(tmp_path / "cache", source)
        entry = {"offset": 100, "size": 50}
        with pytest.raises(m.SubsetError):
            cache.populate([entry])
        assert not list(cache.directory.glob("*.bin"))
        assert all(x[2]["If-Range"] == ETAG for x in server.requests[1:])


def test_prefix_identity_change_and_cross_boundary(tmp_path):
    path, *_ = fixture(tmp_path / "original.h5")
    data = path.read_bytes()
    server = Server(data)
    prefix = tmp_path / "source.h5.part"
    prefix.write_bytes(data[:1234])
    (tmp_path / "source.h5.part.json").write_text(
        json.dumps({"source": m.source_id(URL), "total": len(data), "etag": ETAG})
    )
    with m.RangeSource(URL, session_factory=server.factory, reuse_prefix_dir=tmp_path) as source:
        assert source.prefix_status == "validated_read_only"
        assert source.get_range(1200, 100) == data[1200:1300]
        assert server.requests[-1][:2] == (1234, 1299)
        prefix.write_bytes(data[:1235])
        with pytest.raises(m.SourceChangedError):
            source.get_range(50, 50)


@pytest.mark.parametrize("key,value", [("source", "wrong"), ("total", 1), ("etag", '"changed"')])
def test_mismatched_prefix_not_reused(tmp_path, key, value):
    path, *_ = fixture(tmp_path / "original.h5")
    data = path.read_bytes()
    server = Server(data)
    (tmp_path / "source.h5.part").write_bytes(data[:1000])
    saved = {"source": m.source_id(URL), "total": len(data), "etag": ETAG}
    saved[key] = value
    (tmp_path / "source.h5.part.json").write_text(json.dumps(saved))
    with m.RangeSource(URL, session_factory=server.factory, reuse_prefix_dir=tmp_path) as source:
        assert source.prefix_size == 0
        assert source.get_range(100, 100) == data[100:200]
        assert source.prefix_bytes_read == 0


def test_cache_corruption_and_missing_sidecar_resume(tmp_path):
    path, *_ = fixture(tmp_path / "original.h5")
    data = path.read_bytes()
    server = Server(data)
    with m.RangeSource(URL, session_factory=server.factory) as source:
        cache = m.ChunkCache(tmp_path / "cache", source)
        a, b = {"offset": 100, "size": 50}, {"offset": 150, "size": 50}
        result = cache.populate([a, b])
        assert result["downloaded_range_count"] == 1
        requests = len(server.requests)
        cache = m.ChunkCache(tmp_path / "cache", source)
        result = cache.populate([a, b])
        assert result["downloaded_range_count"] == 0
        cache.paths(a)[0].write_bytes(b"x" * 50)
        assert cache.payload(a) == data[100:150]  # Includes modification after validation.
        cache.paths(b)[1].unlink()
        cache = m.ChunkCache(tmp_path / "cache", source)
        assert cache.populate([a, b])["downloaded_chunk_count"] == 1
        assert len(server.requests) == requests + 2


@pytest.mark.parametrize("compound", [False, True])
@pytest.mark.parametrize(
    "bbox",
    [
        (93.03, 32.91, 93.115, 32.975),
        (93.0, 32.879, 93.17, 33.001),
        (93.05, 32.925, 93.095, 32.965),
    ],
)
def test_subset_exact_arrays_metadata_and_local_scales(tmp_path, monkeypatch, compound, bbox):
    path, z, mask, exc = fixture(tmp_path / "original.h5", compound=compound)
    server = Server(path.read_bytes())
    Original = m.RangeSource
    monkeypatch.setattr(
        m, "RangeSource", lambda *a, **k: Original(*a, **k, session_factory=server.factory)
    )
    output = tmp_path / "subset.h5"
    report = m.download_gslc_subset(URL, output, bbox, cache_dir=tmp_path / "cache", max_workers=2)
    r0, r1, c0, c1 = report["source_window_rc"]
    with h5py.File(output) as f:
        assert f.attrs["isDerivedSubset"] and f.attrs["subset_complete"]
        assert not f.attrs["is_original_nasa_product"]
        g = f[m.GSLC + "/grids/frequencyA"]
        for name, expected in [("HH", z), ("mask", mask), ("inputDataExceptionMask", exc)]:
            np.testing.assert_array_equal(g[name][()], expected[r0:r1, c0:c1])
            np.testing.assert_array_equal(g[name].dims[0][0], 33 - np.arange(r0, r1) * 0.01)
            np.testing.assert_array_equal(g[name].dims[1][0], 93 + np.arange(c0, c1) * 0.01)
        assert g["HH"].dtype == np.complex64
        assert g["HH"].fillvalue == 0
        assert "min_real_value" not in g["HH"].attrs
        assert g["HH"].attrs["valid_min"] == 0
        assert g["inputDataExceptionMask"].attrs["description"] == "0: no anomaly, 2: sample slip"
        assert f[m.IDENT + "/boundingPolygon"][()] != b"FULL SOURCE POLYGON"
        assert f["/sourceMetadata/identification/boundingPolygon"][()] == b"FULL SOURCE POLYGON"
        assert f[m.IDENT + "/listOfFrequencies"][()].tolist() == [b"A"]
        assert g["listOfPolarizations"][()].tolist() == [b"HH"]
        assert (
            f[m.GSLC + "/metadata/processingInformation/parameters/topographicFlatteningApplied"][
                ()
            ]
            == b"True"
        )
    assert report["derived_subset"]
    assert not list(tmp_path.glob("*.partial"))
    result = m.download_gslc_subset(URL, output, bbox, cache_dir=tmp_path / "cache")
    assert result["already_complete"]
    assert result["downloaded_chunk_count"] == 0
    assert "signed.invalid" not in json.dumps(report)
    assert "secret=" not in output.with_suffix(".h5.subset.json").read_text()


def test_interrupted_local_write_not_published_then_resumes(tmp_path, monkeypatch):
    path, *_ = fixture(tmp_path / "original.h5")
    server = Server(path.read_bytes())
    Original = m.RangeSource
    monkeypatch.setattr(
        m, "RangeSource", lambda *a, **k: Original(*a, **k, session_factory=server.factory)
    )
    out = tmp_path / "subset.h5"
    bbox = (93.03, 32.91, 93.115, 32.975)

    def interrupted(event):
        if event["stage"] == "write":
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        m.download_gslc_subset(URL, out, bbox, cache_dir=tmp_path / "cache", progress=interrupted)
    assert not out.exists()
    assert out.with_name(out.name + ".partial").exists()
    result = m.download_gslc_subset(URL, out, bbox, cache_dir=tmp_path / "cache")
    assert out.exists() and result["downloaded_chunk_count"] == 0


def test_owned_corrupt_partial_recovers_and_unknown_partial_preserved(tmp_path, monkeypatch):
    path, *_ = fixture(tmp_path / "original.h5")
    server = Server(path.read_bytes())
    Original = m.RangeSource
    monkeypatch.setattr(
        m, "RangeSource", lambda *a, **k: Original(*a, **k, session_factory=server.factory)
    )
    out = tmp_path / "subset.h5"
    bbox = (93.03, 32.91, 93.115, 32.975)

    def interrupt(event):
        if event["stage"] == "write":
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        m.download_gslc_subset(URL, out, bbox, cache_dir=tmp_path / "cache", progress=interrupt)
    partial = out.with_name(out.name + ".partial")
    partial.write_bytes(b"truncated HDF5")
    result = m.download_gslc_subset(URL, out, bbox, cache_dir=tmp_path / "cache")
    assert result["downloaded_chunk_count"] == 0 and out.exists() and not partial.exists()
    out2 = tmp_path / "unknown.h5"
    partial2 = out2.with_name(out2.name + ".partial")
    partial2.write_bytes(b"unrelated")
    with pytest.raises(m.SubsetError, match="Unrecognized partial"):
        m.download_gslc_subset(URL, out2, bbox, cache_dir=tmp_path / "cache")
    assert partial2.read_bytes() == b"unrelated"


@pytest.mark.parametrize("corruption", ["raster", "coordinate"])
def test_existing_completed_subset_is_reverified(tmp_path, monkeypatch, corruption):
    path, *_ = fixture(tmp_path / "original.h5")
    server = Server(path.read_bytes())
    Original = m.RangeSource
    monkeypatch.setattr(
        m, "RangeSource", lambda *a, **k: Original(*a, **k, session_factory=server.factory)
    )
    out = tmp_path / "subset.h5"
    bbox = (93.03, 32.91, 93.115, 32.975)
    m.download_gslc_subset(URL, out, bbox, cache_dir=tmp_path / "cache")
    with h5py.File(out, "r+") as f:
        g = f[m.GSLC + "/grids/frequencyA"]
        if corruption == "raster":
            g["HH"][0, 0] = 999 + 999j
        else:
            g["xCoordinates"][0] = 999
    with pytest.raises(m.SubsetError):
        m.download_gslc_subset(URL, out, bbox, cache_dir=tmp_path / "cache")


def test_reference_metadata_never_copied_as_invalid_links(tmp_path):
    with (
        h5py.File(tmp_path / "refs.h5", "w") as source,
        h5py.File(tmp_path / "copy.h5", "w") as target,
    ):
        source["values"] = [1, 2]
        ref_dtype = np.dtype([("a", "f4"), ("ref", h5py.ref_dtype)])
        refs = np.empty(1, dtype=ref_dtype)
        refs["a"] = 1
        refs["ref"] = source["values"].ref
        source.create_dataset("references", data=refs)
        source["values"].attrs["nested_reference"] = refs
        source["values"].attrs["units"] = "m"
        m._copy_small_dataset(source["values"], target, "values")
        assert "nested_reference" not in target["values"].attrs
        with pytest.raises(m.SubsetError, match="reference dataset"):
            m._copy_small_dataset(source["references"], target, "references")
        assert "references" not in target


def test_public_api_subset_reads_with_existing_package_reader(tmp_path, monkeypatch):
    from nisar_pytools import download_gslc_subset, open_nisar
    from nisar_pytools.utils.metadata import get_bounding_polygon, get_slc

    path, complex_values, masks, _ = fixture(tmp_path / "original.h5")
    server = Server(path.read_bytes())
    original_source = m.RangeSource
    monkeypatch.setattr(
        m,
        "RangeSource",
        lambda *args, **kwargs: original_source(*args, **kwargs, session_factory=server.factory),
    )
    output = tmp_path / "scene_derived_subset.h5"
    report = download_gslc_subset(
        URL,
        output,
        (93.03, 32.91, 93.115, 32.975),
        cache_dir=tmp_path / "cache",
    )
    r0, r1, c0, c1 = report["source_window_rc"]
    tree = open_nisar(output)
    slc = get_slc(tree, valid_mask=False)
    assert slc.dims == ("y", "x")
    assert slc.rio.crs.to_epsg() == 4326
    np.testing.assert_array_equal(slc.compute(), complex_values[r0:r1, c0:c1])
    np.testing.assert_array_equal(slc.x, 93 + np.arange(c0, c1) * 0.01)
    np.testing.assert_array_equal(slc.y, 33 - np.arange(r0, r1) * 0.01)
    expected_mask = (masks[r0:r1, c0:c1] != 0) & (masks[r0:r1, c0:c1] != 255)
    np.testing.assert_array_equal(np.isfinite(get_slc(tree).compute()), expected_mask)
    assert tree.attrs["isDerivedSubset"]
    assert not tree.attrs["is_original_nasa_product"]
    assert get_bounding_polygon(tree).is_valid


def test_plan_does_not_double_count_cache_overlapping_prefix(tmp_path, monkeypatch):
    path, *_ = fixture(tmp_path / "original.h5")
    data = path.read_bytes()
    server = Server(data)
    bbox = (93.03, 32.91, 93.115, 32.975)
    with h5py.File(path) as f:
        plan = m.plan_gslc_subset(f, bbox)
    chunks = [c for layer in plan["layers"] for c in layer["selected_chunks"] if c["size"]]
    crossing = sorted(chunks, key=lambda c: c["offset"])[2]
    cut = crossing["offset"] + crossing["size"] // 2
    (tmp_path / "source.h5.part").write_bytes(data[:cut])
    (tmp_path / "source.h5.part.json").write_text(
        json.dumps({"source": m.source_id(URL), "total": len(data), "etag": ETAG})
    )
    with m.RangeSource(URL, session_factory=server.factory) as source:
        cache = m.ChunkCache(tmp_path / "cache", source)
        cache.store(crossing, data[crossing["offset"] : crossing["offset"] + crossing["size"]])
    Original = m.RangeSource
    monkeypatch.setattr(
        m, "RangeSource", lambda *a, **k: Original(*a, **k, session_factory=server.factory)
    )
    result = m.download_gslc_subset(
        URL,
        tmp_path / "subset.h5",
        bbox,
        cache_dir=tmp_path / "cache",
        reuse_prefix_dir=tmp_path,
        plan_only=True,
    )
    expected = sum(
        c["size"] - min(c["size"], max(0, cut - c["offset"])) for c in chunks if c is not crossing
    )
    assert result["estimated_missing_chunk_bytes"] == expected
    assert result["cache_reusable_chunk_bytes"] == crossing["size"] - (cut - crossing["offset"])
