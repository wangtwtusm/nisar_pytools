"""Resumable, range-based NISAR GSLC spatial subsetting.

Public bbox order is WEST, SOUTH, EAST, NORTH. Outputs are explicitly marked
user-derived subsets, not complete NASA products. Signed URLs stay in memory.
Only numpy, h5py, requests, and pyproj are required.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import h5py
import numpy as np
import requests
from pyproj import CRS, Transformer

from nisar_pytools.io.download import _source_id as source_id

IDENT = "/science/LSAR/identification"
GSLC = "/science/LSAR/GSLC"
FORMAT_VERSION = 1


class SubsetError(RuntimeError):
    """Safe-to-log exception; never includes a signed URL or credentials."""


class SourceChangedError(SubsetError):
    """Source identity no longer matches the validated object."""


def _atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _plain(value):
    if isinstance(value, (h5py.Reference, h5py.RegionReference)):
        return "HDF5 reference omitted"
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00")
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _strong_etag(value):
    return isinstance(value, str) and value.startswith('"') and value.endswith('"')


class RangeSource(io.RawIOBase):
    """Seekable HDF5 source with strict identity checks and read-only prefix reuse."""

    def __init__(
        self, url, *, expected_size=None, reuse_prefix_dir=None, session_factory=requests.Session
    ):
        self.original_url = url
        self.source = source_id(url)
        self.session_factory = session_factory
        self.session = session_factory()
        self.thread_local = threading.local()
        self.sessions = [self.session]
        self.lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.position = 0
        self.cache = OrderedDict()
        self.prefix = None
        self.prefix_size = 0
        self.prefix_status = "not_requested"
        self.network_bytes = 0
        self.prefix_bytes_read = 0
        self.network_requests = 0
        self.length = None
        self.etag = None
        self._connect()
        if expected_size is not None and self.length != int(expected_size):
            self.close()
            raise SourceChangedError("Remote size differs from the catalog size")
        if reuse_prefix_dir is not None:
            self._attach_prefix(Path(reuse_prefix_dir))
        super().__init__()

    def _connect(self):
        try:
            response = self.session.get(
                self.original_url,
                headers={"Range": "bytes=0-15", "Accept-Encoding": "identity"},
                stream=True,
                timeout=(30, 120),
            )
            with response:
                match = re.fullmatch(r"bytes 0-15/(\d+)", response.headers.get("Content-Range", ""))
                if response.status_code != 206 or not match:
                    raise SubsetError(
                        f"Identity probe was not a valid 206 response (HTTP {response.status_code})"
                    )
                size = int(match.group(1))
                etag = response.headers.get("ETag")
                if not _strong_etag(etag):
                    raise SubsetError("A strong ETag is required for durable range caching")
                if self.etag is not None and (self.etag != etag or self.length != size):
                    raise SourceChangedError("Remote object changed while refreshing access")
                if response.headers.get("Content-Encoding", "identity").lower() not in (
                    "",
                    "identity",
                ):
                    raise SubsetError("Encoded byte-range responses are not supported")
                data = response.raw.read(17)
                if len(data) != 16 or data[:8] != b"\x89HDF\r\n\x1a\n":
                    raise SubsetError("Identity probe did not contain the expected HDF5 signature")
                self.length, self.etag, self.signed_url = size, etag, response.url
                self.network_bytes += len(data)
                self.network_requests += 1
        except SubsetError:
            raise
        except Exception as error:  # noqa: BLE001 - sanitize all HTTP/redirect failures
            raise SubsetError(
                f"Source identity request failed ({type(error).__name__}); request details omitted"
            ) from None

    @property
    def identity(self):
        return {"source": self.source, "total": self.length, "etag": self.etag}

    @staticmethod
    def _file_identity(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _attach_prefix(self, directory):
        name = Path(urlsplit(self.original_url).path).name
        path = directory / (name + ".part")
        sidecar = directory / (name + ".part.json")
        self.prefix_status = "unavailable"
        if not path.exists() or not sidecar.exists():
            return
        try:
            saved = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            self.prefix_status = "sidecar_invalid"
            return
        if any(saved.get(key) != value for key, value in self.identity.items()):
            self.prefix_status = "source_etag_or_total_mismatch"
            return
        self.prefix = path.open("rb", buffering=0)
        self.prefix_identity = self._file_identity(os.fstat(self.prefix.fileno()))
        self.prefix_size = self.prefix_identity[2]
        if self.prefix_size > self.length:
            self.prefix.close()
            self.prefix = None
            self.prefix_size = 0
            self.prefix_status = "prefix_larger_than_source"
            return
        self.prefix_path = path
        self.prefix_status = "validated_read_only"

    def _prefix_read(self, start, size):
        if (
            self._file_identity(os.fstat(self.prefix.fileno())) != self.prefix_identity
            or self._file_identity(self.prefix_path.stat()) != self.prefix_identity
        ):
            raise SourceChangedError("Read-only partial download changed; prefix reuse stopped")
        data = os.pread(self.prefix.fileno(), size, start)
        if (
            self._file_identity(os.fstat(self.prefix.fileno())) != self.prefix_identity
            or self._file_identity(self.prefix_path.stat()) != self.prefix_identity
        ):
            raise SourceChangedError("Read-only partial download changed while reading")
        if len(data) != size:
            raise SourceChangedError("Read-only prefix was truncated")
        with self.lock:
            self.prefix_bytes_read += size
        return data

    def _thread_session(self):
        if not hasattr(self.thread_local, "session"):
            session = self.session_factory()
            self.thread_local.session = session
            with self.lock:
                self.sessions.append(session)
        return self.thread_local.session

    def _remote_read(self, start, end):
        for attempt in range(4):
            try:
                response = self._thread_session().get(
                    self.signed_url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "If-Range": self.etag,
                        "Accept-Encoding": "identity",
                    },
                    stream=True,
                    timeout=(30, 120),
                )
                with response:
                    if response.status_code == 403 and attempt < 3:
                        with self.refresh_lock:
                            self._connect()
                        continue
                    if response.status_code == 200:
                        raise SourceChangedError(
                            "Server ignored If-Range or changed the object; refused full response"
                        )
                    if response.status_code != 206:
                        raise SubsetError(f"Range request returned HTTP {response.status_code}")
                    if response.headers.get("ETag") != self.etag:
                        raise SourceChangedError("Range response ETag changed")
                    if (
                        response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{self.length}"
                    ):
                        raise SourceChangedError(
                            "Range response bounds or total differ from the request"
                        )
                    if response.headers.get("Content-Encoding", "identity").lower() not in (
                        "",
                        "identity",
                    ):
                        raise SubsetError("Unexpected content encoding on byte range")
                    expected = end - start + 1
                    declared = response.headers.get("Content-Length")
                    if declared is not None and declared != str(expected):
                        raise SubsetError("Range Content-Length mismatch")
                    payload = response.raw.read(expected + 1)
                    if len(payload) != expected:
                        raise SubsetError("Incomplete range response")
                with self.lock:
                    self.network_bytes += len(payload)
                    self.network_requests += 1
                return payload
            except SourceChangedError:
                raise
            except Exception as error:
                if attempt == 3:
                    if isinstance(error, SubsetError):
                        raise
                    raise SubsetError(
                        f"Range request failed ({type(error).__name__}); request details omitted"
                    ) from None
                time.sleep(attempt + 1)
        raise SubsetError("Could not refresh range access")

    def get_range(self, start, size):
        if start < 0 or size < 0 or start + size > self.length:
            raise SubsetError("Invalid source byte interval")
        if size == 0:
            return b""
        prefix_count = max(0, min(size, self.prefix_size - start))
        first = self._prefix_read(start, prefix_count) if prefix_count else b""
        if prefix_count == size:
            return first
        return first + self._remote_read(start + prefix_count, start + size - 1)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = (
            offset
            if whence == io.SEEK_SET
            else self.position + offset
            if whence == io.SEEK_CUR
            else self.length + offset
        )
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END) or position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size <= 0:
            return b""
        if size > 16 * 1024 * 1024:
            raise SubsetError("Refusing an unexpectedly large metadata read")
        if size >= 262144:
            data = self.get_range(self.position, size)
        else:
            pieces, remaining, cursor = [], size, self.position
            while remaining:
                index, offset = divmod(cursor, 65536)
                if index not in self.cache:
                    self.cache[index] = self.get_range(
                        index * 65536, min(65536, self.length - index * 65536)
                    )
                    if len(self.cache) > 128:
                        self.cache.popitem(last=False)
                else:
                    self.cache.move_to_end(index)
                chunk = self.cache[index]
                count = min(remaining, len(chunk) - offset)
                pieces.append(chunk[offset : offset + count])
                cursor += count
                remaining -= count
            data = b"".join(pieces)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self):
        if self.prefix is not None:
            self.prefix.close()
            self.prefix = None
        for session in self.sessions:
            session.close()
        self.cache.clear()
        super().close()


def plan_gslc_subset(file, bbox, *, frequency="A", polarization="HH"):
    """Plan exact pixel-center cropping; bbox is (west, south, east, north)."""
    west, south, east, north = map(float, bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be WEST SOUTH EAST NORTH with ascending bounds")
    frequency = frequency.removeprefix("frequency").upper()
    if frequency not in ("A", "B"):
        raise ValueError("frequency must be A or B")
    group = file[f"{GSLC}/grids/frequency{frequency}"]
    x, y = group["xCoordinates"][()], group["yCoordinates"][()]
    dx, dy = float(group["xCoordinateSpacing"][()]), float(group["yCoordinateSpacing"][()])
    epsg = int(group["projection"][()])
    CRS.from_epsg(epsg)
    if dx <= 0 or dy >= 0 or not np.allclose(np.diff(x), dx) or not np.allclose(np.diff(y), dy):
        raise ValueError("Only regular, north-up GSLC grids are supported")
    xmin, ymin, xmax, ymax = Transformer.from_crs(4326, epsg, always_xy=True).transform_bounds(
        west, south, east, north, densify_pts=41
    )
    c0, c1 = int(np.searchsorted(x, xmin)), int(np.searchsorted(x, xmax, side="right"))
    r0, r1 = int(np.searchsorted(-y, -ymax)), int(np.searchsorted(-y, -ymin, side="right"))
    if r1 <= r0 or c1 <= c0:
        raise ValueError("Requested bbox has no pixel centers in the GSLC grid")
    layers = []
    for name in (polarization, "mask", "inputDataExceptionMask"):
        if name not in group:
            if name == "inputDataExceptionMask":
                continue
            raise ValueError(f"Required dataset {name} is missing")
        dataset = group[name]
        if dataset.shape != (len(y), len(x)) or dataset.chunks is None:
            raise ValueError(f"Expected a chunked, two-dimensional GSLC layer: {name}")
        cy, cx = dataset.chunks
        chunks = []
        for row in range(r0 // cy * cy, r1, cy):
            for column in range(c0 // cx * cx, c1, cx):
                info = dataset.id.get_chunk_info_by_coord((row, column))
                chunks.append(
                    {
                        "row": row,
                        "column": column,
                        "offset": info.byte_offset,
                        "size": int(info.size),
                        "filter_mask": int(info.filter_mask),
                    }
                )
        layers.append(
            {
                "name": name,
                "path": dataset.name,
                "chunks": list(dataset.chunks),
                "compression": dataset.compression,
                "source_shape": list(dataset.shape),
                "selected_chunks": chunks,
                "compressed_bytes": sum(chunk["size"] for chunk in chunks),
            }
        )
    return {
        "format_version": FORMAT_VERSION,
        "bbox_wgs84_w_s_e_n": [west, south, east, north],
        "frequency": frequency,
        "polarization": polarization,
        "epsg": epsg,
        "source_window_rc": [r0, r1, c0, c1],
        "output_shape": [r1 - r0, c1 - c0],
        "spacing_m": [dx, dy],
        "selection_rule": "Pixel centers inside projected bbox envelope",
        "layers": layers,
    }


class ChunkCache:
    """Atomic compressed-chunk cache; each completed entry has a local SHA256."""

    def __init__(self, directory, source):
        self.source = source
        self.directory = Path(directory) / source.source
        self.directory.mkdir(parents=True, exist_ok=True)
        identity_path = self.directory / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != source.identity:
                raise SourceChangedError("Persistent chunk-cache source identity differs")
        else:
            _atomic_json(identity_path, source.identity)
        self.validated = set()
        self.lock = threading.Lock()
        self.cache_bytes_read = 0

    def paths(self, chunk):
        stem = f"{chunk['offset']}-{chunk['size']}"
        return self.directory / (stem + ".bin"), self.directory / (stem + ".json")

    def valid(self, chunk):
        if not chunk["size"] or chunk["offset"] is None:
            return True
        if chunk["offset"] + chunk["size"] <= self.source.prefix_size:
            return True  # Validated read-only source bytes need no duplicate disk cache.
        path, meta_path = self.paths(chunk)
        key = (chunk["offset"], chunk["size"])
        if key in self.validated:
            return True
        try:
            meta = json.loads(meta_path.read_text())
            payload = path.read_bytes()
            valid = (
                meta["offset"] == chunk["offset"]
                and meta["size"] == chunk["size"]
                and len(payload) == chunk["size"]
                and hashlib.sha256(payload).hexdigest() == meta["sha256"]
            )
        except (OSError, ValueError, KeyError):
            return False
        if valid:
            self.validated.add(key)
        return valid

    def store(self, chunk, payload):
        if len(payload) != chunk["size"]:
            raise SubsetError("Incomplete chunk cannot enter the persistent cache")
        path, meta_path = self.paths(chunk)
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _atomic_json(
            meta_path,
            {
                "offset": chunk["offset"],
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        )
        self.validated.add((chunk["offset"], chunk["size"]))

    def payload(self, chunk):
        if chunk["offset"] + chunk["size"] <= self.source.prefix_size:
            return self.source.get_range(chunk["offset"], chunk["size"])
        if not self.valid(chunk):
            self.store(chunk, self.source.get_range(chunk["offset"], chunk["size"]))
        path, meta_path = self.paths(chunk)
        payload = path.read_bytes()
        meta = json.loads(meta_path.read_text())
        if len(payload) != chunk["size"] or hashlib.sha256(payload).hexdigest() != meta["sha256"]:
            self.validated.discard((chunk["offset"], chunk["size"]))
            payload = self.source.get_range(chunk["offset"], chunk["size"])
            self.store(chunk, payload)
        with self.lock:
            self.cache_bytes_read += len(payload)
        return payload

    def populate(self, chunks, *, max_workers=2, progress=None):
        missing = [
            chunk
            for chunk in chunks
            if chunk["size"] and chunk["offset"] is not None and not self.valid(chunk)
        ]
        # Merge only directly adjacent target chunks, never intervening non-target data.
        groups = []
        for chunk in sorted(missing, key=lambda item: item["offset"]):
            if (
                groups
                and groups[-1][-1]["offset"] + groups[-1][-1]["size"] == chunk["offset"]
                and chunk["offset"] + chunk["size"] - groups[-1][0]["offset"] <= 8 * 1024 * 1024
            ):
                groups[-1].append(chunk)
            else:
                groups.append([chunk])

        def transfer(group):
            start = group[0]["offset"]
            end = group[-1]["offset"] + group[-1]["size"]
            data = self.source.get_range(start, end - start)
            for chunk in group:
                offset = chunk["offset"] - start
                self.store(chunk, data[offset : offset + chunk["size"]])

        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [pool.submit(transfer, group) for group in groups]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if progress and (
                    completed == 1
                    or completed == len(groups)
                    or completed % max(1, len(groups) // 100) == 0
                ):
                    progress(
                        {
                            "stage": "download",
                            "completed_ranges": completed,
                            "total_ranges": len(groups),
                            "network_bytes": self.source.network_bytes,
                        }
                    )
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        return {"downloaded_chunk_count": len(missing), "downloaded_range_count": len(groups)}


def _contains_reference(dtype):
    dtype = np.dtype(dtype)
    if h5py.check_dtype(ref=dtype) is not None:
        return True
    if dtype.fields:
        return any(_contains_reference(field[0]) for field in dtype.fields.values())
    if dtype.subdtype:
        return _contains_reference(dtype.subdtype[0])
    return False


def _copy_attrs(source, destination, *, raster=False):
    for key, value in source.attrs.items():
        if key in ("DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME"):
            continue
        if raster and (
            key.startswith(("min_", "max_", "mean_", "sample_stddev_")) or key == "statistics"
        ):
            continue
        # Omit any references that point into the original file.
        if isinstance(value, (h5py.Reference, h5py.RegionReference)) or (
            hasattr(value, "dtype") and _contains_reference(value.dtype)
        ):
            continue
        destination.attrs[key] = value


def _copy_small_dataset(source, destination_group, name):
    if _contains_reference(source.dtype):
        raise SubsetError(
            "Source metadata reference dataset cannot be copied into an independent subset"
        )
    if source.size * source.dtype.itemsize > 4 * 1024 * 1024:
        raise SubsetError("Refused an unexpectedly large metadata dataset")
    result = destination_group.create_dataset(name, data=source[()])
    _copy_attrs(source, result)
    return result


class ChunkDecoder:
    """Decode original HDF5 filter pipelines using a one-chunk in-memory file."""

    def __init__(self, source_dataset):
        self.source_dataset = source_dataset
        self.file = h5py.File("chunk_decoder", "w", driver="core", backing_store=False)
        shape = tuple(source_dataset.chunks)
        space = h5py.h5s.create_simple(shape)
        dcpl = source_dataset.id.get_create_plist().copy()
        dtype = source_dataset.id.get_type().copy()
        self.dataset = h5py.Dataset(h5py.h5d.create(self.file.id, b"data", dtype, space, dcpl=dcpl))

    def decode(self, payload, filter_mask):
        self.dataset.id.write_direct_chunk((0, 0), payload, filter_mask=filter_mask)
        self.dataset.id.flush()
        try:
            result = self.dataset[()]
        except (TypeError, ValueError):
            result = self.dataset.astype(np.dtype([("r", "f2"), ("i", "f2")]))[()]
        if result.dtype.names and {"r", "i"}.issubset(result.dtype.names):
            return result["r"].astype(np.float32) + 1j * result["i"].astype(np.float32)
        return result

    def close(self):
        self.file.close()


def _complex_fill(dataset):
    """Read a complex fill value without relying on h5py's unsupported c4 dtype."""
    try:
        value = dataset.fillvalue
    except (TypeError, ValueError):
        value = np.zeros((), dtype=[("r", "f2"), ("i", "f2")])
        dataset.id.get_create_plist().get_fill_value(value)
    if getattr(value, "dtype", None) is not None and value.dtype.names:
        return np.complex64(float(value["r"]) + 1j * float(value["i"]))
    return np.complex64(value)


def _subset_polygon(x, y, dx, dy, epsg):
    west, east = float(x[0] - dx / 2), float(x[-1] + dx / 2)
    north, south = float(y[0] - dy / 2), float(y[-1] + dy / 2)
    xs = np.concatenate(
        [
            np.linspace(west, east, 21),
            np.full(20, east),
            np.linspace(east, west, 21)[1:],
            np.full(20, west),
        ]
    )
    ys = np.concatenate(
        [
            np.full(21, north),
            np.linspace(north, south, 21)[1:],
            np.full(20, south),
            np.linspace(south, north, 21)[1:],
        ]
    )
    longitude, latitude = Transformer.from_crs(epsg, 4326, always_xy=True).transform(xs, ys)
    return "POLYGON ((" + ",".join(f"{a:.10f} {b:.10f}" for a, b in zip(longitude, latitude)) + "))"


def _verify_subset(path, plan, x, y, *, digests=None, mark_complete=False):
    """Reopen and hash every logical raster, including when resuming a final file."""
    group_path = f"{GSLC}/grids/frequency{plan['frequency']}"
    with h5py.File(path, "r+" if mark_complete else "r") as verified:
        if digests is None:
            digests = json.loads(verified["/subsetProvenance/logical_sha256_json"][()])
        target = verified[group_path]
        if not np.array_equal(target["xCoordinates"][()], x) or not np.array_equal(
            target["yCoordinates"][()], y
        ):
            raise SubsetError("Written subset coordinate vectors differ from the requested window")
        if int(target["projection"][()]) != plan["epsg"]:
            raise SubsetError("Written subset projection differs")
        for layer in plan["layers"]:
            dataset = target[layer["name"]]
            if list(dataset.shape) != plan["output_shape"]:
                raise SubsetError("Written subset raster shape differs")
            digest = hashlib.sha256()
            for row in range(0, dataset.shape[0], 256):
                digest.update(np.ascontiguousarray(dataset[row : row + 256, :]).tobytes())
            if digest.hexdigest() != digests[layer["name"]]:
                raise SubsetError("Written subset logical-data checksum mismatch")
            if (
                dataset.dims[0][0].name != target["yCoordinates"].name
                or dataset.dims[1][0].name != target["xCoordinates"].name
            ):
                raise SubsetError("Written subset dimension scales are inconsistent")
        if mark_complete:
            verified.attrs["subset_complete"] = True
            verified.flush()
    return digests


def _write_subset(file, source, plan, cache, output_path, *, progress=None):
    output_path = Path(output_path)
    temporary = output_path.with_name(output_path.name + ".partial")
    assembly_path = output_path.with_name(output_path.name + ".assembly.json")
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "source": source.identity,
                "bbox": plan["bbox_wgs84_w_s_e_n"],
                "frequency": plan["frequency"],
                "polarization": plan["polarization"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    r0, r1, c0, c1 = plan["source_window_rc"]
    group_path = f"{GSLC}/grids/frequency{plan['frequency']}"
    original_group = file[group_path]
    x, y = original_group["xCoordinates"][c0:c1], original_group["yCoordinates"][r0:r1]
    if output_path.exists():
        with h5py.File(output_path, "r") as existing:
            if existing.attrs.get("subset_fingerprint") != fingerprint or not existing.attrs.get(
                "subset_complete"
            ):
                raise FileExistsError("Existing output is not this completed subset")
        digests = _verify_subset(output_path, plan, x, y)
        return {
            "path": str(output_path.resolve()),
            "already_complete": True,
            "logical_sha256": digests,
        }
    ownership = {"subset_fingerprint": fingerprint, "partial_path": str(temporary.resolve())}
    if temporary.exists():
        try:
            if json.loads(assembly_path.read_text()) != ownership:
                raise SubsetError("Unrelated partial output exists; preserved it")
        except (OSError, ValueError):
            raise SubsetError("Unrecognized partial HDF5; preserved it") from None
        temporary.unlink()  # Restart local assembly from verified compressed chunks.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(assembly_path, ownership)
    with h5py.File(temporary, "w") as output:
        output.attrs.update(
            {
                "subset_fingerprint": fingerprint,
                "subset_complete": False,
                "isDerivedSubset": True,
                "derivedSubsetDescription": "User-derived GSLC spatial subset, not an original complete NASA product",
                "is_original_nasa_product": False,
                "product_identity": "USER-DERIVED spatial subset; NOT an original complete NASA NISAR GSLC product",
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        original_ident = output.require_group("/sourceMetadata/identification")
        identification = output.require_group(IDENT)
        for name, item in file[IDENT].items():
            if isinstance(item, h5py.Dataset):
                _copy_small_dataset(item, original_ident, name)
                if name not in ("granuleId", "boundingPolygon", "isFullFrame", "listOfFrequencies"):
                    _copy_small_dataset(item, identification, name)
        granule = (
            _plain(file[f"{IDENT}/granuleId"][()])
            if f"{IDENT}/granuleId" in file
            else source.source
        )
        identification["granuleId"] = np.bytes_(granule + "_USER_DERIVED_SUBSET")
        identification["boundingPolygon"] = np.bytes_(
            _subset_polygon(x, y, *plan["spacing_m"], plan["epsg"])
        )
        identification["isFullFrame"] = np.bytes_("False")
        identification["listOfFrequencies"] = np.array([plan["frequency"].encode()])
        target = output.require_group(group_path)
        for name in (
            "centerFrequency",
            "rangeBandwidth",
            "azimuthBandwidth",
            "xCoordinateSpacing",
            "yCoordinateSpacing",
            "projection",
        ):
            _copy_small_dataset(original_group[name], target, name)
        for name, values in (("xCoordinates", x), ("yCoordinates", y)):
            axis = target.create_dataset(name, data=values)
            _copy_attrs(original_group[name], axis)
            axis.make_scale(name)
        target["listOfPolarizations"] = np.array([plan["polarization"].encode()])
        for relative in (
            "metadata/processingInformation/parameters",
            "metadata/processingInformation/inputs",
            "metadata/orbit",
        ):
            original_path = f"{GSLC}/{relative}"
            if original_path in file:
                destination = output.require_group(original_path)
                destination.attrs["metadata_scope"] = (
                    "Source-product metadata, retained for subset interpretation"
                )
                for name, item in file[original_path].items():
                    if (
                        isinstance(item, h5py.Dataset)
                        and item.size * item.dtype.itemsize <= 4 * 1024 * 1024
                    ):
                        _copy_small_dataset(item, destination, name)
        output.require_group("/subsetProvenance")["plan_json"] = json.dumps(plan)
        output["/subsetProvenance"].attrs["metadata_scope"] = (
            "Selected identification, grid, direct processing-parameter/input/orbit datasets only; calibration and other source metadata groups omitted. Original identification retained under /sourceMetadata."
        )
        output["/subsetProvenance"].attrs["integrity"] = (
            "Validated source strong ETag and byte ranges; local SHA256 chunk cache. Original full-file checksum is NOT a checksum of this derived subset."
        )
        digests = {}
        for layer in plan["layers"]:
            original = file[layer["path"]]
            is_complex = layer["name"] == plan["polarization"]
            dtype = np.dtype("complex64") if is_complex else original.dtype
            fillvalue = _complex_fill(original) if is_complex else original.fillvalue
            dataset = target.create_dataset(
                layer["name"],
                shape=(r1 - r0, c1 - c0),
                dtype=dtype,
                chunks=(min(512, r1 - r0), min(512, c1 - c0)),
                compression="gzip",
                compression_opts=4,
                fillvalue=fillvalue,
            )
            _copy_attrs(original, dataset, raster=True)
            dataset.dims[0].attach_scale(target["yCoordinates"])
            dataset.dims[1].attach_scale(target["xCoordinates"])
            dataset.attrs["source_statistics_omitted"] = True
            mapping = {(entry["row"], entry["column"]): entry for entry in layer["selected_chunks"]}
            cy, cx = layer["chunks"]
            decoded_cache = OrderedDict()
            decoder = ChunkDecoder(original)
            logical_digest = hashlib.sha256()
            try:
                # Row strips yield a canonical row-major content hash and bounded memory.
                for rr in range(r0, r1, 256):
                    rend = min(rr + 256, r1)
                    strip = np.empty((rend - rr, c1 - c0), dtype=dtype)
                    strip[...] = fillvalue
                    for sr in range(rr // cy * cy, rend, cy):
                        for sc in range(c0 // cx * cx, c1, cx):
                            entry = mapping[(sr, sc)]
                            if not entry["size"] or entry["offset"] is None:
                                continue
                            key = (sr, sc)
                            if key not in decoded_cache:
                                decoded_cache[key] = decoder.decode(
                                    cache.payload(entry), entry["filter_mask"]
                                )
                                if len(decoded_cache) > math.ceil((c1 - c0) / cx) + 3:
                                    decoded_cache.popitem(last=False)
                            a0, a1, b0, b1 = (
                                max(rr, sr),
                                min(rend, sr + cy),
                                max(c0, sc),
                                min(c1, sc + cx),
                            )
                            strip[a0 - rr : a1 - rr, b0 - c0 : b1 - c0] = decoded_cache[key][
                                a0 - sr : a1 - sr, b0 - sc : b1 - sc
                            ]
                    dataset[rr - r0 : rend - r0, :] = strip
                    logical_digest.update(np.ascontiguousarray(strip).tobytes())
                    if progress and (rr == r0 or rend == r1 or (rr - r0) // 256 % 10 == 0):
                        progress(
                            {
                                "stage": "write",
                                "layer": layer["name"],
                                "completed_rows": rend - r0,
                                "total_rows": r1 - r0,
                            }
                        )
            finally:
                decoder.close()
            digests[layer["name"]] = logical_digest.hexdigest()
            output.flush()
        output["/subsetProvenance"]["source_identity_json"] = json.dumps(source.identity)
        output["/subsetProvenance"]["logical_sha256_json"] = json.dumps(digests)
        output.flush()
    _verify_subset(temporary, plan, x, y, digests=digests, mark_complete=True)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    assembly_path.unlink(missing_ok=True)
    return {
        "path": str(output_path.resolve()),
        "already_complete": False,
        "logical_sha256": digests,
    }


def download_gslc_subset(
    url,
    output_path,
    bbox,
    *,
    cache_dir,
    reuse_prefix_dir=None,
    frequency="A",
    polarization="HH",
    max_workers=2,
    expected_size=None,
    plan_only=False,
    progress=None,
):
    """Download a derived LSAR GSLC subset with a persistent HTTP range cache.

    Parameters
    ----------
    url : str
        Original download URL. Earthdata authentication uses Requests/.netrc.
    output_path : str or pathlib.Path
        Destination HDF5. The file is explicitly marked as a derived subset.
        A different existing output is never overwritten.
    bbox : sequence of float
        WGS84 ``(west, south, east, north)`` in degrees. Pixel centers inside
        its projected bounding envelope are selected without resampling.
    cache_dir : str or pathlib.Path
        Persistent compressed-chunk cache. Keep this directory to resume.
    reuse_prefix_dir : str or pathlib.Path, optional
        Directory containing whole-granule ``.part`` / ``.part.json`` files
        created by :func:`download_urls`. Matching source, strong ETag and
        total length are required; partial files are opened read-only.
    frequency : str
        ``A`` or ``B`` (``frequencyA`` / ``frequencyB`` also accepted).
    polarization : str
        One complex channel, such as ``HH``. Its validity masks are included.
    max_workers : int
        Concurrent range requests, from 1 to 8. Default 2.
    expected_size : int, optional
        Exact source byte count from the catalog, checked against the server.
    plan_only : bool
        Read metadata and report needed chunks without exporting pixel arrays.
    progress : callable, optional
        Receives progress dictionaries without source URLs.

    Returns
    -------
    dict
        Source identity, bbox/window, chunk byte estimates, and cache/prefix
        reuse statistics. Completed exports also include the output path,
        transferred bytes and logical SHA256 digests for each exported layer.

    Notes
    -----
    Requires chunked layers on a regular north-up LSAR GSLC grid, HTTP Range
    support and a strong ETag. Only selected identification, grid and direct
    processing/input/orbit metadata are retained; the output is not a complete
    official product. Cached chunks are checked before reuse. Do not write to
    the same output/cache concurrently from separate calls.
    """
    if max_workers < 1 or max_workers > 8:
        raise ValueError("max_workers must be between 1 and 8")
    with (
        RangeSource(url, expected_size=expected_size, reuse_prefix_dir=reuse_prefix_dir) as source,
        h5py.File(source, "r") as file,
    ):
        plan = plan_gslc_subset(file, bbox, frequency=frequency, polarization=polarization)
        all_chunks = [
            chunk
            for layer in plan["layers"]
            for chunk in layer["selected_chunks"]
            if chunk["size"] and chunk["offset"] is not None
        ]
        cache = ChunkCache(cache_dir, source)
        total = sum(chunk["size"] for chunk in all_chunks)
        prefix_reusable = sum(
            min(chunk["size"], max(0, source.prefix_size - chunk["offset"])) for chunk in all_chunks
        )
        cache_reusable = sum(
            chunk["size"] - min(chunk["size"], max(0, source.prefix_size - chunk["offset"]))
            for chunk in all_chunks
            if chunk["offset"] + chunk["size"] > source.prefix_size and cache.valid(chunk)
        )
        report = {
            "derived_subset": True,
            "source_identity": source.identity,
            "bbox_wgs84_w_s_e_n": plan["bbox_wgs84_w_s_e_n"],
            "output_shape": plan["output_shape"],
            "source_window_rc": plan["source_window_rc"],
            "selected_chunk_count": len(all_chunks),
            "selected_compressed_bytes": total,
            "prefix_status": source.prefix_status,
            "prefix_size_bytes": source.prefix_size,
            "prefix_reusable_chunk_bytes": prefix_reusable,
            "cache_reusable_chunk_bytes": cache_reusable,
            "estimated_missing_chunk_bytes": max(0, total - prefix_reusable - cache_reusable),
            "layer_summary": [
                {key: value for key, value in layer.items() if key != "selected_chunks"}
                for layer in plan["layers"]
            ],
        }
        _atomic_json(
            Path(cache_dir) / source.source / "last_plan.json", {"plan": plan, "report": report}
        )
        if progress:
            progress({"stage": "plan", **report})
        if plan_only:
            return report
        report.update(cache.populate(all_chunks, max_workers=max_workers, progress=progress))
        report.update(_write_subset(file, source, plan, cache, output_path, progress=progress))
        report.update(
            {
                "network_bytes": source.network_bytes,
                "network_range_requests": source.network_requests,
                "prefix_bytes_read": source.prefix_bytes_read,
                "cache_bytes_read": cache.cache_bytes_read,
                "output_size_bytes": Path(output_path).stat().st_size,
                "is_original_nasa_product": False,
            }
        )
        _atomic_json(
            Path(output_path).with_suffix(Path(output_path).suffix + ".subset.json"), report
        )
        if progress:
            progress({"stage": "complete", **report})
        return report
