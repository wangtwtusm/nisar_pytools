"""Download NISAR products with persistent HTTP resumption and validation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import h5py
import requests

log = logging.getLogger(__name__)


def _url_filename(url: str) -> str:
    """Extract the filename from a URL, stripping query strings."""
    return urlsplit(url).path.rsplit("/", 1)[-1]


def validate_h5_quick(filepath: Path) -> bool:
    """Quick validation — check file is valid HDF5 and can open."""
    try:
        with h5py.File(filepath, "r") as f:
            _ = list(f.keys())
        return True
    except Exception:
        return False


def validate_h5_thorough(filepath: Path) -> bool:
    """Thorough validation — verify NISAR structure and read a small dataset.

    Checks that the file:
    1. Is valid HDF5
    2. Contains a NISAR science band (LSAR or SSAR)
    3. Contains ``identification/productType``
    4. Can read the product type and access the data group
    """
    try:
        with h5py.File(filepath, "r") as f:
            # Detect band (LSAR or SSAR)
            band = None
            if "science/LSAR" in f:
                band = "LSAR"
            elif "science/SSAR" in f:
                band = "SSAR"
            if band is None:
                log.warning("Missing science/LSAR or science/SSAR: %s", filepath.name)
                return False

            pt_path = f"science/{band}/identification/productType"
            if pt_path not in f:
                log.warning("Missing productType: %s", filepath.name)
                return False

            # Read product type (single read)
            product_type = f[pt_path][()]
            if isinstance(product_type, bytes):
                product_type = product_type.decode().strip()

            # Try to access the data group
            for data_group in ["grids", "swaths"]:
                grids_path = f"science/{band}/{product_type}/{data_group}"
                if grids_path in f:
                    _ = list(f[grids_path].keys())
                    return True

            log.warning("Missing product data group: %s", filepath.name)
        return False
    except Exception as e:
        log.warning("Validation failed for %s: %s", filepath.name, e)
        return False


class _DownloadError(Exception):
    """A safe-to-log transfer error (never includes a URL or credentials)."""


class _RestartDownload(_DownloadError):
    """The partial file cannot safely be reused."""


def _source_id(url: str) -> str:
    """Identify the original endpoint without retaining credentials/signed queries."""
    parts = urlsplit(url)
    endpoint = f"{parts.scheme}://{parts.hostname}:{parts.port}{parts.path}"
    return hashlib.sha256(endpoint.encode()).hexdigest()


def _read_metadata(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def _write_metadata(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(data))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}


def _validator(metadata: dict) -> str | None:
    etag = metadata.get("etag")
    if isinstance(etag, str) and etag.startswith('"') and etag.endswith('"'):
        return etag  # Weak ETags cannot be used with If-Range.
    return metadata.get("last_modified") or None


def _validator_matches(metadata: dict, response: requests.Response, *, required=False) -> bool:
    """If-Range protects 206s; a 416 needs explicit identity confirmation."""
    key = "ETag" if _validator(metadata) == metadata.get("etag") else "Last-Modified"
    current = response.headers.get(key)
    return current == _validator(metadata) if current or required else True


def _content_length(response: requests.Response) -> int | None:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    if not value.isdigit():
        raise _RestartDownload("Invalid Content-Length")
    return int(value)


def download_urls(
    urls: list[str],
    out_directory: str | Path,
    reprocess: bool = False,
    max_workers: int = 4,
    retries: int = 3,
    timeout: int = 60,
    validate: bool = True,
) -> list[Path]:
    """Download in parallel, resuming interrupted transfers across calls.

    Transfers use persistent ``<filename>.part`` and ``.part.json`` files.
    Range requests require a strong ETag or Last-Modified validator; otherwise
    a partial download restarts safely. Servers ignoring Range also restart.
    Known total length and optional NISAR HDF5 validation are checked before
    atomic publication. ``.download.json`` records the completed file's length
    and filesystem identity.

    Existing files are checked against saved length metadata and, when
    ``validate=True`` (default), NISAR HDF5 structure. Legacy files without
    metadata retain the existing-file behavior, with HDF5 validation enabled.
    ``reprocess=True`` starts a fresh transfer. ``retries`` is the maximum
    number of attempts per URL; failed files are excluded from the returned,
    alphabetically sorted paths. INFO logging reports progress every 30 seconds.

    Do not run concurrent calls targeting the same filenames/directory.
    """
    if retries < 1 or max_workers < 1:
        raise ValueError("retries and max_workers must be positive")
    out_directory = Path(out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)

    url_to_filename: dict[str, str] = {}
    seen_filenames: set[str] = set()
    for url in urls:
        fname = _url_filename(url)
        if not fname or fname in {".", ".."} or "\\" in fname or any(ord(c) < 32 for c in fname):
            raise ValueError("Download URL has an invalid filename")
        if fname in seen_filenames:
            log.warning("Duplicate filename %s: keeping the first URL", fname)
            continue
        seen_filenames.add(fname)
        url_to_filename[url] = fname

    def _download_one(url: str) -> Path | None:
        out_fp = out_directory / url_to_filename[url]
        part = out_fp.with_name(out_fp.name + ".part")
        part_meta = out_fp.with_name(out_fp.name + ".part.json")
        done_meta = out_fp.with_name(out_fp.name + ".download.json")
        source = _source_id(url)

        def valid_h5(path: Path) -> bool:
            return not (validate and out_fp.suffix.lower() == ".h5") or validate_h5_thorough(path)

        def reset_partial() -> None:
            part.unlink(missing_ok=True)
            part_meta.unlink(missing_ok=True)

        def publish(metadata: dict) -> Path:
            total = metadata.get("total")
            if not isinstance(total, int) or total < 0:
                raise _DownloadError("No trustworthy total length; partial file retained")
            if part.stat().st_size != total:
                raise _DownloadError("Incomplete transfer; partial file retained")
            if not valid_h5(part):
                raise _RestartDownload("Downloaded HDF5 failed validation")
            # Bind the record to this inode before rename: a crash between the
            # two replacements must not pair new metadata with an old final file,
            # even when both files have the same length. Rename preserves identity.
            _write_metadata(done_meta, {**metadata, "file_identity": _file_identity(part)})
            os.replace(part, out_fp)
            part_meta.unlink(missing_ok=True)
            log.info("Downloaded %s (%d bytes)", out_fp.name, total)
            return out_fp

        if out_fp.exists() and out_fp.stat().st_size > 0 and not reprocess:
            completed = _read_metadata(done_meta)
            size_ok = not completed or (
                completed.get("source") == source
                and completed.get("total") == out_fp.stat().st_size
            )
            identity_ok = "file_identity" not in completed or (
                completed["file_identity"] == _file_identity(out_fp)
            )
            if size_ok and identity_ok and valid_h5(out_fp):
                log.info("Verified existing file: %s", out_fp.name)
                return out_fp
            log.warning("Existing file failed checks; downloading again: %s", out_fp.name)
        if reprocess:
            reset_partial()

        with requests.Session() as session:
            for attempt in range(retries):
                try:
                    metadata = _read_metadata(part_meta)
                    offset = part.stat().st_size if part.exists() else 0
                    total = metadata.get("total")
                    if offset and (
                        metadata.get("source") != source
                        or not _validator(metadata)
                        or (isinstance(total, int) and offset > total)
                    ):
                        reset_partial()
                        offset = 0
                    headers = {"Accept-Encoding": "identity"}
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                        headers["If-Range"] = _validator(metadata)
                        log.info("Resuming %s at byte %d", out_fp.name, offset)
                    else:
                        log.info("Starting download: %s", out_fp.name)

                    response = session.get(url, headers=headers, stream=True, timeout=timeout)
                    with response:
                        if response.status_code == 416:
                            match = re.fullmatch(
                                r"bytes \*/(\d+)", response.headers.get("Content-Range", ""),
                            )
                            if (
                                offset and match and int(match[1]) == offset
                                and total in (None, offset)
                                and _validator_matches(metadata, response, required=True)
                            ):
                                metadata["total"] = offset
                                return publish(metadata)
                            raise _RestartDownload("Range rejected; restarting transfer")
                        response.raise_for_status()
                        encoding = response.headers.get("Content-Encoding", "identity").lower()
                        if encoding != "identity":
                            raise _RestartDownload("Unexpected encoded response")
                        length = _content_length(response)
                        if response.status_code == 206:
                            match = re.fullmatch(
                                r"bytes (\d+)-(\d+)/(\d+)",
                                response.headers.get("Content-Range", ""),
                            )
                            if not offset or not match:
                                raise _RestartDownload("Unexpected or invalid Content-Range")
                            start, end, new_total = map(int, match.groups())
                            if (
                                start != offset or not start <= end < new_total
                                or (length is not None and length != end - start + 1)
                                or (total is not None and total != new_total)
                                or not _validator_matches(metadata, response)
                            ):
                                raise _RestartDownload("Range or resource identity changed")
                            expected = end - start + 1
                            metadata["total"] = new_total
                        elif response.status_code == 200:
                            if offset:
                                log.info(
                                    "Server returned full response; restarting %s", out_fp.name,
                                )
                            offset = 0
                            expected = length
                            metadata = {
                                "source": source,
                                "total": length,
                                "etag": response.headers.get("ETag"),
                                "last_modified": response.headers.get("Last-Modified"),
                            }
                        else:
                            raise _RestartDownload("Unexpected success status")

                        # Truncate before replacing metadata on a fresh response,
                        # so an interruption cannot attach new metadata to old bytes.
                        with part.open("ab" if offset else "wb") as output:
                            _write_metadata(part_meta, metadata)
                            received = 0
                            last_report = time.monotonic()
                            for chunk in response.iter_content(chunk_size=256 * 1024):
                                if not chunk:
                                    continue
                                if expected is not None and received + len(chunk) > expected:
                                    raise _RestartDownload("Response exceeds advertised length")
                                output.write(chunk)
                                received += len(chunk)
                                now = time.monotonic()
                                if now - last_report >= 30:
                                    log.info(
                                        "%s: %d / %s bytes", out_fp.name,
                                        offset + received, metadata.get("total") or "unknown",
                                    )
                                    last_report = now
                            output.flush()
                            os.fsync(output.fileno())
                        if expected is not None and received != expected:
                            raise _DownloadError(
                                "Response shorter than advertised; partial retained"
                            )
                        return publish(metadata)
                except Exception as exc:
                    if isinstance(exc, _RestartDownload):
                        reset_partial()
                    # requests errors can contain signed redirect URLs/passwords.
                    reason = str(exc) if isinstance(exc, _DownloadError) else type(exc).__name__
                    log.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt + 1, retries, out_fp.name, reason,
                    )
                    if attempt + 1 < retries:
                        time.sleep(1.5)
        return None

    downloaded: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one, url): name for url, name in url_to_filename.items()
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    downloaded.append(result)
            except Exception as exc:
                log.error("Download failed for %s: %s", futures[future], type(exc).__name__)
    failed = len(url_to_filename) - len(downloaded)
    if failed:
        log.warning("%d downloads failed; rerun to resume retained partial files", failed)
    return sorted(downloaded)
