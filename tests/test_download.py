"""Tests for nisar_pytools.io.download."""

import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from nisar_pytools.io.download import (
    download_urls,
    validate_h5_quick,
    validate_h5_thorough,
)


def _make_valid_nisar_h5(path):
    """Create a minimal valid NISAR HDF5 file."""
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "CF-1.7"
        ident = f.create_group("science/LSAR/identification")
        ident.create_dataset("productType", data=b"GSLC")
        grp = f.create_group("science/LSAR/GSLC/grids/frequencyA")
        grp.create_dataset("xCoordinates", data=np.arange(10.0))
    return path


@pytest.fixture
def http_source(tmp_path, monkeypatch):
    """Real HTTP transfers, including connections closed partway through a body."""
    monkeypatch.setattr("nisar_pytools.io.download.time.sleep", lambda _: None)
    payload_path = _make_valid_nisar_h5(tmp_path / "source.h5")
    with h5py.File(payload_path, "a") as source:
        source.create_dataset("test_payload", data=np.arange(300_000, dtype="i4"))
    state = SimpleNamespace(
        payload=payload_path.read_bytes(), etag='"version-1"',
        modified="Wed, 21 Oct 2026 12:00:00 GMT", actions=[], requests=[],
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            index = len(state.requests)
            state.requests.append(dict(self.headers))
            action = state.actions[min(index, len(state.actions) - 1)] if state.actions else {}
            payload = action.get("payload", state.payload)
            etag = action.get("etag", state.etag)
            modified = action.get("modified", state.modified)
            offset = int(self.headers.get("Range", "bytes=0-")[6:-1])
            requested = "Range" in self.headers
            validator = etag if etag and not etag.startswith("W/") else modified
            matches = self.headers.get("If-Range") == validator
            status = 200
            body = payload
            headers = {}
            if etag:
                headers["ETag"] = etag
            if modified:
                headers["Last-Modified"] = modified
            serve_range = requested and not action.get("ignore_range")
            if serve_range and (matches or action.get("force_range")):
                if offset >= len(payload):
                    status = 416
                    body = b""
                    headers["Content-Range"] = f"bytes */{len(payload)}"
                else:
                    status = 206
                    body = payload[offset:]
                    headers["Content-Range"] = f"bytes {offset}-{len(payload)-1}/{len(payload)}"
            headers["Content-Length"] = str(len(body))
            headers.update(action.get("headers", {}))
            for name in action.get("omit_headers", []):
                headers.pop(name, None)
            status = action.get("status", status)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            if action.get("drop"):
                self.wfile.write(body[:600_000])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            else:
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The downloader rejected headers before consuming the body.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True,
    )
    worker.start()
    state.url = f"http://127.0.0.1:{server.server_port}/product.h5"
    state.out = tmp_path / "downloads"
    state.file = state.out / "product.h5"
    state.part = state.out / "product.h5.part"
    state.meta = state.out / "product.h5.part.json"
    yield state
    server.shutdown()
    server.server_close()
    worker.join()


def _download(source, **kwargs):
    return download_urls([source.url], source.out, max_workers=1, timeout=2, **kwargs)


def _interrupted(source):
    source.actions = [{"drop": True}]
    assert _download(source, retries=1) == []
    assert 0 < source.part.stat().st_size < len(source.payload)
    assert not source.file.exists()
    return source.part.stat().st_size


class TestValidateH5Quick:
    def test_valid_file(self, tmp_path):
        fp = _make_valid_nisar_h5(tmp_path / "valid.h5")
        assert validate_h5_quick(fp) is True

    def test_invalid_file(self, tmp_path):
        fp = tmp_path / "bad.h5"
        fp.write_text("not hdf5")
        assert validate_h5_quick(fp) is False

    def test_missing_file(self, tmp_path):
        assert validate_h5_quick(tmp_path / "nope.h5") is False


class TestValidateH5Thorough:
    def test_valid_nisar(self, tmp_path):
        fp = _make_valid_nisar_h5(tmp_path / "valid.h5")
        assert validate_h5_thorough(fp) is True

    def test_ssar_band(self, tmp_path):
        fp = tmp_path / "ssar.h5"
        with h5py.File(fp, "w") as f:
            ident = f.create_group("science/SSAR/identification")
            ident.create_dataset("productType", data=b"GSLC")
            f.create_group("science/SSAR/GSLC/grids/frequencyA")
        assert validate_h5_thorough(fp) is True

    def test_missing_science_lsar(self, tmp_path):
        fp = tmp_path / "no_lsar.h5"
        with h5py.File(fp, "w") as f:
            f.create_group("other")
        assert validate_h5_thorough(fp) is False

    def test_missing_product_type(self, tmp_path):
        fp = tmp_path / "no_pt.h5"
        with h5py.File(fp, "w") as f:
            f.create_group("science/LSAR/identification")
        assert validate_h5_thorough(fp) is False

    def test_truncated_file(self, tmp_path):
        fp = tmp_path / "truncated.h5"
        fp.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 100)
        assert validate_h5_thorough(fp) is False

    def test_not_hdf5(self, tmp_path):
        fp = tmp_path / "text.h5"
        fp.write_text("just text")
        assert validate_h5_thorough(fp) is False

    def test_missing_product_data(self, tmp_path):
        fp = tmp_path / "metadata_only.h5"
        with h5py.File(fp, "w") as f:
            f.create_dataset("science/LSAR/identification/productType", data=b"GSLC")
        assert validate_h5_thorough(fp) is False


class TestDownloadUrls:
    def test_skips_existing(self, tmp_path):
        existing = tmp_path / "file.h5"
        existing.write_bytes(b"data")

        fps = download_urls(
            ["https://example.com/file.h5"], tmp_path, validate=False
        )
        assert len(fps) == 1
        assert fps[0] == existing

    def test_returns_sorted(self, tmp_path):
        for name in ["c.h5", "a.h5", "b.h5"]:
            (tmp_path / name).write_bytes(b"x")

        urls = [f"https://example.com/{n}" for n in ["c.h5", "a.h5", "b.h5"]]
        fps = download_urls(urls, tmp_path, validate=False)
        names = [fp.name for fp in fps]
        assert names == sorted(names)

    def test_validation_passes_valid_h5(self, tmp_path):
        _make_valid_nisar_h5(tmp_path / "valid.h5")
        fps = download_urls(
            ["https://example.com/valid.h5"], tmp_path, validate=True
        )
        assert len(fps) == 1

    def test_failed_download_excluded(self, tmp_path):
        """A URL that fails all retries should be excluded, not crash."""
        # No files exist, no mock session — will fail to connect
        # Use max_workers=1 and retries=1 for speed
        fps = download_urls(
            ["https://0.0.0.0:1/nonexistent.txt"],
            tmp_path,
            validate=False,
            max_workers=1,
            retries=1,
            timeout=1,
        )
        assert len(fps) == 0

    def test_creates_directory(self, tmp_path):
        out = tmp_path / "new" / "nested"
        # Just test that directory is created with an existing file
        out.mkdir(parents=True, exist_ok=True)
        (out / "f.txt").write_bytes(b"x")
        fps = download_urls(["https://example.com/f.txt"], out, validate=False)
        assert out.is_dir()
        assert len(fps) == 1

    def test_duplicate_filenames_downloaded_once(self, tmp_path):
        """Two URLs colliding on filename yield a single download and entry."""
        (tmp_path / "SAME.h5").write_bytes(b"x")

        fps = download_urls(
            [
                "https://example.com/a_BETA_V1/SAME.h5",
                "https://example.com/a_PROVISIONAL_V1/SAME.h5",
            ],
            tmp_path,
            validate=False,
        )
        assert fps == [tmp_path / "SAME.h5"]

    def test_query_string_stripped_from_filename(self, tmp_path):
        """A presigned URL's query string must not end up in the filename."""
        (tmp_path / "file.h5").write_bytes(b"x")

        fps = download_urls(
            ["https://0.0.0.0:1/file.h5?X-Amz-Signature=abc"],
            tmp_path,
            validate=False,
            max_workers=1,
            retries=1,
            timeout=1,
        )
        assert fps == [tmp_path / "file.h5"]

    def test_atomic_write_no_partial_files(self, tmp_path):
        """After a failed download, no partial .h5 file should remain."""
        download_urls(
            ["https://0.0.0.0:1/test.h5"],
            tmp_path,
            validate=False,
            max_workers=1,
            retries=1,
            timeout=1,
        )
        # Failed transfers are never published under the final .h5 name.
        assert not (tmp_path / "test.h5").exists()
        # No .tmp files left behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestResumableHTTPDownloads:
    def test_real_disconnect_resumes_during_retry(self, http_source):
        s = http_source
        s.actions = [{"drop": True}, {}]
        assert _download(s, retries=2) == [s.file]
        assert s.file.read_bytes() == s.payload
        assert s.requests[1]["Range"] == "bytes=524288-"
        assert s.requests[1]["If-Range"] == s.etag
        assert s.requests[1]["Accept-Encoding"] == "identity"
        assert not s.part.exists() and not s.meta.exists()
        completed = json.loads((s.out / "product.h5.download.json").read_text())
        assert completed["total"] == len(s.payload)

    def test_resumes_after_separate_call(self, http_source):
        s = http_source
        offset = _interrupted(s)
        s.actions = [{}]
        assert _download(s, retries=1) == [s.file]
        assert s.requests[-1]["Range"] == f"bytes={offset}-"
        assert s.file.read_bytes() == s.payload

    def test_resumes_in_fresh_python_process(self, http_source):
        s = http_source
        offset = _interrupted(s)
        s.actions = [{}]
        script = (
            "import sys; from nisar_pytools.io.download import download_urls; "
            "paths = download_urls([sys.argv[1]], sys.argv[2], retries=1); "
            "sys.exit(0 if len(paths) == 1 else 1)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, s.url, str(s.out)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert s.requests[-1]["Range"] == f"bytes={offset}-"
        assert s.file.read_bytes() == s.payload

    def test_server_ignores_range_restarts_without_appending(self, http_source):
        s = http_source
        _interrupted(s)
        s.actions = [{"ignore_range": True}]
        assert _download(s, retries=1) == [s.file]
        assert s.file.read_bytes() == s.payload

    @pytest.mark.parametrize("bad_range", ["bytes 0-8/9", "bytes 524288-524287/1200000", "garbage"])
    def test_invalid_range_is_not_appended_or_published(self, http_source, bad_range):
        s = http_source
        _interrupted(s)
        s.actions = [{"headers": {"Content-Range": bad_range}}]
        assert _download(s, retries=1) == []
        assert not s.file.exists()
        assert not s.part.exists()

    def test_changed_resource_returns_full_new_file(self, http_source):
        s = http_source
        _interrupted(s)
        # Same length, different valid HDF5 content and validator.
        replacement = s.payload[:-1] + bytes([s.payload[-1] ^ 1])
        s.actions = [{"payload": replacement, "etag": '"version-2"'}]
        assert _download(s, retries=1) == [s.file]
        assert s.file.read_bytes() == replacement

    def test_noncompliant_206_changed_validator_retries_fresh(self, http_source):
        s = http_source
        _interrupted(s)
        s.actions = [
            {}, {"etag": '"version-2"', "force_range": True}, {"etag": '"version-2"'},
        ]
        assert _download(s, retries=2) == [s.file]
        assert "Range" not in s.requests[-1]
        assert s.file.read_bytes() == s.payload

    def test_changed_total_rejected(self, http_source):
        s = http_source
        _interrupted(s)
        s.actions = [{"payload": s.payload + b"extra"}]
        assert _download(s, retries=1) == []
        assert not s.file.exists()

    def test_complete_partial_confirmed_by_416(self, http_source):
        s = http_source
        _interrupted(s)
        s.part.write_bytes(s.payload)
        s.actions = [{}]
        assert _download(s, retries=1) == [s.file]
        assert s.requests[-1]["Range"] == f"bytes={len(s.payload)}-"
        assert s.file.read_bytes() == s.payload

    def test_416_requires_matching_validator(self, http_source):
        s = http_source
        _interrupted(s)
        s.part.write_bytes(s.payload)
        s.actions = [{"omit_headers": ["ETag"]}]
        assert _download(s, retries=1) == []
        assert not s.file.exists()

    def test_416_wrong_size_is_not_success(self, http_source):
        s = http_source
        _interrupted(s)
        s.actions = [{"status": 416, "headers": {"Content-Range": "bytes */12"}}]
        assert _download(s, retries=1) == []
        assert not s.file.exists()

    def test_416_full_but_corrupt_hdf5_is_not_published(self, http_source):
        s = http_source
        _interrupted(s)
        s.part.write_bytes(b"x" * len(s.payload))
        s.actions = [{}]
        assert _download(s, retries=1) == []
        assert not s.file.exists()
        assert not s.part.exists()

    def test_last_modified_used_for_weak_etag(self, http_source):
        s = http_source
        s.etag = 'W/"weak"'
        _interrupted(s)
        s.actions = [{}]
        assert _download(s, retries=1) == [s.file]
        assert s.requests[-1]["If-Range"] == s.modified

    def test_no_validator_restarts(self, http_source):
        s = http_source
        s.etag = s.modified = None
        _interrupted(s)
        s.actions = [{}]
        assert _download(s, retries=1) == [s.file]
        assert "Range" not in s.requests[-1]

    def test_unknown_length_is_not_success_even_when_hdf5_opens(self, http_source):
        s = http_source
        s.actions = [{"omit_headers": ["Content-Length"]}]
        assert _download(s, retries=1) == []
        assert not s.file.exists()
        assert s.part.read_bytes() == s.payload

    def test_unknown_length_can_be_confirmed_by_later_range(self, http_source):
        s = http_source
        # An unknown-length body that ends early still leaves resumable bytes.
        s.actions = [{"omit_headers": ["Content-Length"], "drop": True}]
        assert _download(s, retries=1) == []
        offset = s.part.stat().st_size
        s.actions = [{}]
        assert _download(s, retries=1) == [s.file]
        assert s.requests[-1]["Range"] == f"bytes={offset}-"
        assert s.file.read_bytes() == s.payload

    def test_unknown_length_complete_file_confirmed_by_416(self, http_source):
        s = http_source
        s.actions = [{"omit_headers": ["Content-Length"]}]
        assert _download(s, retries=1) == []
        s.actions = [{}]
        assert _download(s, retries=1) == [s.file]
        assert s.requests[-1]["Range"] == f"bytes={len(s.payload)}-"
        assert s.file.read_bytes() == s.payload

    def test_206_content_length_must_match_range(self, http_source):
        s = http_source
        _interrupted(s)
        s.actions = [{"headers": {"Content-Length": "42"}}]
        assert _download(s, retries=1) == []
        assert not s.part.exists()
        assert not s.file.exists()

    def test_encoded_response_is_rejected(self, http_source):
        s = http_source
        s.actions = [{"headers": {"Content-Encoding": "gzip"}}]
        assert _download(s, retries=1) == []
        assert not s.file.exists()

    def test_explicitly_empty_non_hdf5_download(self, http_source):
        s = http_source
        s.payload = b""
        s.url = s.url.replace("product.h5", "empty.txt")
        expected = s.out / "empty.txt"
        assert _download(s, retries=1) == [expected]
        assert expected.read_bytes() == b""

    def test_short_response_kept_but_not_success(self, http_source):
        s = http_source
        _interrupted(s)
        offset = s.part.stat().st_size
        # Another actual disconnect during the ranged response.
        assert _download(s, retries=1) == []
        assert s.part.stat().st_size > offset
        assert not s.file.exists()

    def test_corrupt_existing_hdf5_replaced(self, http_source):
        s = http_source
        s.out.mkdir()
        s.file.write_bytes(b"not an HDF5 file")
        assert _download(s, retries=1) == [s.file]
        assert s.file.read_bytes() == s.payload

    def test_existing_size_checked_even_when_validation_disabled(self, http_source):
        s = http_source
        assert _download(s, retries=1) == [s.file]
        s.file.write_bytes(b"truncated")
        assert _download(s, retries=1, validate=False) == [s.file]
        assert len(s.requests) == 2
        assert s.file.read_bytes() == s.payload

    def test_crash_before_data_rename_cannot_validate_old_same_length_file(
        self, http_source, monkeypatch,
    ):
        s = http_source
        old_payload = s.payload
        assert _download(s, retries=1) == [s.file]
        s.payload = old_payload[:-1] + bytes([old_payload[-1] ^ 1])
        s.etag = '"version-2"'
        original_replace = os.replace

        def interrupt_publication(source, destination):
            if source == s.part and destination == s.file:
                raise OSError("Simulated interruption before publishing the data file")
            return original_replace(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr("nisar_pytools.io.download.os.replace", interrupt_publication)
            assert _download(s, retries=1, reprocess=True) == []
        completed = json.loads((s.out / "product.h5.download.json").read_text())
        assert s.file.read_bytes() == old_payload
        assert s.part.read_bytes() == s.payload
        assert s.file.stat().st_size == s.part.stat().st_size
        assert completed["file_identity"]["inode"] == s.part.stat().st_ino
        assert completed["file_identity"]["inode"] != s.file.stat().st_ino

        # Recovery must check the partial against the server and publish it,
        # rather than returning the old, valid, equally sized HDF5 file.
        assert _download(s, retries=1) == [s.file]
        assert s.requests[-1]["Range"] == f"bytes={len(s.payload)}-"
        assert len(s.requests) == 3
        assert s.file.read_bytes() == s.payload

    def test_completed_file_identity_remains_valid_after_rename(self, http_source):
        s = http_source
        assert _download(s, retries=1) == [s.file]
        completed = json.loads((s.out / "product.h5.download.json").read_text())
        stat = s.file.stat()
        assert completed["file_identity"] == {
            "device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
        }
        assert _download(s, retries=1) == [s.file]
        assert len(s.requests) == 1

    def test_legacy_completion_metadata_without_identity_remains_compatible(self, http_source):
        s = http_source
        assert _download(s, retries=1) == [s.file]
        path = s.out / "product.h5.download.json"
        completed = json.loads(path.read_text())
        completed.pop("file_identity")
        path.write_text(json.dumps(completed))
        assert _download(s, retries=1) == [s.file]
        assert len(s.requests) == 1

    def test_failed_hdf5_validation_is_never_published(self, http_source):
        s = http_source
        s.payload = b"not HDF5"
        assert _download(s, retries=2) == []
        assert len(s.requests) == 2
        assert not s.file.exists()
        assert not s.part.exists()

    def test_reprocess_starts_fresh_but_retries_resume(self, http_source):
        s = http_source
        _interrupted(s)
        s.actions = [{}, {"drop": True}, {}]
        assert _download(s, retries=2, reprocess=True) == [s.file]
        assert "Range" not in s.requests[1]
        assert "Range" in s.requests[2]

    def test_source_mismatch_restarts(self, http_source):
        s = http_source
        _interrupted(s)
        metadata = json.loads(s.meta.read_text())
        metadata["source"] = "some-other-resource"
        s.meta.write_text(json.dumps(metadata))
        s.actions = [{}]
        assert _download(s, retries=1) == [s.file]
        assert "Range" not in s.requests[-1]

    def test_query_tokens_never_logged_or_persisted(self, http_source, caplog):
        s = http_source
        s.url += "?X-Amz-Signature=SECRET_SIGNATURE"
        _interrupted(s)
        assert "SECRET_SIGNATURE" not in s.meta.read_text()
        assert "SECRET_SIGNATURE" not in caplog.text
        assert "http://" not in s.meta.read_text()
        # Renewing a signed query does not invalidate safely validated bytes.
        s.url = s.url.replace("SECRET_SIGNATURE", "RENEWED_SIGNATURE")
        s.actions = [{}]
        assert _download(s, retries=1) == [s.file]
        assert "Range" in s.requests[-1]

    def test_rejected_http_status_never_logs_signed_url(self, http_source, caplog):
        s = http_source
        s.url += "?token=SECRET_TOKEN"
        s.actions = [{"status": 403}]
        assert _download(s, retries=1) == []
        assert "SECRET_TOKEN" not in caplog.text
        assert "HTTPError" in caplog.text

    @pytest.mark.parametrize("name", ["", ".", "..", "..\\outside.h5"])
    def test_unsafe_filename_rejected(self, http_source, name):
        s = http_source
        with pytest.raises(ValueError, match="filename"):
            download_urls([s.url.rsplit("/", 1)[0] + "/" + name], s.out)
