#!/usr/bin/env python3
"""Start a real, short-lived Uvicorn process with a temporary SQLite database.

The server is stopped the way an operator stops it (SIGINT, i.e. Ctrl+C), and
the smoke test requires a graceful shutdown: exit code 0 and no traceback.
"""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

# Runs the normal `uvicorn` command-line entry point. Uvicorn replaces the SIGINT and
# SIGTERM handlers while serving and re-installs the previous ones on shutdown. A
# handler installed natively before Python started — e.g. by an LD_PRELOAD job
# monitor such as XALT inside Slurm jobs — is reported by signal.signal() as None and
# cannot be re-installed, which made the shutdown end in a TypeError traceback.
# Only that unrepresentable SIGTERM handler is replaced by the default before Uvicorn
# starts; a normal SIG_DFL/SIG_IGN/Python handler is left untouched.
SERVER = """
import signal, sys
if signal.getsignal(signal.SIGTERM) is None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
from uvicorn.main import main
main(sys.argv[1:], prog_name="uvicorn")
"""


def main():
    with tempfile.TemporaryDirectory(prefix="hw1-startup-") as directory:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {**os.environ, "DATABASE_PATH": str(Path(directory) / "smoke.sqlite3"),
               "CLASSIFICATION_RUNS_DIR": str(Path(directory) / "runs")}
        imported = subprocess.run(
            [sys.executable, "-c", "from app.main import app; assert app.title == 'Neural 3D Data Viewer'"],
            cwd=ROOT / "backend", env=env, check=True,
        )
        assert imported.returncode == 0
        assert not Path(env["DATABASE_PATH"]).exists(), "Import must not create a database"
        log_path = Path(directory) / "server.log"
        log = log_path.open("w")
        process = subprocess.Popen(
            [sys.executable, "-c", SERVER, "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT / "backend", env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60  # PyTorch is imported at startup
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Backend exited before startup")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
                        assert json.load(response) == {"status": "ok", "database": "sqlite"}
                    with urlopen(f"http://127.0.0.1:{port}/api/scenes", timeout=1) as response:
                        assert json.load(response)[0]["id"] == "spectrum-garden"
                    assert Path(env["DATABASE_PATH"]).exists()
                    print("PASS: import without side effects, Uvicorn startup, HTTP health/scenes, SQLite creation")
                    classify(port)
                    break
                except URLError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Backend startup timed out")
        finally:
            code = stop(process)
            log.close()
            server_log = log_path.read_text()
            print(server_log, end="")  # the server's own log, after the checks above
        if code != 0 or "Traceback" in server_log:
            raise RuntimeError(f"Backend did not shut down cleanly: exit code {code}, see the log above")
        print("PASS: graceful shutdown on SIGINT, exit code 0, no traceback")


def stop(process: subprocess.Popen) -> int:
    """Stop Uvicorn like Ctrl+C: it finishes in-flight requests, runs shutdown and exits 0."""
    process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise RuntimeError("Backend did not stop within 20 s of SIGINT") from None


def classify(port: int) -> None:
    """Real inference over HTTP with the bundled checkpoints and samples."""
    index = json.loads((ROOT / "backend" / "samples" / "modelnet10" / "index.json").read_text())
    entry = index["samples"][0]
    for model in ("shallow", "mini_pointnet"):
        body = json.dumps({"sample_id": entry["id"], "model_id": model}).encode()
        request = Request(f"http://127.0.0.1:{port}/api/classification/predict/sample", data=body,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
        assert result["predicted_class"] == entry["predictions"][model], result
        assert len(result["top_k"]) == 3 and result["model"]["id"] == model
    data = (ROOT / "backend" / "samples" / "modelnet10" / entry["file"]).read_bytes()
    request = Request(f"http://127.0.0.1:{port}/api/classification/predict/upload?filename={entry['file']}",
                      data=data, headers={"Content-Type": "application/octet-stream"})
    with urlopen(request, timeout=30) as response:
        uploaded = json.load(response)
    assert uploaded["predicted_class"] == entry["predictions"]["mini_pointnet"], uploaded
    print(f"PASS: HTTP classification of {entry['id']} ({entry['class']}) by both bundled models; "
          f"upload path agrees ({uploaded['predicted_class']}, {uploaded['confidence']:.3f})")


if __name__ == "__main__":
    main()
