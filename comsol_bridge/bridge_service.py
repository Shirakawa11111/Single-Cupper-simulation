import os
import threading
import queue
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ---------------------------
# Configuration
# ---------------------------
SERVER_HOST = os.getenv("COMSOL_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("COMSOL_SERVER_PORT", "2036"))
SERVER_USER = os.getenv("COMSOL_SERVER_USER")
SERVER_PASSWORD = os.getenv("COMSOL_SERVER_PASSWORD")

COMSOL_VERSION = os.getenv("COMSOL_VERSION", "") or None
MODEL_ROOT = os.getenv("COMSOL_MODEL_ROOT", r"D:\Comsol_model")
OUTPUT_ROOT = os.getenv("COMSOL_OUTPUT_ROOT", r"D:\Comsol_model\_out")
EXPORT_DIR_PARAM = os.getenv("COMSOL_EXPORT_DIR_PARAM", "export_dir")

BRIDGE_HOST = os.getenv("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8000"))


# ---------------------------
# Data models
# ---------------------------
class ExportItem(BaseModel):
    tag: str = Field(..., description="Export node tag, e.g. 'data1' or 'img1'")


class JobRequest(BaseModel):
    model_path: str = Field(..., description="Absolute path or relative to COMSOL_MODEL_ROOT")
    params: Dict[str, str] = Field(default_factory=dict, description="COMSOL parameters")
    study: Optional[str] = Field(None, description="Study tag, e.g. 'std1'")
    exports: List[ExportItem] = Field(default_factory=list, description="Export node tags")
    export_dir_param: Optional[str] = Field(None, description="Override export dir parameter name")


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None
    files: List[str] = Field(default_factory=list)


# ---------------------------
# Helper functions
# ---------------------------
def _ensure_env():
    if not SERVER_USER or not SERVER_PASSWORD:
        raise RuntimeError("COMSOL_SERVER_USER/COMSOL_SERVER_PASSWORD not set")


def _abs_model_path(path: str) -> str:
    if os.path.isabs(path):
        model_path = path
    else:
        model_path = os.path.join(MODEL_ROOT, path)

    model_path = os.path.normpath(model_path)
    model_root = os.path.normpath(MODEL_ROOT)
    if os.path.commonpath([model_root, model_path]) != model_root:
        raise ValueError("model_path must be within COMSOL_MODEL_ROOT")
    return model_path


def _make_output_dir(job_id: str) -> str:
    out_dir = os.path.join(OUTPUT_ROOT, job_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _list_files(base_dir: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(base_dir):
        for name in filenames:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, base_dir)
            files.append(rel.replace("\\", "/"))
    return sorted(files)


def _safe_join(base_dir: str, rel_path: str) -> str:
    rel_path = rel_path.replace("/", os.sep)
    target = os.path.normpath(os.path.join(base_dir, rel_path))
    base_dir = os.path.normpath(base_dir)
    if os.path.commonpath([base_dir, target]) != base_dir:
        raise ValueError("invalid path")
    return target


# ---------------------------
# COMSOL runner
# ---------------------------
class ComsolRunner:
    def __init__(self):
        self.client = None

    def _connect(self):
        _ensure_env()
        import mph  # pylint: disable=import-error

        if COMSOL_VERSION:
            client = mph.Client(host=None, version=COMSOL_VERSION)
        else:
            client = mph.Client(host=None)

        # Use Java API directly to pass username/password.
        client.java.connect(SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASSWORD)
        client.host = SERVER_HOST
        client.port = SERVER_PORT
        self.client = client

    def run_job(self, job_id: str, req: JobRequest) -> str:
        if self.client is None:
            self._connect()

        model_path = _abs_model_path(req.model_path)
        out_dir = _make_output_dir(job_id)

        # COMSOL accepts forward slashes on Windows; keep a trailing slash.
        export_dir_value = out_dir.replace("\\", "/") + "/"
        export_dir_param = req.export_dir_param or EXPORT_DIR_PARAM

        model = self.client.load(model_path)
        try:
            if export_dir_param:
                model.parameter(export_dir_param, export_dir_value)

            for name, value in req.params.items():
                model.parameter(name, value)

            if req.study:
                model.java.study(req.study).run()
            else:
                model.solve()

            for item in req.exports:
                model.java.result().export(item.tag).run()
        finally:
            self.client.remove(model)

        return out_dir


# ---------------------------
# Job management
# ---------------------------
app = FastAPI(title="COMSOL Bridge Service")
_jobs: Dict[str, JobStatus] = {}
_jobs_lock = threading.Lock()
_job_queue: "queue.Queue[tuple[str, JobRequest]]" = queue.Queue()


def _set_status(job_id: str, **kwargs):
    with _jobs_lock:
        status = _jobs[job_id]
        for k, v in kwargs.items():
            setattr(status, k, v)


def _worker_loop():
    runner = ComsolRunner()
    while True:
        job_id, req = _job_queue.get()
        if job_id is None:
            break

        _set_status(job_id, status="running", started_at=time.time())
        try:
            out_dir = runner.run_job(job_id, req)
            files = _list_files(out_dir)
            _set_status(
                job_id,
                status="done",
                finished_at=time.time(),
                output_dir=out_dir,
                files=files,
            )
        except Exception as exc:  # pragma: no cover - surface error to client
            _set_status(
                job_id,
                status="error",
                finished_at=time.time(),
                error=str(exc),
            )


_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()


# ---------------------------
# API endpoints
# ---------------------------
@app.get("/health")
def health():
    return {
        "server_host": SERVER_HOST,
        "server_port": SERVER_PORT,
        "model_root": MODEL_ROOT,
        "output_root": OUTPUT_ROOT,
        "version": COMSOL_VERSION,
        "queue_size": _job_queue.qsize(),
    }


@app.post("/jobs", response_model=JobStatus)
def submit_job(req: JobRequest):
    job_id = uuid.uuid4().hex
    status = JobStatus(job_id=job_id, status="queued", created_at=time.time())
    with _jobs_lock:
        _jobs[job_id] = status

    _job_queue.put((job_id, req))
    return status


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="job not found")
        return _jobs[job_id]


@app.get("/jobs/{job_id}/files")
def list_job_files(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="job not found")
        status = _jobs[job_id]

    if status.status != "done":
        raise HTTPException(status_code=400, detail="job not finished")

    return {"files": status.files}


@app.get("/jobs/{job_id}/files/{path:path}")
def download_file(job_id: str, path: str):
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="job not found")
        status = _jobs[job_id]

    if status.status != "done" or not status.output_dir:
        raise HTTPException(status_code=400, detail="job not finished")

    try:
        full_path = _safe_join(status.output_dir, path)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid path")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(full_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bridge_service:app", host=BRIDGE_HOST, port=BRIDGE_PORT, log_level="info")
