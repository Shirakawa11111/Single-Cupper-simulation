# COMSOL Bridge Service (Windows)

This service runs on Windows and connects to a local COMSOL Server. It exposes a small HTTP API so your local Python can submit jobs and download results.

## 1) Windows setup

1. Start COMSOL Server on the Windows machine (listening on localhost:2036).
2. Ensure the COMSOL Server account works by logging in with COMSOL Client on the same machine.
3. Create a Python venv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Configure environment variables

Set these on Windows (PowerShell or cmd). Replace placeholders with your own values:

```bat
setx COMSOL_SERVER_HOST 127.0.0.1
setx COMSOL_SERVER_PORT 2036
setx COMSOL_SERVER_USER your_user
setx COMSOL_SERVER_PASSWORD your_password
setx COMSOL_MODEL_ROOT D:\Comsol_model
setx COMSOL_OUTPUT_ROOT D:\Comsol_model\_out
setx COMSOL_EXPORT_DIR_PARAM export_dir
setx COMSOL_VERSION 5.6
setx BRIDGE_HOST 0.0.0.0
setx BRIDGE_PORT 8000
```

Notes:
- `COMSOL_EXPORT_DIR_PARAM` is optional. If set, the service will set that COMSOL parameter to the job output directory.
- In your COMSOL model, set export file names using this parameter, e.g. `export_dir + "table1.csv"`.

## 3) Prepare your model exports

In COMSOL, pre-create export nodes for what you want:
- Tables
- Curves (plot data)
- Fields (data or images)

Give each export node a tag (e.g., `data1`, `img1`). The service will call these tags.

## 4) Run the service

```bash
python bridge_service.py
```

The service listens on `0.0.0.0:8000` by default. Allow inbound access on this port for your ZeroTier interface.

## 5) Call from your local machine

See `client_example.py` for a minimal call.

## API Summary

- `GET /health`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/files`
- `GET /jobs/{job_id}/files/{path}`

## Job request example

```json
{
  "model_path": "beam.mph",
  "params": {"L": "10[mm]", "E": "110[GPa]"},
  "study": "std1",
  "exports": [{"tag": "data1"}, {"tag": "img1"}]
}
```
