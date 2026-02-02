import time
import requests

BRIDGE_HOST = "192.168.196.81"  # Windows ZeroTier or LAN IP
BRIDGE_PORT = 8000
BASE = f"http://{BRIDGE_HOST}:{BRIDGE_PORT}"

payload = {
    "model_path": "demo.mph",  # relative to COMSOL_MODEL_ROOT
    "params": {
        "L": "10[mm]",
        "E": "110[GPa]",
    },
    "study": "std1",
    "exports": [
        {"tag": "data1"},
        {"tag": "img1"},
    ],
}

r = requests.post(f"{BASE}/jobs", json=payload, timeout=10)
r.raise_for_status()
job = r.json()
job_id = job["job_id"]

while True:
    s = requests.get(f"{BASE}/jobs/{job_id}", timeout=10).json()
    print("status:", s["status"])
    if s["status"] in ("done", "error"):
        print(s)
        break
    time.sleep(2)

if s["status"] == "done":
    files = requests.get(f"{BASE}/jobs/{job_id}/files", timeout=10).json()["files"]
    print("files:", files)
