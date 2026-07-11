import json
import os
import sys
import time
import urllib.request

workflow_path = sys.argv[1]
run_id = sys.argv[2]
backend = os.environ.get("RUNFLOW_BACKEND_URL", "http://127.0.0.1:8001")

data = json.load(open(workflow_path))["data"]
data["run_id"] = run_id
body = json.dumps(data).encode()

req = urllib.request.Request(f"{backend}/graphs/runs", data=body, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req) as resp:
    print("submit:", resp.status, json.load(resp)["state"])

for _ in range(120):
    time.sleep(2)
    with urllib.request.urlopen(f"{backend}/runs/{run_id}") as resp:
        status = json.load(resp)
    state = status["state"]
    if state not in ("pending", "running", "queued", "starting"):
        print("final:", state, "error:", status.get("error"))
        break
    print("...", state)
else:
    print("timeout waiting")
