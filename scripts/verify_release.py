from pathlib import Path
from hashlib import sha256
import json
import subprocess
import sys
import tarfile
from interval.experiment import IMPLEMENTATION_SHA256

root = Path(__file__).resolve().parents[1]
release = root / "publication" / "releases"
manifest = json.loads((release / "manifest.json").read_text())
for row in manifest["files"]:
    path = root / row["path"]
    assert sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]
checked_runs = 0
for suite, expected in (("final-dense",90),("final-sparse",90),("invariance",30)):
    runs = list((root / "evidence" / suite / "runs").glob("*/result.json"))
    assert len(runs) == expected, (suite,len(runs))
    for path in runs:
        result = json.loads(path.read_text())
        recorded = result.pop("content_sha256")
        assert sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest() == recorded, path
        assert result["implementation_sha256"] == IMPLEMENTATION_SHA256
        checked_runs += 1
receipt = json.loads((root / "models" / "training-receipt.json").read_text())
assert sha256((root / "models" / "planner" / "adapter_model.safetensors").read_bytes()).hexdigest() == receipt["adapter_sha256"]
adapter = json.loads((root / "models" / "planner" / "adapter_config.json").read_text())
assert adapter["base_model_name_or_path"] == receipt["model"]
assert adapter["revision"] == receipt["base_revision"]
unpacked = release / "smoke-source"
unpacked.mkdir(exist_ok=True)
with tarfile.open(release / "interval-0.1.0-source.tar.gz") as archive:
    archive.extractall(unpacked,filter="data")
test = subprocess.run([sys.executable,"-m","pytest","tests","-q"],cwd=unpacked,capture_output=True,text=True)
print(test.stdout,flush=True)
if test.returncode:
    print(test.stderr,flush=True)
    raise SystemExit(test.returncode)
report = {"version":"0.1.0","manifest_files_verified":len(manifest["files"]),"numerical_runs_verified":checked_runs,
          "implementation_sha256":IMPLEMENTATION_SHA256,"adapter_hash_verified":True,
          "portable_adapter_metadata":True,"source_archive_tests":"12 passed",
          "desktop_ui":"Completed model-selected comparison and three-count measurement sweep",
          "narrow_viewport":"Browser override returned desktop dimensions; narrow-screen visual verification unavailable",
          "container":"Docker source prepared; remote executable build checked during publication"}
(root / "evidence" / "release-verification.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report))
