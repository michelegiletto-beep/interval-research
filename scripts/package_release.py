"""Create a reviewed source release, executable Space bundle and file hashes."""
from pathlib import Path
from hashlib import sha256
import json
import tarfile
import zipfile

root = Path(__file__).resolve().parents[1]
release = root / "publication" / "releases"
space = root / "publication" / "space"
release.mkdir(parents=True, exist_ok=True)
space.mkdir(parents=True, exist_ok=True)
roots = ["interval", "web", "docs", "tests", "configs", "profiles", "scripts", "datasets/planner", ".github"]
files = [root / name for name in ("README.md","LICENSE","NOTICE","CITATION.cff","pyproject.toml","Dockerfile","requirements-docker.txt",".gitignore",".dockerignore")]
for directory in roots:
    files += [p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc"}]
for directory in ("evidence/final-dense","evidence/final-sparse","evidence/invariance","evidence/figures","evidence/showcase"):
    files += [p for p in (root / directory).rglob("*") if p.is_file()]
for name in ("evidence/report-summary.json","evidence/environment-lock.json","models/base-model.json","models/training-receipt.json",
             "models/planner/adapter_config.json","models/planner/adapter_model.safetensors"):
    files.append(root / name)
files += [p for p in (root / "publication").glob("*.md") if p.is_file()]
files.append(root / "publication" / "hf-app.py")
files = sorted(set(files))
missing = [str(p) for p in files if not p.exists()]
if missing:
    raise RuntimeError(f"Missing release inputs: {missing}")
manifest = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p.read_bytes()).hexdigest()} for p in files]
assert any(r["path"] == "models/planner/adapter_model.safetensors" for r in manifest)
assert not any("models/base/" in r["path"] or "runtime/" in r["path"] or ".env" in r["path"] for r in manifest)
manifest_path = release / "manifest.json"
manifest_path.write_text(json.dumps({"version":"0.1.0","files":manifest},indent=2))
archive = release / "interval-0.1.0-source.tar.gz"
with tarfile.open(archive,"w:gz") as handle:
    for path in files:
        handle.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
    handle.add(manifest_path,arcname="release-manifest.json",recursive=False)
zip_path = release / "interval-0.1.0-source.zip"
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as handle:
    for path in files:
        handle.write(path,path.relative_to(root).as_posix())
    handle.write(manifest_path,"release-manifest.json")
import shutil
shutil.copy2(archive,space / "interval-source.tar.gz")
shutil.copy2(root / "requirements-docker.txt",space / "requirements-docker.txt")
shutil.copy2(root / "publication" / "hf-app.py",space / "app.py")
space_requirements = (root / "requirements-docker.txt").read_text().splitlines()
space_requirements = [line for line in space_requirements if not line.startswith("huggingface-hub")]
(space / "requirements.txt").write_text("\n".join(space_requirements) + "\n")
docker=(root / "Dockerfile").read_text()
docker=docker.replace("COPY . /app", "COPY interval-source.tar.gz /tmp/interval-source.tar.gz\nRUN tar -xzf /tmp/interval-source.tar.gz -C /app && rm /tmp/interval-source.tar.gz")
(space / "Dockerfile").write_text(docker)
(space / "README.md").write_text('''---
title: Interval
emoji: 🔬
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: 3.12.12
license: apache-2.0
short_description: Local neural planner and reproducible change detection
---

# Interval

Executable synthetic demonstration of the open-source research workspace.
An embedded, locally adapted SmolLM2-135M model selects bounded research
procedures; native Python trains and evaluates the numerical models.

The public demonstration accepts synthetic observations only. Own-data work,
multi-run sweeps and local adaptation are available in the source application.
This is a single-user research reference, not a clinically or field-validated
institutional service. No quantum advantage is established.

- [Source and installation](https://github.com/michelegiletto-beep/interval-research)
- [System Labs](https://giletto-systems-lab.michele-giletto.chatgpt.site)
- [Manuscript draft](https://github.com/michelegiletto-beep/interval-research/blob/main/docs/manuscript.md)
- [Complete source, adapter and evidence archive](https://huggingface.co/spaces/Magic51/interval-research/resolve/main/interval-source.tar.gz)
- [Upstream model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

The base weights are pinned and downloaded during the first startup. All
subsequent model inference executes inside the application runtime. A Gradio
status panel is mounted alongside the FastAPI research interface.
''',encoding="utf-8")
receipt = {"files":len(files),"source_archive":str(archive),"source_zip":str(zip_path),
           "archive_sha256":sha256(archive.read_bytes()).hexdigest(),"bytes":archive.stat().st_size,
           "adapter_included":True,"base_weights":"pinned upstream download during installation/image build"}
(release / "package-receipt.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
