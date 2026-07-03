from __future__ import annotations
import json, hashlib
from pathlib import Path
from pydantic import BaseModel

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj.model_dump(mode='json') if isinstance(obj, BaseModel) else obj
    path.write_text(json.dumps(data, indent=2, sort_keys=True)+"\n")

def read_json(path: Path): return json.loads(path.read_text())
def append_jsonl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj.model_dump(mode='json') if isinstance(obj, BaseModel) else obj
    with path.open('a') as f: f.write(json.dumps(data, sort_keys=True)+"\n")
def read_jsonl(path: Path):
    if not path.exists(): return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def sha256_file(path: Path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
