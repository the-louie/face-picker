"""One-off (2026-11-09): turn 0.1.0's per-visit folders into 0.2.0's flat files.

incoming/<stamp>[_k]/{1.jpg | p1_1.jpg, *_context.jpg, meta.json}
  -> incoming/<stamp>_<n>.jpg, n in person-then-rank order; the meta goes into the faces table.
"""
import json
import shutil
import sys

sys.path.insert(0, "/app")
import facepicker as fp  # noqa: E402

cfg = fp.Config()
state = fp.State(cfg.state_dir / "state.sqlite")
moved = folders = 0
for d in sorted(p for p in cfg.incoming.iterdir() if p.is_dir()):
    if d.name.startswith(".tmp-"):
        shutil.rmtree(d)
        continue
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"faces": []}
    faces = sorted(meta.pop("faces", []), key=lambda f: (f.get("person", 1), f.get("rank", 1)))
    st = d.name[:19]
    n = fp.next_index(cfg.incoming, st)
    records = []
    for f in faces:
        src = d / f["file"]
        if not src.exists():
            continue
        f["file"] = f"{st}_{n}.jpg"
        src.rename(cfg.incoming / f["file"])
        records.append(f)
        n += 1
        moved += 1
    state.save_faces(meta.get("review_id", ""), meta, records)
    shutil.rmtree(d)
    folders += 1
print(f"{folders} folders -> {moved} images")
