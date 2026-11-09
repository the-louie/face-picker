#!/usr/bin/env python3
"""Pick the best face images from Frigate's doorbell recordings, for face training.

Every round (default 5 min):
  PICK   new doorbell visits (Frigate review items with a person) -> the visit's clip ->
         1 frame/s -> YuNet faces -> quality gates -> eDifFIQA ranking -> top 3 per person
         -> incoming/<YYYY-MM-DD_HH-MM-SS>_<n>.jpg (n = rank; details in the state database)
  TRAIN  pending/<name>/<image> -> Frigate POST /api/faces/<name>/register
         -> trained/<name>/ (ok) or pending/<name>/failed/ (+ .reason.txt)

It only reads Frigate's media (through Frigate's API) and never deletes anything there.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import requests

VERSION = "0.2.0"
LOG = logging.getLogger("facepicker")


# ---------------------------------------------------------------------------- config

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    frigate_url: str = _env("FRIGATE_URL", "http://127.0.0.1:5000")
    camera: str = _env("CAMERA", "doorbell_cam")
    zone: str = _env("ZONE", "doorbell_zone")
    data: Path = Path(_env("DATA_DIR", "/data"))
    models: Path = Path(_env("MODEL_DIR", "/app/models"))
    tz: ZoneInfo = ZoneInfo(_env("TZ", "Europe/Stockholm"))  # noqa: RUF009 - read once at start
    interval_s: int = int(_env("INTERVAL_S", "300"))
    backfill_days: float = float(_env("BACKFILL_DAYS", "30"))
    settle_s: int = int(_env("SETTLE_S", "120"))         # a visit must have ended this long ago
    max_clip_s: int = int(_env("MAX_CLIP_S", "300"))
    max_items_per_round: int = int(_env("MAX_ITEMS_PER_ROUND", "40"))
    max_attempts: int = int(_env("MAX_ATTEMPTS", "5"))
    pause_hours_mod: int = 6                              # Frigate restarts at :13 every 6 h
    pause_minutes: tuple[int, int] = (10, 18)
    vaapi_device: str = _env("VAAPI_DEVICE", "/dev/dri/renderD128")
    analysis_height: int = 1080                           # Frigate's MAX_DETECTION_HEIGHT
    # gates (section 5.3 of the plan); full-resolution pixels
    min_det_score: float = float(_env("MIN_DET_SCORE", "0.75"))
    min_face_h: int = int(_env("MIN_FACE_H", "120"))
    min_eye_dist: int = int(_env("MIN_EYE_DIST", "30"))
    max_yaw: float = float(_env("MAX_YAW", "30"))
    max_roll: float = float(_env("MAX_ROLL", "20"))
    edge_margin: float = 0.02
    min_sharpness: float = float(_env("MIN_SHARPNESS", "25"))
    lum_range: tuple[int, int] = (30, 225)
    max_clipped: float = 0.10
    # selection
    per_person: int = 3
    max_persons: int = 3
    min_spacing_s: float = 1.0
    track_max_dx: float = float(_env("TRACK_MAX_DX", "0.15"))  # face-to-person distance, fraction of width
    track_max_dt: float = 3.0                             # path point must be this close in time
    crop_margin: float = 0.4
    crop_max_side: int = 1024
    train_settle_s: int = 30

    @property
    def incoming(self) -> Path:
        return self.data / "incoming"

    @property
    def pending(self) -> Path:
        return self.data / "pending"

    @property
    def trained(self) -> Path:
        return self.data / "trained"

    @property
    def discarded(self) -> Path:
        return self.data / "discarded"

    @property
    def state_dir(self) -> Path:
        return self.data / ".state"


# ---------------------------------------------------------------------------- Frigate

class Frigate:
    def __init__(self, url: str, timeout: float = 20) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()

    def _get(self, path: str, **params):
        r = self.s.get(f"{self.url}{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def version(self) -> str:
        r = self.s.get(f"{self.url}/api/version", timeout=self.timeout)
        r.raise_for_status()
        return r.text.strip()

    def reviews(self, camera: str, after: float, before: float) -> list[dict]:
        items = self._get("/api/review", cameras=camera, after=after, before=before, limit=10000)
        return sorted(items, key=lambda r: r["start_time"])

    def event(self, event_id: str) -> dict | None:
        r = self.s.get(f"{self.url}/api/events/{event_id}", timeout=self.timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def clip(self, camera: str, start: float, end: float, dest: Path) -> None:
        url = f"{self.url}/api/{camera}/start/{start:.3f}/end/{end:.3f}/clip.mp4"
        with self.s.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.writelines(r.iter_content(1 << 20))
        if dest.stat().st_size < 10_000:
            raise OSError(f"clip too small ({dest.stat().st_size} bytes)")

    def faces(self) -> dict:
        return self._get("/api/faces")

    def create_face(self, name: str) -> dict:
        r = self.s.post(f"{self.url}/api/faces/{name}/create", timeout=self.timeout)
        return r.json()

    def register(self, name: str, image: Path) -> dict:
        with open(image, "rb") as f:
            r = self.s.post(f"{self.url}/api/faces/{name}/register",
                            files={"file": (image.name, f, "application/octet-stream")},
                            timeout=60)
        try:
            return r.json()
        except ValueError:
            return {"success": False, "message": f"HTTP {r.status_code}: {r.text[:200]}"}


# ---------------------------------------------------------------------------- state

class State:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY, start_time REAL, status TEXT, attempts INTEGER DEFAULT 0,
            folder TEXT, faces INTEGER, persons INTEGER, seconds REAL, error TEXT,
            review TEXT, updated REAL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS faces (
            file TEXT PRIMARY KEY, review_id TEXT, record TEXT, visit TEXT, created REAL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS trained (
            file TEXT, name TEXT, time REAL, success INTEGER, message TEXT)""")
        self.db.commit()

    def status(self, item_id: str) -> tuple[str, int] | None:
        row = self.db.execute("SELECT status, attempts FROM items WHERE id=?", (item_id,)).fetchone()
        return (row[0], row[1]) if row else None

    def save(self, item_id: str, start: float, status: str, review: dict, **kw) -> None:
        old = self.status(item_id)
        attempts = (old[1] if old else 0) + (1 if status == "retry" else 0)
        self.db.execute("""INSERT OR REPLACE INTO items
            (id, start_time, status, attempts, folder, faces, persons, seconds, error, review, updated)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (item_id, start, status, attempts, kw.get("folder"), kw.get("faces"),
                         kw.get("persons"), kw.get("seconds"), kw.get("error"),
                         json.dumps(review), time.time()))
        self.db.commit()

    def forget(self, item_id: str) -> None:
        self.db.execute("DELETE FROM items WHERE id=?", (item_id,))
        self.db.commit()

    def retries(self, max_attempts: int) -> list[dict]:
        rows = self.db.execute("SELECT review FROM items WHERE status='retry' AND attempts<?",
                               (max_attempts,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def watermark(self) -> float | None:
        row = self.db.execute("SELECT MAX(start_time) FROM items").fetchone()
        return row[0]

    def counts(self) -> dict:
        return dict(self.db.execute("SELECT status, COUNT(*) FROM items GROUP BY status").fetchall())

    def save_faces(self, review_id: str, visit: dict, records: list[dict]) -> None:
        for r in records:
            self.db.execute("INSERT OR REPLACE INTO faces VALUES (?,?,?,?,?)",
                            (r["file"], review_id, json.dumps(r), json.dumps(visit), time.time()))
        self.db.commit()

    def log_trained(self, file: str, name: str, success: bool, message: str) -> None:
        self.db.execute("INSERT INTO trained VALUES (?,?,?,?,?)",
                        (file, name, time.time(), int(success), message))
        self.db.commit()


# ---------------------------------------------------------------------------- video

def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def analysis_size(w: int, h: int, target_h: int) -> tuple[int, int]:
    if h <= target_h:
        return w, h
    return int(round(w * target_h / h / 2) * 2), target_h


def _ffmpeg_frames(cmd: list[str], w: int, h: int):
    size = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        rc = proc.wait()
        if rc not in (0, -9) and err:
            LOG.debug("ffmpeg rc=%s: %s", rc, err[-300:])


def sample_frames(path: Path, cfg: Config, use_vaapi: bool):
    """Yield (t_seconds, full-resolution BGR frame), 1 per second. VAAPI with CPU fallback.

    Full resolution on purpose: the crops must come from the very frame that was scored.
    Re-reading a frame by seeking later gave a different moment (tested 2026-11-09).
    """
    w, h = probe_size(path)
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    out = ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:"]
    cpu = base + ["-i", str(path), "-vf", "fps=1"] + out
    if use_vaapi and Path(cfg.vaapi_device).exists():
        hw = base + ["-hwaccel", "vaapi", "-hwaccel_device", cfg.vaapi_device,
                     "-hwaccel_output_format", "vaapi", "-i", str(path), "-vf",
                     "fps=1,hwdownload,format=nv12"] + out
        got = False
        for i, fr in enumerate(_ffmpeg_frames(hw, w, h)):
            got = True
            yield float(i), fr
        if got:
            return
        LOG.warning("VAAPI decode gave no frames for %s; using the CPU", path.name)
    for i, fr in enumerate(_ffmpeg_frames(cpu, w, h)):
        yield float(i), fr


# ---------------------------------------------------------------------------- faces

@dataclass
class Face:
    t: float
    row: np.ndarray            # YuNet row (15 values) in analysis-frame coordinates
    scale: float               # analysis -> full resolution
    det: float
    quality: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    sharpness: float = 0.0
    lum: float = 0.0
    clipped: float = 0.0
    emb: np.ndarray | None = None
    score: float = 0.0
    reject: list[str] = field(default_factory=list)
    person: int = 0
    xn: float = 0.0            # face centre x as a fraction of the frame width
    event: str | None = None   # the Frigate person event this face belongs to

    @property
    def box_full(self) -> list[int]:
        x, y, w, h = self.row[:4]
        return [round(v * self.scale) for v in (x, y, w, h)]

    @property
    def eye_dist_full(self) -> float:
        (rx, ry), (lx, ly) = self.row[4:6], self.row[6:8]
        return math.hypot(lx - rx, ly - ry) * self.scale


def pose(row: np.ndarray) -> tuple[float, float]:
    """(yaw, roll) in degrees, estimated from YuNet's 5 landmarks."""
    rx, ry, lx, ly, nx, _ny = row[4:10]
    eye_dist = math.hypot(lx - rx, ly - ry) or 1.0
    roll = math.degrees(math.atan2(ly - ry, lx - rx))
    mid_x = (rx + lx) / 2
    # nose offset from the eye midpoint, as a fraction of the eye distance; ~0.5 at 45 degrees
    yaw = max(-90.0, min(90.0, float(nx - mid_x) / float(eye_dist) * 90.0))
    return float(yaw), float(roll)


def rank_score(quality: float, face_h_full: float, yaw: float) -> float:
    size_factor = min(1.0, face_h_full / 300.0) ** 0.5
    frontal = max(0.0, 1.0 - abs(yaw) / 60.0)
    return quality * size_factor * frontal


class Analyzer:
    def __init__(self, cfg: Config) -> None:
        m = cfg.models
        self.cfg = cfg
        self.det = cv2.FaceDetectorYN.create(str(m / "yunet.onnx"), "", (320, 320), 0.6, 0.3, 50)
        self.rec = cv2.FaceRecognizerSF.create(str(m / "sface.onnx"), "")
        self.fiqa = cv2.dnn.readNet(str(m / "ediffiqa.onnx"))

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        self.det.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.det.detect(frame)
        return [] if faces is None else list(faces)

    def evaluate(self, frame: np.ndarray, t: float, scale: float) -> list[Face]:
        cfg = self.cfg
        out = []
        H, W = frame.shape[:2]
        for row in self.detect(frame):
            f = Face(t=t, row=row, scale=scale, det=float(row[14]))
            x, y, w, h = row[:4]
            f.xn = float((x + w / 2) / W)
            f.yaw, f.roll = pose(row)
            if f.det < cfg.min_det_score:
                f.reject.append("det")
            if h * scale < cfg.min_face_h:
                f.reject.append("small")
            m = cfg.edge_margin
            if x < m * W or y < m * H or x + w > (1 - m) * W or y + h > (1 - m) * H:
                f.reject.append("edge")
            if abs(f.yaw) > cfg.max_yaw:
                f.reject.append("yaw")
            if abs(f.roll) > cfg.max_roll:
                f.reject.append("roll")
            if f.eye_dist_full < cfg.min_eye_dist:
                f.reject.append("eyes")
            if f.reject:            # cheap gates first; skip the networks
                out.append(f)
                continue
            aligned = self.rec.alignCrop(frame, row)
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            f.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            f.lum = float(gray.mean())
            f.clipped = float(((gray <= 3) | (gray >= 252)).mean())
            if f.sharpness < cfg.min_sharpness:
                f.reject.append("blur")
            if not cfg.lum_range[0] <= f.lum <= cfg.lum_range[1] or f.clipped > cfg.max_clipped:
                f.reject.append("exposure")
            blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5),
                                         swapRB=True)
            self.fiqa.setInput(blob)
            f.quality = float(self.fiqa.forward().ravel()[0])
            f.emb = self.rec.feature(aligned).ravel().copy()
            f.score = float(rank_score(f.quality, float(h * scale), f.yaw))
            out.append(f)
        return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


def track_x(event: dict, wall: float, max_dt: float) -> float | None:
    """The person's x (fraction of width) at wall-clock time, from Frigate's path_data.

    path_data holds the box's bottom-centre point, recorded each time the person moves 5 %
    of the frame; x is the same in the detect crop and the full frame (full-width crop).
    """
    pts = [(float(p[0][0]), float(p[1])) for p in (event.get("data") or {}).get("path_data") or []]
    box = (event.get("data") or {}).get("box")
    if box and event.get("data", {}).get("snapshot_frame_time"):
        pts.append((float(box[0]) + float(box[2]) / 2, float(event["data"]["snapshot_frame_time"])))
    pts = [p for p in pts if abs(p[1] - wall) <= max_dt]
    if not pts:
        return None
    pts.sort(key=lambda p: p[1])
    before = [p for p in pts if p[1] <= wall]
    after = [p for p in pts if p[1] >= wall]
    if before and after and after[0][1] > before[-1][1]:
        (x0, t0), (x1, t1) = before[-1], after[0]
        return x0 + (x1 - x0) * (wall - t0) / (t1 - t0)
    return min(pts, key=lambda p: abs(p[1] - wall))[0]


def assign_to_tracks(faces: list[Face], events: list[dict], clip_start: float,
                     cfg: Config) -> bool:
    """Attach each passing face to the Frigate person track closest in x at that moment.

    Faces with a track nearby but none within track_max_dx are rejected ("no_person": a
    wheel or a poster). Returns False if no track had data, so the caller falls back.
    """
    people = [e for e in events if e.get("label") == "person"]
    if not people:
        return False
    any_data = False
    for f in faces:
        if f.reject:
            continue
        wall = clip_start + f.t
        best, best_dx = None, None
        for e in people:
            x = track_x(e, wall, cfg.track_max_dt)
            if x is None:
                continue
            any_data = True
            dx = abs(x - f.xn)
            if best_dx is None or dx < best_dx:
                best, best_dx = e, dx
        if best is None:
            if len(people) == 1:     # one person in the visit: no doubt whose face it is
                f.event = people[0]["id"]
            continue
        if best_dx > cfg.track_max_dx:
            f.reject.append("no_person")
        else:
            f.event = best["id"]
    return any_data


def select(faces: list[Face], cfg: Config) -> list[list[Face]]:
    """Group passing faces into persons, then pick the top N per person spread in time.

    The number of persons is the most passing faces seen in one frame: two faces in the
    same frame are two people for certain, whereas embedding distance alone split one
    person into several when distance and light changed (seen in testing 2026-11-09).
    The faces of that busiest frame seed the persons; every other face joins the seed it
    resembles most (SFace cosine).
    """
    ok = sorted((f for f in faces if not f.reject), key=lambda f: f.score, reverse=True)
    if not ok:
        return []
    if any(f.event for f in ok):
        groups: dict[str, list[Face]] = {}
        for f in ok:   # faces without a track nearby in time go with the best-scoring track
            groups.setdefault(f.event or "", []).append(f)
        if "" in groups and len(groups) > 1:
            del groups[""]          # unassignable when 2+ people: safer to drop than to mix
        return _pick(sorted(groups.values(), key=lambda p: p[0].score, reverse=True), cfg)
    by_t: dict[float, list[Face]] = {}
    for f in ok:
        by_t.setdefault(f.t, []).append(f)
    seed_t = max(by_t, key=lambda t: (len(by_t[t]), max(g.score for g in by_t[t])))
    seeds = sorted(by_t[seed_t], key=lambda f: f.score, reverse=True)[:cfg.max_persons]
    persons: list[list[Face]] = [[] for _ in seeds]
    for f in ok:
        if len(seeds) == 1 or f.emb is None:
            k = 0
        else:
            k = max(range(len(seeds)),
                    key=lambda i: cosine(f.emb, seeds[i].emb) if seeds[i].emb is not None else -1)
        persons[k].append(f)
    persons.sort(key=lambda p: p[0].score if p else 0, reverse=True)
    return _pick(persons, cfg)


def _pick(persons: list[list[Face]], cfg: Config) -> list[list[Face]]:
    chosen = []
    for idx, p in enumerate(persons[:cfg.max_persons], start=1):
        pick: list[Face] = []
        for f in p:                       # already in score order
            if all(abs(f.t - g.t) >= cfg.min_spacing_s for g in pick):
                f.person = idx
                pick.append(f)
            if len(pick) == cfg.per_person:
                break
        chosen.append(pick)
    return chosen


# ---------------------------------------------------------------------------- output

def crop_face(full: np.ndarray, box: list[int], margin: float, max_side: int) -> np.ndarray:
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    H, W = full.shape[:2]
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    crop = full[y0:y1, x0:x1]
    s = max(crop.shape[:2])
    if s > max_side:
        k = max_side / s
        crop = cv2.resize(crop, (int(crop.shape[1] * k), int(crop.shape[0] * k)),
                          interpolation=cv2.INTER_AREA)
    return crop


def stamp(start: float, tz: ZoneInfo) -> str:
    return dt.datetime.fromtimestamp(start, tz).strftime("%Y-%m-%d_%H-%M-%S")


def next_index(incoming: Path, st: str) -> int:
    """The first free n for <stamp>_<n>.jpg (two visits can start in the same second)."""
    used = [int(m.group(1)) for p in incoming.glob(f"{st}_*.jpg")
            if (m := re.fullmatch(re.escape(st) + r"_(\d+)\.jpg", p.name))]
    return max(used, default=0) + 1


def write_output(cfg: Config, frames: dict[float, np.ndarray], start: float,
                 chosen: list[list[Face]], dry_run: bool = False) -> tuple[str | None, list[dict]]:
    """Write incoming/<YYYY-MM-DD_HH-MM-SS>_<n>.jpg, n = rank (person 1 first, then person 2).

    The stamp is the visit's start, so one visit's images sort together. Returns the stamp
    and one record per written image (kept in the state database, not next to the images).
    """
    flat = [f for p in chosen for f in p]
    if not flat:
        return None, []
    cfg.incoming.mkdir(parents=True, exist_ok=True)
    st = stamp(start, cfg.tz)
    n = next_index(cfg.incoming, st)
    records = []
    for p in chosen:
        for rank, f in enumerate(p, start=1):
            full = frames.get(f.t)
            if full is None:
                LOG.warning("frame at %.1f s was not kept", f.t)
                continue
            name = f"{st}_{n}.jpg"
            n += 1
            box = f.box_full
            if not dry_run:
                tmp = cfg.incoming / f".tmp-{name}"
                cv2.imwrite(str(tmp), crop_face(full, box, cfg.crop_margin, cfg.crop_max_side),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                tmp.rename(cfg.incoming / name)
            records.append({
                "file": name, "person": f.person, "rank": rank, "frigate_event": f.event,
                "offset_s": round(float(f.t), 1),
                "wall_time": dt.datetime.fromtimestamp(start + f.t, cfg.tz).isoformat(timespec="seconds"),
                "box_fullres": [int(v) for v in box], "yunet": round(float(f.det), 3),
                "ediffiqa": round(float(f.quality), 3), "score": round(float(f.score), 3),
                "yaw_deg": round(float(f.yaw), 1), "roll_deg": round(float(f.roll), 1),
                "sharpness": round(float(f.sharpness), 1), "luminance": round(float(f.lum), 1)})
    return (st if records else None), records


def analyse_clip(cfg: Config, analyzer: Analyzer, clip: Path, use_vaapi: bool,
                 keep: int = 12):
    """Return (frames sampled, all faces, {t: full frame} for the best frames with faces)."""
    faces: list[Face] = []
    kept: dict[float, tuple[float, np.ndarray]] = {}
    frames = 0
    for t, full in sample_frames(clip, cfg, use_vaapi):
        frames += 1
        aw, ah = analysis_size(full.shape[1], full.shape[0], cfg.analysis_height)
        small = cv2.resize(full, (aw, ah), interpolation=cv2.INTER_AREA) \
            if (aw, ah) != (full.shape[1], full.shape[0]) else full
        found = analyzer.evaluate(small, t, full.shape[1] / aw)
        faces += found
        best = max((f.score for f in found if not f.reject), default=None)
        if best is not None:
            kept[t] = (best, full)
            if len(kept) > keep:
                del kept[min(kept, key=lambda k: kept[k][0])]
    return frames, faces, {t: fr for t, (_, fr) in kept.items()}


# ---------------------------------------------------------------------------- pick

class Picker:
    def __init__(self, cfg: Config, frigate: Frigate, state: State, analyzer: Analyzer | None,
                 use_vaapi: bool = True, dry_run: bool = False) -> None:
        self.cfg, self.frigate, self.state, self.an = cfg, frigate, state, analyzer
        self.use_vaapi, self.dry_run = use_vaapi, dry_run
        self.stats = {"processed": 0, "folders": 0, "failed": 0}

    def candidates(self, since: float | None = None) -> list[dict]:
        cfg = self.cfg
        now = time.time()
        wm = self.state.watermark()
        if since is not None:
            after = since
        elif wm is None:
            after = now - cfg.backfill_days * 86400
        else:
            after = wm - 3600
        items = self.frigate.reviews(cfg.camera, after, now)
        out = []
        for r in items:
            st = self.state.status(r["id"])
            if st and st[0] != "retry":
                continue
            if not r.get("end_time") or r["end_time"] > now - cfg.settle_s:
                continue
            data = r.get("data") or {}
            if "person" not in (data.get("objects") or []):
                continue
            if cfg.zone and cfg.zone not in (data.get("zones") or []):
                continue
            out.append(r)
        seen = {r["id"] for r in out}
        out += [r for r in self.state.retries(cfg.max_attempts) if r["id"] not in seen]
        return sorted(out, key=lambda r: r["start_time"])

    def window(self, review: dict) -> tuple[float, float, list[dict]]:
        start, end = review["start_time"], review["end_time"]
        ids = (review.get("data") or {}).get("detections") or []
        events = []
        for eid in ids:
            ev = self.frigate.event(eid)
            if ev:
                events.append(ev)
                start = min(start, ev["start_time"])
                end = max(end, ev.get("end_time") or end)
        # A person can be tracked long before entering the zone, and Frigate keeps no video of a
        # stationary stretch: anchor on the visit, at most 30 s before it (seen 2026-11-09).
        start = max(start, review["start_time"] - 30) - 2
        end += 2
        if end - start > self.cfg.max_clip_s:
            LOG.info("%s: %.0f s visit capped to %d s", review["id"], end - start, self.cfg.max_clip_s)
            end = start + self.cfg.max_clip_s
        return start, end, events

    def process(self, review: dict) -> None:
        rid = review["id"]
        t0 = time.time()
        cpu0 = time.process_time()
        try:
            start, end, events = self.window(review)
            with tempfile.TemporaryDirectory(dir="/tmp") as td:
                clip = Path(td) / "clip.mp4"
                self.frigate.clip(self.cfg.camera, start, end, clip)
                frames, faces, kept = analyse_clip(self.cfg, self.an, clip, self.use_vaapi)
                tracked = assign_to_tracks(faces, events, start, self.cfg)
                chosen = select(faces, self.cfg)
                meta = {
                    "source": "frigate", "camera": self.cfg.camera, "review_id": rid,
                    "event_ids": [e["id"] for e in events],
                    "persons_by": "frigate tracks" if tracked else "faces seen together",
                    "clip_start": dt.datetime.fromtimestamp(start, self.cfg.tz).isoformat(timespec="seconds"),
                    "clip_seconds": round(end - start, 1), "frames_sampled": frames,
                    "faces_found": len(faces),
                    "faces_passed": sum(1 for f in faces if not f.reject),
                    "rejected": _reject_counts(faces),
                    "frigate_review_url": f"/review?id={rid}"}
                folder, records = write_output(self.cfg, kept, start, chosen, self.dry_run)
        except requests.HTTPError as err:
            if err.response is not None and "No recordings" in err.response.text:
                LOG.info("%s: Frigate has no recordings for this visit", rid)
                if not self.dry_run:
                    self.state.save(rid, review["start_time"], "no_recording", review)
                return
            self._fail(rid, review, err)
            return
        except (OSError, requests.RequestException, subprocess.CalledProcessError, ValueError) as err:
            self._fail(rid, review, err)
            return
        self._done(rid, review, t0, cpu0, frames, faces, chosen, meta, folder, records)

    def _fail(self, rid: str, review: dict, err: Exception) -> None:
        LOG.warning("%s: %s", rid, err)
        self.stats["failed"] += 1
        if not self.dry_run:
            self.state.save(rid, review["start_time"], "retry", review, error=str(err)[:300])
            st = self.state.status(rid)
            if st and st[1] >= self.cfg.max_attempts:
                self.state.save(rid, review["start_time"], "failed", review, error=str(err)[:300])

    def _done(self, rid, review, t0, cpu0, frames, faces, chosen, meta, folder, records) -> None:
        persons = len([p for p in chosen if p])
        status = "done" if folder else "done_no_face"
        secs = round(time.time() - t0, 1)
        LOG.info("%s: %d frames, %d faces (%d passed), %d person(s) -> %s  [%.1f s, cpu %.1f s]",
                 rid, frames, len(faces), meta["faces_passed"], persons,
                 f"{folder}_1..{len(records)}" if folder else "no images",
                 secs, time.process_time() - cpu0)
        self.stats["processed"] += 1
        self.stats["folders"] += bool(folder)
        if not self.dry_run:
            self.state.save_faces(rid, meta, records)
            self.state.save(rid, review["start_time"], status, review, folder=folder,
                            faces=len(faces), persons=persons, seconds=secs)

    def run(self, since: float | None = None, limit: int | None = None) -> int:
        items = self.candidates(since)
        todo = items[: limit or self.cfg.max_items_per_round]
        for r in todo:
            self.process(r)
        return len(items) - len(todo)


def _reject_counts(faces: list[Face]) -> dict:
    out: dict[str, int] = {}
    for f in faces:
        for r in f.reject:
            out[r] = out.get(r, 0) + 1
    return out


# ---------------------------------------------------------------------------- train

NAME_RE = re.compile(r"^[A-Za-z0-9_ ]{1,64}$")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def train_pending(cfg: Config, frigate: Frigate, state: State, dry_run: bool = False) -> dict:
    stats = {"registered": 0, "failed": 0, "waiting": 0}
    if not cfg.pending.exists():
        return stats
    known: dict | None = None
    now = time.time()
    for person_dir in sorted(p for p in cfg.pending.iterdir() if p.is_dir()):
        name = person_dir.name
        files = sorted(f for f in person_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in IMAGE_EXT)
        if not files:
            continue
        if not NAME_RE.match(name):
            LOG.warning("pending/%s: invalid name (letters, digits, space, _ only); skipped", name)
            stats["waiting"] += len(files)
            continue
        for img in files:
            if now - img.stat().st_mtime < cfg.train_settle_s:
                stats["waiting"] += 1
                continue
            if dry_run:
                LOG.info("would register %s as %s", img.name, name)
                continue
            if known is None:
                known = frigate.faces()
            if name not in known:
                frigate.create_face(name)
                known[name] = []
            res = frigate.register(name, img)
            ok = bool(res.get("success"))
            msg = str(res.get("message", ""))
            state.log_trained(str(img.relative_to(cfg.data)), name, ok, msg)
            if ok:
                dest = cfg.trained / name
                dest.mkdir(parents=True, exist_ok=True)
                target = _unique(dest / img.name)
                shutil.move(str(img), target)
                with open(dest / "log.jsonl", "a") as log:
                    log.write(json.dumps({"time": dt.datetime.now(cfg.tz).isoformat(timespec="seconds"),
                                          "file": target.name, "name": name,
                                          "frigate": res}, ensure_ascii=False) + "\n")
                stats["registered"] += 1
                LOG.info("registered %s as %s", img.name, name)
            else:
                dest = person_dir / "failed"
                dest.mkdir(exist_ok=True)
                target = _unique(dest / img.name)
                shutil.move(str(img), target)
                target.with_name(target.name + ".reason.txt").write_text(msg + "\n")
                stats["failed"] += 1
                LOG.warning("Frigate refused %s for %s: %s", img.name, name, msg)
    return stats


def _unique(p: Path) -> Path:
    n = 2
    out = p
    while out.exists():
        out = p.with_name(f"{p.stem}_{n}{p.suffix}")
        n += 1
    return out


# ---------------------------------------------------------------------------- main

def in_pause(cfg: Config, now: dt.datetime) -> bool:
    lo, hi = cfg.pause_minutes
    return now.hour % cfg.pause_hours_mod == 0 and lo <= now.minute <= hi


def write_status(cfg: Config, state: State, extra: dict) -> None:
    incoming = list(cfg.incoming.glob("*.jpg")) if cfg.incoming.exists() else []
    pending = sum(1 for d in cfg.pending.glob("*/*") if d.is_file() and d.suffix.lower() in IMAGE_EXT) \
        if cfg.pending.exists() else 0
    status = {"time": dt.datetime.now(cfg.tz).isoformat(timespec="seconds"), "version": VERSION,
              "items": state.counts(), "incoming_images": len(incoming),
              "pending_files": pending, **extra}
    tmp = cfg.state_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(cfg.state_dir / "status.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="one round, then exit")
    ap.add_argument("--since", help="process visits from this date (YYYY-MM-DD), once")
    ap.add_argument("--redo", metavar="REVIEW_ID", help="process one visit again")
    ap.add_argument("--clip", type=Path, help="analyse a local mp4 (no Frigate); output to DATA_DIR")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--pick-only", action="store_true")
    ap.add_argument("--no-vaapi", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    os.umask(0o000)   # the owner moves files over the share with another uid
    cfg = Config()
    for d in (cfg.incoming, cfg.pending, cfg.trained, cfg.discarded, cfg.state_dir):
        d.mkdir(parents=True, exist_ok=True)
        if d != cfg.state_dir:
            d.chmod(0o777)     # the owner works over NFS as the anonymous user
    use_vaapi = not a.no_vaapi

    if a.clip:
        an = Analyzer(cfg)
        frames, faces, kept = analyse_clip(cfg, an, a.clip, use_vaapi)
        chosen = select(faces, cfg)
        start = a.clip.stat().st_mtime
        name, records = write_output(cfg, kept, start, chosen, a.dry_run)
        for r in records:
            LOG.info("  %s", json.dumps(r))
        LOG.info("%s: %d frames, %d faces, %d passed, persons %d -> %s", a.clip.name, frames,
                 len(faces), sum(1 for f in faces if not f.reject), len(chosen), name)
        return 0

    frigate = Frigate(cfg.frigate_url)
    state = State(cfg.state_dir / "state.sqlite")
    an = None if a.train_only else Analyzer(cfg)
    try:
        LOG.info("facepicker %s; Frigate %s; camera %s; data %s", VERSION, frigate.version(),
                 cfg.camera, cfg.data)
    except requests.RequestException as err:
        LOG.warning("Frigate not reachable yet: %s", err)

    if a.redo:
        picker = Picker(cfg, frigate, state, an, use_vaapi, a.dry_run)
        row = state.db.execute("SELECT review FROM items WHERE id=?", (a.redo,)).fetchone()
        if not row:
            LOG.error("unknown review id %s (not in the state database)", a.redo)
            return 1
        picker.process(json.loads(row[0]))
        return 0

    since = None
    if a.since:
        since = dt.datetime.strptime(a.since, "%Y-%m-%d").replace(tzinfo=cfg.tz).timestamp()

    while True:
        now = dt.datetime.now(cfg.tz)
        if in_pause(cfg, now) and not (a.once or a.since):
            time.sleep(60)
            continue
        extra: dict = {}
        backlog = 0
        try:
            if not a.pick_only:
                extra["train"] = train_pending(cfg, frigate, state, a.dry_run)
            if not a.train_only:
                picker = Picker(cfg, frigate, state, an, use_vaapi, a.dry_run)
                backlog = picker.run(since, limit=10**9 if since else None)
                extra["pick"] = picker.stats
                extra["backlog"] = backlog
        except requests.RequestException as err:
            LOG.warning("Frigate unavailable: %s", err)
            extra["error"] = str(err)[:200]
        if not a.dry_run:
            write_status(cfg, state, extra)
        if a.once or a.since:
            return 0
        time.sleep(5 if backlog else cfg.interval_s)


if __name__ == "__main__":
    sys.exit(main())
