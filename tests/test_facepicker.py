"""Tests without real faces: nothing in this repository may show a person."""
import datetime as dt
import json
import os
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import facepicker as fp


def cfg(tmp_path):
    c = fp.Config()
    c.data = tmp_path
    c.tz = ZoneInfo("Europe/Stockholm")
    for d in (c.incoming, c.pending, c.trained, c.state_dir):
        d.mkdir(parents=True, exist_ok=True)
    return c


def face(t, score, emb, reject=()):
    row = np.array([100, 100, 200, 260, 150, 180, 250, 180, 200, 230, 160, 280, 240, 280, 0.9],
                   np.float32)
    f = fp.Face(t=t, row=row, scale=2.0, det=0.9, quality=score, score=score,
                emb=np.array(emb, np.float32))
    f.reject = list(reject)
    return f


def test_pose_frontal_and_turned():
    row = np.zeros(15, np.float32)
    row[4:10] = [100, 100, 200, 100, 150, 150]          # nose centred
    yaw, roll = fp.pose(row)
    assert abs(yaw) < 1e-6 and abs(roll) < 1e-6
    row[8] = 200                                          # nose under the left eye
    assert fp.pose(row)[0] == pytest.approx(45.0)
    row[4:8] = [100, 100, 200, 200]                       # eye line at 45 degrees
    assert fp.pose(row)[1] == pytest.approx(45.0)


def test_rank_score_prefers_large_frontal():
    assert fp.rank_score(0.5, 300, 0) == pytest.approx(0.5)
    assert fp.rank_score(0.5, 75, 0) == pytest.approx(0.25)
    assert fp.rank_score(0.5, 600, 30) == pytest.approx(0.25)


def test_select_one_person_unless_seen_together(tmp_path):
    c = cfg(tmp_path)
    a, b = [1, 0, 0], [0, 1, 0]
    # different embeddings but never in the same frame: one person
    faces = [face(0, 0.9, a), face(0.5, 0.88, a), face(1, 0.85, b), face(3, 0.8, a),
             face(4, 0.99, a, reject=["blur"])]
    chosen = fp.select(faces, c)
    assert len(chosen) == 1
    assert [f.t for f in chosen[0]] == [0, 1, 3]          # 0.5 is within 1 s of 0


def test_select_two_people_in_one_frame(tmp_path):
    c = cfg(tmp_path)
    a, b = [1, 0, 0], [0, 1, 0]
    faces = [face(0, 0.9, a), face(0, 0.6, b), face(2, 0.8, a), face(3, 0.7, [0.1, 0.9, 0]),
             face(5, 0.5, b)]
    chosen = fp.select(faces, c)
    assert len(chosen) == 2
    assert [f.t for f in chosen[0]] == [0, 2]
    assert [f.t for f in chosen[1]] == [3, 0, 5]
    assert {f.person for f in chosen[1]} == {2}


def test_select_caps_persons(tmp_path):
    c = cfg(tmp_path)
    faces = [face(0, 1 - i / 10, np.eye(5)[i]) for i in range(5)]
    assert len(fp.select(faces, c)) == 3


def test_next_index_continues_after_existing(tmp_path):
    assert fp.next_index(tmp_path, "2026-11-09_14-21-07") == 1
    for n in (1, 2, 3):
        (tmp_path / f"2026-11-09_14-21-07_{n}.jpg").write_bytes(b"x")
    (tmp_path / "2026-11-09_14-21-07_x.jpg").write_bytes(b"x")
    assert fp.next_index(tmp_path, "2026-11-09_14-21-07") == 4
    assert fp.next_index(tmp_path, "2026-11-09_14-21-08") == 1


def test_crop_face_clamps_and_limits():
    img = np.zeros((1920, 2560, 3), np.uint8)
    c = fp.crop_face(img, [2400, 1800, 300, 300], 0.4, 1024)
    assert c.shape[0] > 0 and c.shape[1] > 0
    big = fp.crop_face(img, [100, 100, 1500, 1500], 0.4, 1024)
    assert max(big.shape[:2]) == 1024


def test_pause_window():
    c = fp.Config()
    tz = ZoneInfo("Europe/Stockholm")
    assert fp.in_pause(c, dt.datetime(2026, 1, 1, 12, 13, tzinfo=tz))
    assert fp.in_pause(c, dt.datetime(2026, 1, 1, 0, 10, tzinfo=tz))
    assert not fp.in_pause(c, dt.datetime(2026, 1, 1, 12, 19, tzinfo=tz))
    assert not fp.in_pause(c, dt.datetime(2026, 1, 1, 13, 13, tzinfo=tz))


def test_state_retry_then_failed(tmp_path):
    s = fp.State(tmp_path / "s.sqlite")
    rev = {"id": "r1", "start_time": 1.0}
    for _ in range(3):
        s.save("r1", 1.0, "retry", rev)
    assert s.status("r1") == ("retry", 3)
    assert [r["id"] for r in s.retries(5)] == ["r1"]
    assert s.retries(3) == []
    s.save("r1", 1.0, "done", rev, folder="x")
    assert s.status("r1")[0] == "done" and s.watermark() == 1.0


class FakeFrigate:
    def __init__(self, reviews=(), register_ok=True):
        self._reviews = list(reviews)
        self.register_ok = register_ok
        self.registered, self.created = [], []

    def reviews(self, camera, after, before):
        return [r for r in self._reviews if after <= r["start_time"] <= before]

    def event(self, eid):
        return None

    def faces(self):
        return {"kim": []}

    def create_face(self, name):
        self.created.append(name)

    def register(self, name, path):
        self.registered.append((name, path.name))
        return {"success": self.register_ok, "message": "ok" if self.register_ok else "No face was detected."}


def review(rid, start, end, objects=("person",), zones=("doorbell_zone",)):
    return {"id": rid, "start_time": start, "end_time": end,
            "data": {"objects": list(objects), "zones": list(zones), "detections": []}}


def test_candidates_filters(tmp_path):
    c = cfg(tmp_path)
    now = time.time()
    revs = [review("ok", now - 3000, now - 2990),
            review("recent", now - 60, now - 30),                 # not settled
            review("open", now - 600, None),                        # still running
            review("cat", now - 2000, now - 1990, objects=("cat",)),
            review("street", now - 1500, now - 1490, zones=())]
    st = fp.State(c.state_dir / "s.sqlite")
    p = fp.Picker(c, FakeFrigate(revs), st, analyzer=None)
    assert [r["id"] for r in p.candidates()] == ["ok"]
    st.save("ok", now - 3000, "done", revs[0])
    assert p.candidates() == []


def write_img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xe0 not a real jpeg")
    old = time.time() - 120
    os.utime(path, (old, old))


def test_train_pending_success_and_failure(tmp_path):
    c = cfg(tmp_path)
    st = fp.State(c.state_dir / "s.sqlite")
    write_img(c.pending / "kim" / "1.jpg")
    write_img(c.pending / "alex" / "2.jpg")
    write_img(c.pending / "bad-name" / "3.jpg")
    fresh = c.pending / "kim" / "fresh.jpg"
    fresh.write_bytes(b"x")                                   # still being copied
    ff = FakeFrigate()
    stats = fp.train_pending(c, ff, st)
    assert stats == {"registered": 2, "failed": 0, "waiting": 2}
    assert ff.created == ["alex"]
    assert (c.trained / "kim" / "1.jpg").exists() and (c.trained / "alex" / "2.jpg").exists()
    log = [json.loads(x) for x in (c.trained / "kim" / "log.jsonl").read_text().splitlines()]
    assert log[0]["name"] == "kim"
    assert fresh.exists() and (c.pending / "bad-name" / "3.jpg").exists()

    write_img(c.pending / "kim" / "4.jpg")
    stats = fp.train_pending(c, FakeFrigate(register_ok=False), st)
    assert stats["failed"] == 1
    assert (c.pending / "kim" / "failed" / "4.jpg").exists()
    assert "No face" in (c.pending / "kim" / "failed" / "4.jpg.reason.txt").read_text()


def test_write_output_flat_files(tmp_path):
    c = cfg(tmp_path)
    a, b = [1, 0, 0], [0, 1, 0]
    chosen = fp.select([face(0, 0.9, a), face(0, 0.6, b), face(3, 0.8, a)], c)
    frames = {0: np.full((1920, 2560, 3), 128, np.uint8), 3: np.full((1920, 2560, 3), 90, np.uint8)}
    ts = dt.datetime(2026, 11, 9, 14, 21, 7, tzinfo=c.tz).timestamp()
    st, recs = fp.write_output(c, frames, ts, chosen)
    assert st == "2026-11-09_14-21-07"
    assert sorted(p.name for p in c.incoming.iterdir()) == [f"{st}_{n}.jpg" for n in (1, 2, 3)]
    assert [(r["file"][-5:], r["person"], r["rank"]) for r in recs] == \
        [("1.jpg", 1, 1), ("2.jpg", 1, 2), ("3.jpg", 2, 1)]
    assert recs[1]["offset_s"] == 3 and recs[0]["wall_time"].endswith("14:21:07+01:00")
    json.dumps(recs)                                     # plain types only
    _, recs2 = fp.write_output(c, frames, ts, chosen)    # same second again: numbering continues
    assert [r["file"] for r in recs2] == [f"{st}_{n}.jpg" for n in (4, 5, 6)]


def test_sample_frames_rate(tmp_path):
    clip = tmp_path / "t.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=2560x1920:rate=20:duration=5.5", "-pix_fmt", "yuv420p", str(clip)],
                   check=True)
    c = fp.Config()
    c.fps = 1
    frames = list(fp.sample_frames(clip, c, use_vaapi=False))
    assert 5 <= len(frames) <= 6
    c.fps = 2
    two = list(fp.sample_frames(clip, c, use_vaapi=False))
    assert 11 <= len(two) <= 12 and two[1][0] == 0.5
    _t, fr = frames[0]
    assert fr.shape == (1920, 2560, 3)
    assert fp.analysis_size(2560, 1920, 1080) == (1440, 1080)


@pytest.mark.skipif(not os.environ.get("MODEL_DIR"), reason="needs the models")
def test_analyzer_on_blank_frame_finds_nothing():
    c = fp.Config()
    an = fp.Analyzer(c)
    assert an.evaluate(np.zeros((1080, 1440, 3), np.uint8), 0.0, 2.0) == []


def ev(eid, path, label="person"):
    return {"id": eid, "label": label, "data": {"path_data": [[[x, 0.8], t] for x, t in path]}}


def test_track_x_interpolates_and_limits_time():
    e = ev("a", [(0.2, 100.0), (0.4, 102.0)])
    assert fp.track_x(e, 101.0, 3) == pytest.approx(0.3)
    assert fp.track_x(e, 104.0, 3) == pytest.approx(0.4)
    assert fp.track_x(e, 110.0, 3) is None


def test_assign_separates_people_who_came_one_after_another(tmp_path):
    c = cfg(tmp_path)
    same = [1, 0, 0]                          # identical embeddings: appearance can't tell
    f1, f2, f3, f4 = (face(t, s, same) for t, s in ((1, 0.9), (2, 0.8), (6, 0.85), (7, 0.7)))
    for f, x in ((f1, 0.30), (f2, 0.32), (f3, 0.70), (f4, 0.72)):
        f.xn = x
    events = [ev("left", [(0.3, 1000.0), (0.33, 1003.0)]), ev("right", [(0.7, 1005.0), (0.7, 1008.0)])]
    assert fp.assign_to_tracks([f1, f2, f3, f4], events, 999.0, c)
    chosen = fp.select([f1, f2, f3, f4], c)
    assert [[f.event for f in p] for p in chosen] == [["left", "left"], ["right", "right"]]


def test_assign_rejects_face_far_from_any_person(tmp_path):
    c = cfg(tmp_path)
    wheel = face(2, 0.9, [1, 0, 0])
    wheel.xn = 0.75
    real = face(3, 0.5, [0, 1, 0])
    real.xn = 0.62
    assign = fp.assign_to_tracks([wheel, real], [ev("p", [(0.6, 1001.0), (0.6, 1003.0)])], 999.0, c)
    assert assign and wheel.reject == ["no_person"] and real.event == "p"
    assert [[f.t for f in p] for p in fp.select([wheel, real], c)] == [[3]]


def test_single_person_without_path_keeps_faces(tmp_path):
    c = cfg(tmp_path)
    f = face(2, 0.9, [1, 0, 0])
    fp.assign_to_tracks([f], [ev("p", [])], 999.0, c)
    assert f.event == "p" and not f.reject


def test_window_anchors_on_visit_not_on_early_tracking(tmp_path):
    c = cfg(tmp_path)

    class F(FakeFrigate):
        def event(self, eid):
            return {"id": eid, "label": "person", "start_time": 1000.0, "end_time": 2050.0}

    p = fp.Picker(c, F(), fp.State(c.state_dir / "s.sqlite"), analyzer=None)
    rev = {"id": "r", "start_time": 2000.0, "end_time": 2040.0, "data": {"detections": ["e"]}}
    start, end, events = p.window(rev)
    assert start == 1968.0 and end == 2052.0 and events[0]["id"] == "e"
