# facepicker

Picks the best face images from Frigate's doorbell recordings, so a person can choose the
training images for Frigate's face recognition. It runs on the Frigate host, as a container
next to Frigate.

## What it does

Every 5 minutes:

1. **Pick.**
   - Finds new doorbell visits: Frigate review items with a person in `doorbell_zone`, ended
     at least 2 minutes ago.
   - Downloads each visit's clip from Frigate's API and decodes 1 frame per second at full
     resolution (VA-API on the iGPU, with a CPU fallback).
   - Finds faces with **YuNet** (the same detector Frigate uses).
   - Drops faces that fail the quality gates: detection score, size, frame edge, head turn
     and tilt, eye distance, blur and exposure.
   - Ranks the rest with **eDifFIQA** (a face-image-quality model), weighted for size and
     frontality.
   - **Keeps visitors apart:** each face is tied to the Frigate person track standing
     closest to it at that moment. Each tracked person gets their own top 3, so two people
     who come one after the other never share a folder's ranking. A "face" with no tracked
     person near it (a wheel, a poster) is rejected.
   - Without track data, it falls back to: two faces in the same frame are two people.
   - Writes flat files: `incoming/<YYYY-MM-DD_HH-MM-SS>_<n>.jpg`. The stamp is the visit's
     start, so one visit's images sort together. `n` is the rank, 1 the best; with several
     people, person 1's images come first, then person 2's. When two visits start in the same
     second, the numbering simply continues.
   - The details (scores, the time inside the clip, the Frigate review and event ids) go into
     the state database's `faces` table, not next to the images.
2. **Train.**
   - Moves every image you drop into `pending/<name>/` to Frigate:
     `POST /api/faces/<name>/register`.
   - Success: the image moves to `trained/<name>/`, and a line is added to `log.jsonl`.
   - Refused: it moves to `pending/<name>/failed/`, with a `.reason.txt`.

It never writes to or deletes Frigate's media. Nothing is pruned: unpicked folders are kept.

## Your part

1. Look in `/path/to/facetraining/incoming/`.
2. Move the best image of a person to `pending/<name>/`, for example `pending/kim/`. Use
   letters, digits, spaces and `_` in names, and the same lowercase name each time.
3. Move the images you don't want to `discarded/` before you delete them. All five folders
   are writable over NFS as the anonymous user; the picker recreates them at each start.
4. Within 5 minutes it's in Frigate's face library. Start with 1–3 images per person, then
   use Frigate's own **Train** tab, which starts filling once the library has someone in it.

## Running

```bash
docker compose build
docker compose run --rm facepicker --once --dry-run   # what would happen, writes nothing
docker compose run --rm facepicker --since 2026-10-10 # one-off backfill from a date
docker compose up -d                                  # the service
docker logs -f facepicker
```

Other options:
- `--redo <review id>`: run one visit again.
- `--clip file.mp4`: analyse a local file, without Frigate.
- `--train-only` / `--pick-only`.
- `--no-vaapi`.

**Limits:**
- 1 CPU, lowest priority (`cpu_shares` 128, `nice 19`), 1 GB of memory.
- It pauses from :10 to :18 every 6 hours, while Frigate's cron restarts it.

**State:**
- `/data/.state/state.sqlite`: the visits processed (`items`), each image's scores and
  origin (`faces`), and what was trained (`trained`).
- `/data/.state/status.json`: the last round's figures. No notifications, by the owner's
  choice.

Settings are environment variables, listed in `Config` in `facepicker.py`: the gates
(`MIN_DET_SCORE`, `MIN_FACE_H`, `MAX_YAW`, …), `BACKFILL_DAYS`, `INTERVAL_S`, `TRACK_MAX_DX`.

## Tests

```bash
docker build -t facepicker:test .
docker run --rm --entrypoint sh -v $PWD:/src:ro -e MODEL_DIR=/app/models facepicker:test \
  -c "pip install -q pytest && cd /tmp && cp -r /src/tests /src/facepicker.py . && python -m pytest -q tests"
```

**No real faces in this repository, ever.** The tests use synthetic data, and `.gitignore`
blocks images and videos.

## Models and licences

From the OpenCV model zoo, downloaded at build time and checked against `models.sha256`:

| Model | Use | Licence |
|---|---|---|
| YuNet `face_detection_yunet_2023mar` | face detection | MIT |
| eDifFIQA(T) `ediffiqa_tiny_jun2024` (Babnik, Peer, Štruc, 2024) | face-image quality | CC BY 4.0 |
| SFace `face_recognition_sface_2021dec` | face alignment and embeddings | Apache-2.0 |

The Coral isn't used: Frigate holds it exclusively.
