FROM python:3.12-slim-trixie
# ffmpeg with VA-API, and Intel's iHD driver (non-free) for the HD 530 (Gen9)
RUN sed -i 's/^Components: main$/Components: main non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg intel-media-va-driver-non-free vainfo curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt models.sha256 fetch_models.sh ./
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt \
 && sh fetch_models.sh /app/models
COPY facepicker.py ./
ENV PYTHONUNBUFFERED=1 LIBVA_DRIVER_NAME=iHD
HEALTHCHECK --interval=5m --timeout=20s CMD python -c "import json,time,sys; s=json.load(open('/data/.state/status.json')); sys.exit(0)" || exit 1
ENTRYPOINT ["nice", "-n", "19", "python", "/app/facepicker.py"]
