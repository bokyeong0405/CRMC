# Device image: TensorFlow + FastAPI + httpx. This is the heavyweight one
# (~1.5 GB once TF is installed) — three identical copies will run as
# server-1/server-2/server-3 with different env vars.

FROM python:3.10-slim

WORKDIR /app

# TF wheels need a couple of glibc-side libraries the slim base lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY device/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Server imports model.resnet_split, so the model package must be in the image.
COPY model /app/model
COPY device /app/device

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "-m", "device.server"]
