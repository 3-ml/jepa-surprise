FROM python:3.11-slim@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libdbus-1-3 \
      libegl1 \
      libfontconfig1 \
      libfreetype6 \
      libgl1 \
      libgomp1 \
      libice6 \
      libsm6 \
      libwayland-client0 \
      libx11-6 \
      libxcursor1 \
      libxext6 \
      libxfixes3 \
      libxi6 \
      libxinerama1 \
      libxkbcommon0 \
      libxkbcommon-x11-0 \
      libxrandr2 \
      libxrender1 \
      libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /opt/kubric-runtime/requirements.lock
RUN python -m pip install --no-cache-dir --upgrade pip==26.0.1 \
    && python -m pip install --no-cache-dir -r /opt/kubric-runtime/requirements.lock
COPY hypertune.py /usr/local/lib/python3.11/site-packages/hypertune.py

WORKDIR /workspace
