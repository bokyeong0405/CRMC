# Lightweight client image: just numpy + httpx, no TF.
# Build context (set in docker-compose.yml) is the *repo root*, so paths
# below are relative to that root.

FROM python:3.10-slim

WORKDIR /app

# Install dependencies first so they get cached as their own layer.
COPY user/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the user package. We don't need the rest of the repo here.
COPY user /app/user

ENV PYTHONUNBUFFERED=1

# Default command runs the client once and exits. The compose file
# overrides this when needed (e.g. running multiple iterations).
CMD ["python", "-m", "user.client"]
