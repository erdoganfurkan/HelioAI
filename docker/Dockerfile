# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install deps before copying source — better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY helioai/ helioai/
COPY README.md ./
RUN uv sync --frozen --no-dev

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

COPY --from=builder /app/.venv /app/.venv
# Recipes live in helioai/data/recipes/ and travel with the package; user data
# (chroma, sessions, workspace) comes from the volume mounted at /app/data.
COPY --from=builder /app/helioai /app/helioai

WORKDIR /app
# Without HELIOAI_DATA_DIR the runtime stage has no pyproject.toml, so the repo
# check fails and data_dir lands in /root/.local/share — outside the volume below,
# which made sessions and the Chroma index vanish on every recreate.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HELIOAI_DATA_DIR=/app/data

# bubblewrap is the sandbox's only real isolation, and SECURITY.md tells people to
# use this image to get it. preexec_fn drops the sandbox subprocess to the dedicated
# user so LLM-generated code cannot read /proc/<server>/environ (see sandbox.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /bin/false helioai-sandbox

VOLUME /app/data
EXPOSE 7890

ENTRYPOINT ["helioai"]
CMD ["serve", "--web", "--host", "0.0.0.0"]
