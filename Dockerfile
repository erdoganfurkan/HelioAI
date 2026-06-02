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
COPY --from=builder /app/helioai /app/helioai
# Recipes ship with the image; user data (chroma, sessions, workspace) comes from the volume
COPY data/recipes/ /app/data/recipes/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

VOLUME /app/data
EXPOSE 7890

ENTRYPOINT ["helioai"]
CMD ["serve", "--web", "--host", "0.0.0.0"]
