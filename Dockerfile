# webscout — one image with everything: the API, the MCP server at /mcp, the
# web UI, and the obscura headless browser. Built by CI and published to
# ghcr.io; runs as-is on Railway, DigitalOcean App Platform or any container
# host. Configure it with environment variables (see .env.example); the
# platform's $PORT is honoured.
#
# docker-compose.yml keeps the two-container setup (Dockerfile.api + ui/) for
# local use; this file is the deployable one.

# ── Stage 1: web UI ─────────────────────────────────────────────────────────
FROM node:22-slim AS ui
WORKDIR /app
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

# ── Stage 2: Rust dependencies (cached apart from the source) ───────────────
FROM rust:1-slim-bookworm AS deps
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs \
    && cargo build --release \
    && rm -rf src

# ── Stage 3: the binary ─────────────────────────────────────────────────────
FROM deps AS builder
COPY src/ src/
RUN touch src/main.rs && cargo build --release

# ── Stage 4: runtime ────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 1001 --create-home webscout \
    && mkdir -p /cache && chown webscout /cache

# Obscura: the headless browser webscout drives for search and page reads.
# Both binaries ship in one archive and must sit side by side. x86_64 only.
ARG OBSCURA_URL=https://github.com/h4ckf0r0day/obscura/releases/latest/download/obscura-x86_64-linux.tar.gz
RUN curl -fsSL "$OBSCURA_URL" -o /tmp/obscura.tar.gz \
    && tar xzf /tmp/obscura.tar.gz -C /usr/local/bin obscura obscura-worker \
    && chmod +x /usr/local/bin/obscura /usr/local/bin/obscura-worker \
    && rm -f /tmp/obscura.tar.gz \
    && obscura --version

COPY --from=builder /build/target/release/webscout /usr/local/bin/webscout
COPY --from=ui /app/dist /usr/share/webscout/ui

ENV WEBSCOUT_UI_DIR=/usr/share/webscout/ui \
    XDG_CACHE_HOME=/cache \
    PORT=8080

USER webscout
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# A shell so $PORT (set by Railway and similar platforms) reaches the flag.
CMD ["sh", "-c", "exec webscout --api --api-port \"${PORT:-8080}\""]
