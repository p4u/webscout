# ── Stage 1: Node builder ─────────────────────────────────────────────────────
# Install dependencies first so that a source-only change does not reinstall
# the world.
FROM node:22-slim AS builder

WORKDIR /app

# Copy manifests only; npm ci downloads and vendors all dependencies
# (marked, dompurify, vite) so the runtime image needs no CDN access.
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the UI source and produce the production bundle.
COPY . .
RUN npm run build

# ── Stage 2: nginx serving static files ──────────────────────────────────────
FROM nginx:alpine AS runtime

# Replace the default nginx config with one that:
#  - Serves ui/dist as static files with index.html fallback for SPA routing.
#  - Proxies /api/ to the api container (same origin → no CORS needed).
#  - Disables buffering so the NDJSON search stream reaches the browser live.
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
