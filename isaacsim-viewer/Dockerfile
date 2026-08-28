# Build context is this directory:
#   docker build -t pait/isaacsim-viewer:latest ./isaacsim-viewer
FROM node:20-alpine

# ---- Non-root user pait (UID/GID 1000) ----
# node:alpine already ships a "node" user at 1000:1000, so rename it instead of
# creating a duplicate UID.
RUN deluser node 2>/dev/null || true; \
    addgroup -g 1000 pait 2>/dev/null || true; \
    adduser -u 1000 -G pait -s /bin/sh -D pait

WORKDIR /app
RUN chown pait:pait /app
USER pait

COPY --chown=pait:pait package*.json ./

# Pulls @nvidia/omniverse-webrtc-streaming-library from NVIDIA's registry, so the
# build needs outbound access to edge.urm.nvidia.com.
RUN npm ci

COPY --chown=pait:pait . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
