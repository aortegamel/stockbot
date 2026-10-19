FROM docker/sandbox-templates:shell-docker
COPY --from=ghcr.io/astral-sh/uv:0.12.11 /uv /usr/local/bin/uv
USER root
ARG BUN_VERSION=1.4.2
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl unzip nodejs \
 && rm -rf /var/lib/apt/lists/* \
 && case "${TARGETARCH}" in \
      amd64) BUN_ARCH=x64; BUN_SHA256=36368faef7527875d5ffa52e53cd48021741f2a83eb6208a8dd64068d422a913;; \
      arm64) BUN_ARCH=aarch64; BUN_SHA256=54328bbc2d9c8e0c9f892c544d66c57a83b84139e34909e5ee81758f1ac8fda7;; \
      *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1;; \
    esac \
 && curl -fsSL -o /tmp/bun.zip "https://github.com/oven-sh/bun/releases/download/bun-v${BUN_VERSION}/bun-linux-${BUN_ARCH}.zip" \
 && echo "${BUN_SHA256}  /tmp/bun.zip" | sha256sum -c - \
 && unzip -q /tmp/bun.zip -d /tmp \
 && mv "/tmp/bun-linux-${BUN_ARCH}/bun" /usr/local/bin/bun \
 && chmod +x /usr/local/bin/bun \
 && rm -rf /tmp/bun.zip "/tmp/bun-linux-${BUN_ARCH}"
WORKDIR /app
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
COPY .python-version requirements.txt package.json bun.lock ./
RUN bun run setup \
 && bun install
COPY . .
RUN id -u agent >/dev/null 2>&1 || useradd -m -u 1000 -d /home/agent -s /bin/bash agent \
 && mkdir -p /home/agent /data \
 && chown -R agent:agent /app /data /home/agent \
 && mkdir -p /etc/sudoers.d \
 && printf 'Defaults env_keep += "http_proxy https_proxy HTTP_PROXY HTTPS_PROXY DOCKER_SANDBOX_*"\n' > /etc/sudoers.d/10-sandbox-proxy \
 && chmod 440 /etc/sudoers.d/10-sandbox-proxy
USER agent
ENV STOCKBOT_DATA_DIR=/data
ENV PATH="/app/venv/bin:$PATH"
