FROM python:3.12-slim-bookworm

# A few everyday tools the agent can use. No compilers, no ssh, no sudo.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates git jq \
 && rm -rf /var/lib/apt/lists/*

COPY tool_server.py entrypoint.sh /opt/tools/
RUN useradd --uid 10002 --create-home agent && mkdir -p /work && chown agent /work
USER agent
WORKDIR /work
EXPOSE 9000
ENTRYPOINT ["/opt/tools/entrypoint.sh"]
