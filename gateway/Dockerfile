# Build context is the repo root (see docker-compose.yml), so monitor/ is shared
# with training/ and eval/ instead of copied.
FROM python:3.12-slim-bookworm

ARG WITH_ENCODER=0
WORKDIR /app
COPY gateway/requirements.txt gateway/requirements-encoder.txt gateway/
RUN pip install --no-cache-dir -r gateway/requirements.txt \
 && if [ "$WITH_ENCODER" = "1" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.0 \
      && pip install --no-cache-dir transformers==4.51.3; \
    fi

COPY monitor/ monitor/
COPY gateway/ gateway/
# policy/ and models/ are mounted read-only by compose so edits do not need a rebuild.

RUN useradd --uid 10001 --create-home gateway && mkdir -p /logs && chown gateway /logs
USER gateway
EXPOSE 8080
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]
