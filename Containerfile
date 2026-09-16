FROM registry.access.redhat.com/ubi9/python-311:latest AS builder
USER 0
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --prefix=/install .

FROM registry.access.redhat.com/ubi9/python-311-minimal:latest
USER 0
COPY --from=builder /install /usr/local
WORKDIR /app
COPY config ./config
RUN chown -R 1001:0 /app && chmod -R g=u /app
USER 1001
EXPOSE 8080
ENV GPU_PROFILES_PATH=/app/config/gpu-profiles.yaml
CMD ["uvicorn", "gpu_usage_calculator.app:app", "--host", "0.0.0.0", "--port", "8080"]
