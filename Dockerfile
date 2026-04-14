FROM rust:1.80-slim AS builder

RUN apt-get update && apt-get install -y python3 python3-pip python3-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages maturin

WORKDIR /ohe-rs
COPY Cargo.toml Cargo.lock build.rs pyproject.toml ./
COPY src/ src/
COPY python/ python/

RUN maturin build --release --out /wheels

FROM python:3.12-slim

COPY --from=builder /wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Verify installation
RUN python -c "from ohe_rs import encode_sparse, gpu_available; print('ohe-rs OK, GPU:', gpu_available())"

ENTRYPOINT ["python"]
