# Mandated ROCm base — checked by layer identity. DO NOT change, DO NOT squash.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV HF_HOME=/models DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --no-deps: plain pip upgrades swap ROCm torch for CUDA (documented trap).
# Verify torch still ends in +rocm after this layer (CI checks too).
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt \
 && python3 -c "import torch; assert torch.__version__.endswith('+rocm'), torch.__version__"

# Bake weights: NO network at evaluation. ~2GB total, well under 60 GiB cap.
COPY scripts/fetch_weights.py /tmp/fetch_weights.py
RUN python3 /tmp/fetch_weights.py

COPY app/ /app/
RUN mkdir -p /app/corpus /app/output /app/index

# No service/port/entrypoint: harness execs into the running container.
# --index starts the resident daemon itself (model loaded once).
CMD ["sleep", "infinity"]
