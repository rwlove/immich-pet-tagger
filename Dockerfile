FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# GPU support:
#   NVIDIA: set CUDA=true
#   AMD:    set ROCM=true  (requires ROCm drivers on the host)
#   None:   leave both false (CPU-only, slow but works)
ARG CUDA=false
ARG ROCM=false
# P40-compat fork: torch 2.7+cu128 dropped sm_61 (Pascal); stay on the
# 2.6.x+cu124 line so this image runs on a P40. Revert to upstream pins
# when this image no longer targets Pascal hardware.
RUN if [ "$CUDA" = "true" ]; then \
      pip install --no-cache-dir \
        torch==2.6.0+cu124 \
        torchvision==0.21.0+cu124 \
        --extra-index-url https://download.pytorch.org/whl/cu124; \
    elif [ "$ROCM" = "true" ]; then \
      pip install --no-cache-dir \
        torch==2.7.0 \
        torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/rocm6.3; \
    else \
      pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless

# Pre-stage YOLO weights at build time so first-run detection works
# without internet access and the container can be deployed with a
# read-only rootfs or a non-root user (ultralytics would otherwise try
# to download the weights into the CWD at first inference).
ADD https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt /app/yolov8m.pt
RUN chmod 0644 /app/yolov8m.pt

COPY VERSION .
COPY app/ .

# /data is the mounted volume: pets/luna/, pets/config.json, state files, logs
VOLUME ["/data"]

EXPOSE 8000

CMD ["python", "main.py"]
