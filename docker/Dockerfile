FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# The devel image already ships Nsight Compute (cuda-nsight-compute-13-2) but
# only symlinks nsys when that package happens to contain it, so Nsight
# Systems is installed explicitly from the CUDA apt repo the image configures.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        make cmake g++ git \
        cuda-nsight-systems-13-2=13.2.1-1 \
    && rm -rf /var/lib/apt/lists/* \
    && if ! command -v nsys >/dev/null; then \
        ln -s "$(find /opt/nvidia/nsight-systems -type f -path '*/bin/nsys' | head -n 1)" /usr/local/bin/nsys; \
    fi \
    && command -v ncu && command -v nsys

# Ubuntu 24.04 marks the system Python as externally managed, so pip
# installs go into a venv.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu132 \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

WORKDIR /workspace
EXPOSE 8888

# Runs as root so files written to the bind-mounted /workspace are not
# blocked by a UID mismatch with the host user.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
