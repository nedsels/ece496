FROM debian:bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang-format \
    cmake \
    curl \
    git \
    perl \
    vim-tiny \
    wget \
  && rm -rf /var/lib/apt/lists/*

# CUDA 11.8 toolkit only (no driver). --override skips the host GCC version check.
RUN curl -fsSL -o /tmp/cuda.run \
    https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run \
  && sh /tmp/cuda.run --silent --toolkit --override \
  && rm -f /tmp/cuda.run

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVCC_PREPEND_FLAGS="--allow-unsupported-compiler"

WORKDIR /workspace
CMD ["bash", "tests/compile_test.sh"]