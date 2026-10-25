FROM debian:bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libxml2 \
    perl \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --http1.1 --retry 10 --retry-delay 5 --retry-all-errors -C - \
    -o /tmp/cuda.run \
    https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

RUN sh /tmp/cuda.run --silent --toolkit --override \
  && rm -f /tmp/cuda.run

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"
ENV NVCC_PREPEND_FLAGS="--allow-unsupported-compiler"

RUN apt-get update && apt-get install -y --no-install-recommends \
    clang-format \
    cmake \
    gdb \
    git \
    openssh-client \
    vim-tiny \
    wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
