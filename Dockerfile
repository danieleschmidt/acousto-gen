# Multi-stage Dockerfile for Acousto-Gen
# Optimized for both development and production use

# ==============================================================================
# Base Stage - Common dependencies
# ==============================================================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libfftw3-dev \
    libhdf5-dev \
    libsndfile1-dev \
    libasound2-dev \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create app user
RUN useradd --create-home --shell /bin/bash acousto
USER acousto
WORKDIR /home/acousto

# Create virtual environment
RUN python -m venv /home/acousto/venv
ENV PATH="/home/acousto/venv/bin:$PATH"

# ==============================================================================
# Dependencies Stage - Install Python dependencies
# ==============================================================================
FROM base as dependencies

# Copy requirements first for better caching
COPY --chown=acousto:acousto pyproject.toml ./
COPY --chown=acousto:acousto README.md ./

# Install dependencies
RUN pip install --no-cache-dir -e .[full]

# ==============================================================================
# Development Stage - For development and testing
# ==============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    pre-commit \
    pytest-xvfb

# Copy source code
COPY --chown=acousto:acousto . .

# Install in development mode
RUN pip install -e .[dev]

# Install pre-commit hooks
RUN git init . && pre-commit install || true

# Expose ports for Jupyter and other services
EXPOSE 8888 6006 8080

# Default command for development
CMD ["bash"]

# ==============================================================================
# Testing Stage - For CI/CD testing
# ==============================================================================
FROM dependencies as testing

# Copy source code and tests
COPY --chown=acousto:acousto acousto_gen/ ./acousto_gen/
COPY --chown=acousto:acousto tests/ ./tests/
COPY --chown=acousto:acousto conftest.py pytest.ini ./

# Install package in test mode
RUN pip install -e .[dev]

# Run tests by default
CMD ["pytest", "--cov=acousto_gen", "--cov-report=html", "--cov-report=term-missing"]

# ==============================================================================
# Production Stage - Minimal production image
# ==============================================================================
FROM dependencies as production

# Copy only necessary source code
COPY --chown=acousto:acousto acousto_gen/ ./acousto_gen/
COPY --chown=acousto:acousto README.md LICENSE CHANGELOG.md ./

# Install package in production mode
RUN pip install --no-deps .

# Remove unnecessary files
RUN pip uninstall -y pip setuptools wheel && \
    rm -rf ~/.cache/pip && \
    find /home/acousto/venv -name "*.pyc" -delete && \
    find /home/acousto/venv -name "__pycache__" -type d -exec rm -rf {} + || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import acousto_gen; print('OK')" || exit 1

# Default command
CMD ["acousto-gen", "--help"]

# ==============================================================================
# GPU Stage - For GPU-accelerated workloads
# ==============================================================================
FROM production as gpu

# Verify CUDA installation
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Set GPU-specific environment variables
ENV ACOUSTO_DEVICE=cuda
ENV CUDA_VISIBLE_DEVICES=0

# Default command for GPU workloads
CMD ["acousto-gen", "--device", "cuda"]

# ==============================================================================
# Jupyter Stage - For interactive development
# ==============================================================================
FROM development as jupyter

# Install additional Jupyter extensions
RUN pip install --no-cache-dir \
    jupyter-contrib-nbextensions \
    jupyter_nbextensions_configurator \
    plotly \
    ipyvolume

# Configure Jupyter
RUN jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable --user

# Create notebooks directory
RUN mkdir -p /home/acousto/notebooks

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ==============================================================================
# Build metadata
# ==============================================================================

LABEL maintainer="Acousto-Gen Team <maintainers@acousto-gen.org>"
LABEL version="0.1.0"
LABEL description="Generative acoustic holography toolkit"
LABEL org.opencontainers.image.title="Acousto-Gen"
LABEL org.opencontainers.image.description="Generative acoustic holography toolkit for creating 3D pressure fields"
LABEL org.opencontainers.image.url="https://github.com/danieleschmidt/acousto-gen"
LABEL org.opencontainers.image.documentation="https://acousto-gen.readthedocs.io"
LABEL org.opencontainers.image.source="https://github.com/danieleschmidt/acousto-gen"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.created="2025-01-01T00:00:00Z"
LABEL org.opencontainers.image.licenses="MIT"