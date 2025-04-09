FROM nvcr.io/nvidia/pytorch:25.01-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    swig \
    graphviz \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Create directory for TorchCAP
WORKDIR /workspace/torchcap

# Copy and install TorchCAP
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY configs/ ./configs/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY torchcap/ ./torchcap/
COPY pyproject.toml ./pyproject.toml
COPY MANIFEST.in ./MANIFEST.in
COPY README.md ./README.md
COPY LICENSE ./LICENSE

RUN pip install -e .

# Set default command
CMD ["/bin/bash"]
