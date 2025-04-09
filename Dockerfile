FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    swig \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN python -m pip install --upgrade pip setuptools

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
