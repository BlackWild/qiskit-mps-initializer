# Base image
FROM ghcr.io/astral-sh/uv:0.7.6-python3.13-bookworm-slim

# Install git
RUN apt-get update && \
    apt-get install -y git

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*