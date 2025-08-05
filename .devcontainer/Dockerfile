# Base image
FROM ghcr.io/astral-sh/uv:0.8.4-bookworm-slim

# Install git
RUN apt-get update && \
  apt-get install -y git


# Build tools for kahypar
# TODO: must remove this and install kahypar from pypi when the wheel for arch64 dev container is provided
RUN apt-get install -y build-essential libboost-all-dev cmake

# Clean up
RUN apt-get clean && \
  rm -rf /var/lib/apt/lists/*
