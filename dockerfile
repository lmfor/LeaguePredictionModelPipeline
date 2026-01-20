FROM python:3.11-slim

ENV PYTHONBUFFERED = 1 \ 
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# deps / wheels
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    git \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install --no-cache-dir uv

# copy over dependencies
COPY pyproject.toml uv.lock ./

# venv & install
RUN uv python install 3.11
RUN uv sync --frozen

# COPY rest of repo
COPY . .

# inc
RUN mkdir -p /app/data /app/models

# run entry
CMD ["uv", "run", "python", "main.py"]

# need to run w/ -i -t flags