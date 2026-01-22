# Multi-stage Dockerfile for text-to-image difussion generation per HTTP API
FROM python:3.10-slim

WORKDIR /app
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app
COPY models src pyproject.toml poetry.lock ./

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    apt-get clean && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    poetry install

CMD ["poetry", "run", "python", "-m", "vae.api", "--host", "0.0.0.0", "--port", "8000"]