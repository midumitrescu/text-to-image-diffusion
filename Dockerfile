FROM python:3.10-slim AS build

WORKDIR /app

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

COPY pyproject.toml poetry.lock ./

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    apt-get clean && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false && \
    poetry config cache-dir /var/cache/pypoetry && \
    poetry install --no-root --no-interaction --no-ansi && \
    rm -rf /var/cache/pypoetry/*

FROM python:3.10-slim AS production

WORKDIR /app

COPY --from=build /app /app

#COPY models ./models/
COPY src ./src/

RUN apt-get purge -y --auto-remove build-essential curl && \
    rm -rf /var/lib/apt/lists/*

CMD ["poetry", "run", "python", "-m", "vae.api", "--host", "0.0.0.0", "--port", "8000"]
