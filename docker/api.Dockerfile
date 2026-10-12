# The serving image: the API and the model bundle, on numpy and scipy alone.
# Build (from the project root, with the bundle present: `uv run dvc pull bundle`):
#   docker build -f docker/api.Dockerfile --build-arg MODEL_VERSION=1 -t recsys-api .
# Run with the host's memory limit:
#   docker run --rm --memory=512m -p 8000:8000 --env-file .env recsys-api
FROM python:3.12.8-slim-bookworm AS build

ENV UV_PROJECT_ENVIRONMENT=/opt/venv     UV_COMPILE_BYTECODE=1     UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
# Only the `api` dependency group, at the versions in uv.lock: not the training project
# and none of its dependencies (pandas, implicit, torch, mlflow).
RUN uv sync --frozen --only-group api


FROM python:3.12.8-slim-bookworm

# One BLAS/OpenMP thread: scoring a user is a small sparse product, and every extra thread
# costs memory the 512 MB host doesn't have.
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     OMP_NUM_THREADS=1     OPENBLAS_NUM_THREADS=1     PATH=/opt/venv/bin:$PATH     BUNDLE_DIR=/app/bundle

RUN useradd --create-home --uid 1000 api
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY api/recsys_api ./recsys_api
COPY bundle ./bundle

# The bundle's version in the model registry, reported by /health. It is passed in rather
# than stored in the bundle, which has to hash the same on every build.
ARG MODEL_VERSION=dev
# The commit the image was built from, also reported by /health: a deploy waits for it.
ARG GIT_COMMIT=dev
ENV MODEL_VERSION=$MODEL_VERSION     GIT_COMMIT=$GIT_COMMIT
# Links the package on ghcr.io to the repository.
LABEL org.opencontainers.image.source=https://github.com/webmaster99-stack/movie-recommendation-system

USER api
EXPOSE 8000
# One worker: each would load its own copy of the model. Render sets PORT.
CMD ["sh", "-c", "exec uvicorn recsys_api.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
