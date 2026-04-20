FROM python:3.12-slim

WORKDIR /app

# Install runner package first (from sibling directory)
COPY ../vineland_runner /app/vineland_runner
COPY ../pyproject.toml /app/runner_pyproject.toml
RUN pip install --no-cache-dir -e /app/vineland_runner/../

# Install API package
COPY vineland_api/ /app/vineland_api/
COPY pyproject.toml /app/vineland_api/pyproject.toml
RUN pip install --no-cache-dir -e /app/vineland_api/

# Item and config files (mount in production)
COPY ../items /app/items
COPY ../configs /app/configs

ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "vineland_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
