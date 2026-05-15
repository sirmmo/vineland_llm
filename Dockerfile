FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
      "httpx>=0.27" "pydantic>=2.0" "pyyaml>=6.0" \
      "tenacity>=8.0" "jsonschema>=4.0" "tqdm>=4.66"

COPY vineland_runner ./vineland_runner
COPY items ./items
COPY configs ./configs
COPY scripts ./scripts
COPY README.md ./

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["vineland-runner"]
CMD ["run", "--config", "configs/pilot.yaml"]
