FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /service

RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir .

COPY scripts ./scripts
RUN mkdir -p /service/data/images /service/models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
