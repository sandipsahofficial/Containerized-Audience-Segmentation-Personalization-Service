FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

RUN mkdir -p /app/data /app/models && \
    chown -R appuser:appuser /app

USER appuser

CMD ["python", "train.py"]
