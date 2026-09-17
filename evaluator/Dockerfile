FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_URL=http://api:5000

RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluate.py test_cases.json ./

RUN mkdir -p /app/models /app/results && \
    chown -R appuser:appuser /app

USER appuser

CMD ["python", "evaluate.py"]
