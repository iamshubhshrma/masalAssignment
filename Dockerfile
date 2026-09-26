# Runs on any free-tier container host: Render, Hugging Face Spaces, Fly, Railway.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

# Leads persist to a JSON file; on ephemeral hosts this resets on redeploy.
RUN mkdir -p /app/data

# Hosts inject $PORT (Render, Railway); 7860 is the Hugging Face Spaces default.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
