# Runs on any free-tier container host: Hugging Face Spaces, Render, Railway, Fly.
#
# HF Spaces runs the container as UID 1000, so everything is owned by that user
# and nothing is written outside its home. Render/Railway run as root and are
# unaffected by the extra user.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user static ./static

# Leads persist to a JSON file here. On HF and other ephemeral hosts this resets
# on restart; the store degrades to in-memory if the path is not writable.
RUN mkdir -p $HOME/app/data

# HF Spaces expects 7860; Render/Railway inject $PORT.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
