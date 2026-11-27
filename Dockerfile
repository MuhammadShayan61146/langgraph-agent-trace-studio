FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

RUN useradd -m -u 1000 user
WORKDIR /app

RUN pip install --no-cache-dir python-dotenv

COPY --chown=user src ./src

USER user

ENV AGENTIC_VIZ_HOST=0.0.0.0 \
    AGENTIC_VIZ_PORT=7860 \
    AGENTIC_VIZ_DATA_DIR=/tmp/agentic_viz \
    AGENTIC_VIZ_PACE=0.4

EXPOSE 7860
CMD ["python", "-m", "agentic_research.viz"]
