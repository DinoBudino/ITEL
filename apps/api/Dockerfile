FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/service/.cache/torch

WORKDIR /service

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY apps/api/requirements.txt ./apps/api/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r ./apps/api/requirements.txt

COPY apps/api/app ./app
COPY packages ./packages
COPY models ./models

RUN python -c "import torchaudio; torchaudio.pipelines.HUBERT_BASE.get_model(); print('Preloaded HUBERT_BASE')"

RUN useradd --create-home --uid 10001 itel \
    && chown -R itel:itel /service

USER itel

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
