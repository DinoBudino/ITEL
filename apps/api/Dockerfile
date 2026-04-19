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
COPY assets ./assets

RUN python -c "import os; import torchaudio; bundle_name=os.environ.get('ITEL_HUBERT_BUNDLE', 'HUBERT_BASE'); bundle=getattr(torchaudio.pipelines, bundle_name); bundle.get_model(); print(f'Preloaded {bundle_name} into {os.environ.get(\"TORCH_HOME\")}')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
