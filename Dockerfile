FROM python:3.9-slim

# -------------------- System deps --------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------- Env safety --------------------
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV OPENCV_DISABLE_DNN_TYPING=1
ENV OPENCV_LOG_LEVEL=ERROR

WORKDIR /app

# -------------------- Python deps --------------------
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# -------------------- App code --------------------
COPY . .

# -------------------- DeepFace cache --------------------
RUN mkdir -p /app/.deepface && chmod -R 777 /app/.deepface
ENV DEEPFACE_HOME=/app/.deepface

# -------------------- Port --------------------
EXPOSE 7860

# -------------------- Run --------------------
# IMPORTANT: workers=1 (YOLO + DeepFace will OOM otherwise)
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--threads", "1", \
     "--timeout", "120"]
