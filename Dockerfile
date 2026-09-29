FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (build-essential, curl for audio/codecs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "agent.py"]
CMD ["start"]
