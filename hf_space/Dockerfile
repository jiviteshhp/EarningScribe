FROM python:3.10-slim

# install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all app files
COPY . .

# create directory for chroma db persistence
RUN mkdir -p /app/data/chroma_db

# HF Spaces runs on port 7860
EXPOSE 7860

# start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
