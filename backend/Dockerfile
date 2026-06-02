# Use the official Python 3.11 slim image to keep the container lightweight
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, including ffmpeg for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend source code into the container
COPY . .

# Expose port 10000 (Render's default port)
EXPOSE 10000

# Start FastAPI using Uvicorn (uses shell syntax to read Render's PORT environment variable)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
