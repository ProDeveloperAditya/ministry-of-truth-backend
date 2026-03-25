# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies for OpenCV and C2PATool
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (no-cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Download and Setup C2PATool (Linux Binary)
RUN mkdir -p /tmp/c2pa && \
    curl -L https://github.com/contentauth/c2pa-rs/releases/download/c2patool-v0.26.33/c2patool-v0.26.33-x86_64-unknown-linux-gnu.tar.gz -o /tmp/c2pa/c2pa.tar.gz && \
    cd /tmp/c2pa && tar -xzf c2pa.tar.gz && \
    ls -laR /tmp/c2pa/ && \
    find /tmp/c2pa -name "c2patool" -type f -exec cp {} /app/c2patool \; && \
    chmod 755 /app/c2patool && \
    ls -la /app/c2patool && \
    /app/c2patool --version && \
    rm -rf /tmp/c2pa


# Environment variables
ENV PORT 8080
ENV PYTHONUNBUFFERED 1

# Run the application using Gunicorn (Production standard)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app


