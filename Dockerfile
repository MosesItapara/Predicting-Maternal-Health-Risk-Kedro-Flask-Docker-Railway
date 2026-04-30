FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source code
COPY pyproject.toml ./
COPY src ./src
COPY conf ./conf
COPY data ./data
COPY app.py ./app.py
COPY templates ./templates

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -e .

# Runtime env settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose UI port
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]