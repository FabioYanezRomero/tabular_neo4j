FROM ubuntu:22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and setup files
COPY Tabular_to_Neo4j/requirements.txt .
COPY setup.py .
COPY setup_env.sh .

# Copy the application code
COPY Tabular_to_Neo4j/ /app/Tabular_to_Neo4j/

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Set the default command to keep container running
CMD ["tail", "-f", "/dev/null"]
