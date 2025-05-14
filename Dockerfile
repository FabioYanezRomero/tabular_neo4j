FROM ubuntu:22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variable for LM Studio connection
ENV LMSTUDIO_HOST=host.docker.internal
ENV LMSTUDIO_PORT=1234

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

# Create entrypoint script to set up host.docker.internal for Linux hosts
RUN echo '#!/bin/bash\n\
if ! grep -q host.docker.internal /etc/hosts; then\n\
    # Add host.docker.internal to /etc/hosts\n\
    HOST_IP=$(ip route | grep default | awk "{print \$3}")\n\
    echo "$HOST_IP host.docker.internal" >> /etc/hosts\n\
fi\n\
exec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Set the default command to keep container running
CMD ["tail", "-f", "/dev/null"]
