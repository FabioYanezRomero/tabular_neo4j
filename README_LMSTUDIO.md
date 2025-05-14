# Using LMStudio with Tabular to Neo4j Converter

This guide explains how to use the Tabular to Neo4j converter with LMStudio integration.

## Prerequisites

1. [LMStudio](https://lmstudio.ai/) installed and running on your local machine
2. Docker and Docker Compose installed (if running in Docker)
3. Python 3.8+ (if running locally)

## Setting Up LMStudio

1. Open LMStudio on your local machine
2. Go to the "Local Server" tab
3. Select a model to use (any model that supports text completion)
4. Click "Start Server"
5. Make sure the server is running on port 1234 (default)

## Running with Docker

The repository includes Docker configuration to connect to LMStudio running on your host machine. You can use either Docker Compose directly or the provided Makefile targets.

### Using the Makefile (Recommended)

The repository includes a Makefile with targets for Docker Compose operations:

1. Build the Docker image:

```bash
make compose-build
```

2. Start the container:

```bash
make compose-up
```

3. Check if LMStudio is reachable from the container:

```bash
make check-lmstudio
```

4. Run the analysis with LMStudio integration:

```bash
make run-lmstudio CSV_FILE=/app/Tabular_to_Neo4j/sample_data/csv/your_file.csv
```

5. Enter the container shell:

```bash
make compose-shell
```

6. Stop and remove the container:

```bash
make compose-down
```

### Using Docker Compose Directly

Alternatively, you can use Docker Compose commands directly:

1. Build and run the Docker container:

```bash
docker-compose up -d
```

2. Enter the container:

```bash
docker-compose exec tabular-neo4j bash
```

3. Run the check script to verify LMStudio connection:

```bash
python -m Tabular_to_Neo4j.utils.check_lmstudio
```

4. Run the analysis with LMStudio integration:

```bash
python -m Tabular_to_Neo4j.run_with_lmstudio /app/Tabular_to_Neo4j/sample_data/csv/your_file.csv
```

## Running Locally

If you're running the application locally (not in Docker):

1. Install the required dependencies:

```bash
pip install -e .
```

2. Run the check script to verify LMStudio connection:

```bash
python -m Tabular_to_Neo4j.utils.check_lmstudio --host localhost
```

3. Run the analysis with LMStudio integration:

```bash
python -m Tabular_to_Neo4j.run_with_lmstudio --lmstudio-host localhost /path/to/your/csv/file.csv
```

## Command-Line Options

The `run_with_lmstudio.py` script accepts the following command-line options:

- `csv_file`: Path to the CSV file to analyze (required)
- `--output`, `-o`: Path to save the results
- `--verbose`, `-v`: Print verbose output
- `--lmstudio-host`: LMStudio host address (default: from environment or host.docker.internal)
- `--lmstudio-port`: LMStudio port number (default: from environment or 1234)
- `--retries`: Number of connection retries (default: 3)

## Troubleshooting

### Connection Issues

If you're having trouble connecting to LMStudio:

1. Make sure LMStudio is running and the server is started
2. Check that the server is listening on the expected port (default: 1234)
3. If running in Docker:
   - Make sure the `host.docker.internal` DNS name is properly set up
   - Try using the host's actual IP address instead of `host.docker.internal`

### Docker Network Issues

If the Docker container can't reach the host:

1. Try using the host network mode:

```bash
docker run --network host ...
```

2. Or specify the host IP explicitly:

```bash
docker run -e LMSTUDIO_HOST=<your_host_ip> ...
```

## Directory Structure

The repository uses the following directory structure for data files:

1. CSV files are stored in `/sample_data/csv/` directory
2. Metadata files are stored in `/sample_data/metadata/` directory with the same filename but .json extension
3. Metadata files contain language information, column descriptions, and other metadata about the CSV files

## How It Works

The integration with LMStudio works by:

1. Checking if LMStudio is available at the specified host and port
2. Using the LMStudio API to get available models
3. Sending prompts to LMStudio for processing
4. Parsing the responses and integrating them into the Tabular to Neo4j workflow

The system falls back to the default LLM provider if LMStudio is not available.
