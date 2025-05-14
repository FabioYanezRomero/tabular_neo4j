.PHONY: build run stop shell clean help compose-build compose-up compose-down compose-shell check-lmstudio run-lmstudio

# Docker image and container names
IMAGE_NAME = tabular-neo4j
CONTAINER_NAME = tabular-neo4j-container

# Default target
help:
	@echo "Available targets:"
	@echo "  build        - Build the Docker image"
	@echo "  run          - Run the container in detached mode"
	@echo "  shell        - Open a shell in the running container"
	@echo "  stop         - Stop the running container"
	@echo "  clean        - Remove the container and image"
	@echo "  compose-build - Build the Docker image using docker-compose"
	@echo "  compose-up   - Start the container using docker-compose"
	@echo "  compose-down - Stop and remove the container using docker-compose"
	@echo "  compose-shell - Open a shell in the running docker-compose container"
	@echo "  check-lmstudio - Check if LMStudio is reachable from the container"
	@echo "  run-lmstudio <CSV_FILE> - Run analysis with LMStudio integration"

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the container in detached mode
run:
	docker run -d --name $(CONTAINER_NAME) \
		-v $(PWD):/app \
		-p 7474:7474 -p 7687:7687 \
		$(IMAGE_NAME)
	@echo "Container started. Access shell with 'make shell'"
	@docker ps | grep $(CONTAINER_NAME) || echo "Warning: Container may have stopped. Check logs with 'docker logs $(CONTAINER_NAME)'"

# Open a shell in the running container
shell:
	docker exec -it $(CONTAINER_NAME) bash

# Stop the running container
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Remove the container and image
clean: stop
	docker rmi $(IMAGE_NAME) || true

# Rebuild and restart the container
restart: stop build run

# Docker Compose targets
compose-build:
	docker-compose build

# Start the container using docker-compose
compose-up:
	docker-compose up -d
	@echo "Container started with docker-compose. Access shell with 'make compose-shell'"

# Stop and remove the container using docker-compose
compose-down:
	docker-compose down

# Open a shell in the running docker-compose container
compose-shell:
	docker-compose exec tabular-neo4j bash

# Check if LMStudio is reachable from the container
check-lmstudio:
	docker-compose exec tabular-neo4j python -m Tabular_to_Neo4j.utils.check_lmstudio

# Run analysis with LMStudio integration
run-lmstudio:
	@if [ -z "$(CSV_FILE)" ]; then \
		echo "Error: CSV_FILE parameter is required. Usage: make run-lmstudio CSV_FILE=/path/to/file.csv"; \
		exit 1; \
	fi
	docker-compose exec tabular-neo4j python -m Tabular_to_Neo4j.run_with_lmstudio $(CSV_FILE)

# Rebuild and restart the container with docker-compose
compose-restart: compose-down compose-build compose-up
