.PHONY: build run stop shell clean help compose-build compose-up compose-down compose-shell check-lmstudio

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
	@echo "  run-and-load - Build, start containers, run pipeline, and load Cypher into Neo4j"

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

# Build, up, run pipeline, and load Cypher into Neo4j
default_csv ?= /app/Tabular_to_Neo4j/sample_data/csv/customers.csv
run-and-load:
	docker compose build
	docker compose up -d
	# Wait for Neo4j to be healthy
	@echo "Waiting for Neo4j to be healthy..." && sleep 20
	docker compose exec tabular-neo4j python -m Tabular_to_Neo4j.main $(default_csv) --save-node-outputs
	@echo "Pipeline run complete. Cypher should be loaded into Neo4j. Access Neo4j at http://localhost:7474 (neo4j/password)"
	docker rm $(CONTAINER_NAME) || true

# Remove the container and image
clean: stop
	docker rmi $(IMAGE_NAME) || true

# Rebuild and restart the container
restart: stop build run

# Docker Compose targets
compose-build:
	docker compose build

# Compose without cache
compose-build-no-cache:
	docker compose build --no-cache

# Start the container using docker-compose
compose-up:
	docker compose up -d
	@echo "Container started with docker-compose. Access shell with 'make compose-shell'"

# Stop and remove the container using docker-compose
compose-down:
	docker compose down

# Open a shell in the running docker-compose container
compose-shell:
	docker compose exec tabular-neo4j bash


# Stop and remove the container using docker-compose
compose-down:
	docker compose down

# Open a shell in the running docker-compose container
compose-shell:
	docker compose exec tabular-neo4j bash

# Build the Docker image using docker-compose
compose-build:
	docker compose build

# Start the container using docker-compose
compose-up:
	docker compose up -d
	@echo "Container started with docker-compose. Access shell with 'make compose-shell'"

# Check if LMStudio is reachable from the container
check-lmstudio:
	docker compose exec tabular-neo4j python -m Tabular_to_Neo4j.utils.check_lmstudio
