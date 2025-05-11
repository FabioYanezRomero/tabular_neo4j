.PHONY: build run stop shell clean help

# Docker image and container names
IMAGE_NAME = tabular-neo4j
CONTAINER_NAME = tabular-neo4j-container

# Default target
help:
	@echo "Available targets:"
	@echo "  build  - Build the Docker image"
	@echo "  run    - Run the container in detached mode"
	@echo "  shell  - Open a shell in the running container"
	@echo "  stop   - Stop the running container"
	@echo "  clean  - Remove the container and image"

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
