.PHONY: clean train predict all fresh help

help:
	@echo "Available commands:"
	@echo "  make clean     - Clean up all outputs"
	@echo "  make train     - Train models only"
	@echo "  make predict   - Make predictions only"
	@echo "  make all       - Run full pipeline (train + predict)"
	@echo "  make fresh     - Clean then run full pipeline"
	@echo "  make verify    - Verify project structure"

clean:
	@echo "Cleaning up project..."
	python scripts/cleanup.py

train:
	@echo "Training models..."
	python main.py --mode train

predict:
	@echo "Making predictions..."
	python main.py --mode predict

all:
	@echo "Running full pipeline..."
	python main.py --mode all

fresh:clean all
	@echo "Running fresh pipeline..."


verify:
	@echo "Verifying project structure..."
	python scripts/cleanup.py --verify-only

# ================================
# DOCKER COMMANDS
# ================================

.PHONY: docker-build docker-run docker-dev docker-jupyter docker-clean

# Build Docker image
docker-build:
	docker build -t ride-hailing-ml .

# Run Docker container
docker-run:
	docker run --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/artifacts:/app/artifacts \
		-v $(PWD)/logs:/app/logs \
		ride-hailing-ml

# Development environment
docker-dev:
	docker-compose -f docker-compose.dev.yml up

# Jupyter notebook
docker-jupyter:
	docker-compose up jupyter

# Clean Docker
docker-clean:
	docker-compose down -v
	docker system prune -f