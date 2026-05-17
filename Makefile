.PHONY: clean train predict all fresh help docker-build docker-run docker-stop streamlit

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
NC=\033[0m # No Color

help:
	@echo "$(GREEN)Ride-Hailing ML System - Available Commands$(NC)"
	@echo "=========================================="
	@echo ""
	@echo "$(YELLOW)Local Development:$(NC)"
	@echo "  make clean       - Clean up all outputs"
	@echo "  make train       - Train models only"
	@echo "  make predict     - Make predictions only"
	@echo "  make all         - Run full pipeline (train + predict)"
	@echo "  make fresh       - Clean then run full pipeline"
	@echo "  make streamlit   - Run Streamlit UI locally"
	@echo "  make verify      - Verify project structure"
	@echo ""
	@echo "$(YELLOW)Docker Commands:$(NC)"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container locally"
	@echo "  make docker-stop    - Stop running container"
	@echo ""
	@echo "$(YELLOW)Streamlit Cloud:$(NC)"
	@echo "  Just push to GitHub and deploy on https://share.streamlit.io"
	@echo ""

# ================================
# LOCAL DEVELOPMENT
# ================================

clean:
	@echo "$(GREEN)Cleaning up project...$(NC)"
	python scripts/cleanup.py

train:
	@echo "$(GREEN)Training models...$(NC)"
	python main.py --mode train

predict:
	@echo "$(GREEN)Making predictions...$(NC)"
	python main.py --mode predict

all:
	@echo "$(GREEN)Running full pipeline...$(NC)"
	python main.py --mode all

fresh: clean all
	@echo "$(GREEN)Fresh pipeline completed!$(NC)"

streamlit:
	@echo "$(GREEN)Starting Streamlit UI locally...$(NC)"
	streamlit run app.py --server.port 8501 --server.address 0.0.0.0

verify:
	@echo "$(GREEN)Verifying project structure...$(NC)"
	python scripts/cleanup.py --verify-only

# ================================
# DOCKER COMMANDS
# ================================

# Variables
IMAGE_NAME = ride-hailing-ml
IMAGE_TAG = latest

docker-build:
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "$(GREEN)Image built: $(IMAGE_NAME):$(IMAGE_TAG)$(NC)"

docker-run:
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run --rm -p 8501:8501 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/artifacts:/app/artifacts \
		-v $(PWD)/logs:/app/logs \
		$(IMAGE_NAME):$(IMAGE_TAG)

docker-stop:
	@echo "$(GREEN)Stopping container...$(NC)"
	docker stop $$(docker ps -q --filter ancestor=$(IMAGE_NAME):$(IMAGE_TAG)) 2>/dev/null || true

# ================================
# QUICK START
# ================================

dev: streamlit
	@echo "$(GREEN)Starting development server...$(NC)"

prod: docker-build docker-run
	@echo "$(GREEN)Starting production container...$(NC)"