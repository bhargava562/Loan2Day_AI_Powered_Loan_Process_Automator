# Loan2Day Docker Management Makefile

.PHONY: help build up down logs clean dev prod test

# Default target
help:
	@echo "Loan2Day Docker Management Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development environment"
	@echo "  make dev-build    - Build and start development environment"
	@echo "  make dev-logs     - View development logs"
	@echo "  make dev-down     - Stop development environment"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start production environment"
	@echo "  make prod-build   - Build and start production environment"
	@echo "  make prod-logs    - View production logs"
	@echo "  make prod-down    - Stop production environment"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run tests in containers"
	@echo "  make test-build   - Build test environment and run tests"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Clean up containers, images, and volumes"
	@echo "  make clean-all    - Clean everything including volumes"
	@echo "  make logs         - View all service logs"
	@echo "  make ps           - Show running containers"
	@echo "  make health       - Check service health"

# Development commands
dev:
	docker-compose up -d

dev-build:
	docker-compose up -d --build

dev-logs:
	docker-compose logs -f

dev-down:
	docker-compose down

# Production commands
prod:
	docker-compose -f docker-compose.yml --profile production up -d

prod-build:
	docker-compose -f docker-compose.yml --profile production up -d --build

prod-logs:
	docker-compose -f docker-compose.yml --profile production logs -f

prod-down:
	docker-compose -f docker-compose.yml --profile production down

# Testing commands
test:
	docker-compose exec loan2day-api pytest tests/ -v

test-build:
	docker-compose up -d --build
	docker-compose exec loan2day-api pytest tests/ -v

# Maintenance commands
logs:
	docker-compose logs -f

ps:
	docker-compose ps

health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | jq '.' || echo "API health check failed"
	@curl -s http://localhost:3000/health || echo "Frontend health check failed"

clean:
	docker-compose down --rmi local --volumes --remove-orphans

clean-all:
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f

# Database commands
db-init:
	docker-compose exec postgres psql -U loan2day -d loan2day -f /docker-entrypoint-initdb.d/init-db.sql

db-backup:
	docker-compose exec postgres pg_dump -U loan2day loan2day > backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore:
	@read -p "Enter backup file name: " backup_file; \
	docker-compose exec -T postgres psql -U loan2day -d loan2day < $$backup_file

# Redis commands
redis-cli:
	docker-compose exec redis redis-cli

redis-flush:
	docker-compose exec redis redis-cli FLUSHALL

# Kafka commands
kafka-topics:
	docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list

kafka-create-topic:
	@read -p "Enter topic name: " topic_name; \
	docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --create --topic $$topic_name --partitions 3 --replication-factor 1

# Monitoring commands
monitor:
	@echo "Service Status:"
	@docker-compose ps
	@echo ""
	@echo "Resource Usage:"
	@docker stats --no-stream

# Setup commands
setup: setup-env dev-build
	@echo "Loan2Day development environment is ready!"
	@echo "API: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "API Docs: http://localhost:8000/docs"

setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template. Please review and update as needed."; \
	fi

# Install commands
install-deps:
	pip install -r requirements.txt
	cd frontend && npm install

# Linting and formatting
lint:
	docker-compose exec loan2day-api black app/ --check
	docker-compose exec loan2day-api flake8 app/
	cd frontend && npm run lint

format:
	docker-compose exec loan2day-api black app/
	cd frontend && npm run format

# Security scanning
security-scan:
	docker-compose exec loan2day-api safety check
	cd frontend && npm audit

# Performance testing
perf-test:
	@echo "Running performance tests..."
	@curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Backup and restore
backup: db-backup
	@echo "Creating full backup..."
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@docker-compose exec redis redis-cli --rdb backups/$(shell date +%Y%m%d_%H%M%S)/redis.rdb
	@echo "Backup completed"

# Update commands
update:
	git pull
	docker-compose pull
	docker-compose up -d --build

# SSL setup (for production)
ssl-setup:
	@mkdir -p nginx/ssl
	@openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
		-keyout nginx/ssl/key.pem \
		-out nginx/ssl/cert.pem \
		-subj "/C=IN/ST=State/L=City/O=Loan2Day/CN=localhost"
	@echo "Self-signed SSL certificates created for development"