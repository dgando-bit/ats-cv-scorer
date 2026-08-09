.PHONY: help build up down restart logs logs-backend logs-frontend ps clean prune shell-backend shell-frontend

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

build: ## Build les images Docker (backend + frontend)
	docker compose build

up: ## Démarre les services (build si nécessaire)
	docker compose up --build

up-d: ## Démarre les services en arrière-plan
	docker compose up --build -d

down: ## Arrête et supprime les conteneurs
	docker compose down

restart: down up-d ## Redémarre les services en arrière-plan

logs: ## Affiche les logs de tous les services (suivi en direct)
	docker compose logs -f

logs-backend: ## Affiche les logs du backend uniquement
	docker compose logs -f backend

logs-frontend: ## Affiche les logs du frontend uniquement
	docker compose logs -f frontend

ps: ## Liste l'état des conteneurs du projet
	docker compose ps

shell-backend: ## Ouvre un shell dans le conteneur backend
	docker compose exec backend /bin/bash

shell-frontend: ## Ouvre un shell dans le conteneur frontend
	docker compose exec frontend /bin/sh

clean: ## Supprime les conteneurs et volumes du projet (⚠️ efface le cache des modèles NLP)
	docker compose down -v

prune: ## Nettoie les images Docker inutilisées sur la machine
	docker image prune -f
