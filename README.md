version: '3.8'

# This file orchestrates all the services of the application.
# It defines how each container (frontend, api_gateway, etc.) is built and connected.

services:
  # The Nginx server serves the React frontend and acts as a reverse proxy to the backend.
  frontend:
    build:
      context: ./frontend
    container_name: frontend
    ports:
      - "3000:80" # Maps port 3000 on your PC to port 80 in the container.
    depends_on:
      - api_gateway # Ensures the backend is started before the frontend.
    networks:
      - credtech_network

  # The main backend API that the frontend communicates with.
  api_gateway:
    build: ./services/api_gateway
    container_name: api_gateway
    # The API Gateway is not exposed to the public directly in a production setup.
    # Nginx proxies to it. Port is exposed here for local development API testing.
    ports:
      - "8000:8000"
    volumes:
      - ./services/api_gateway/app:/app # Mounts local code for live reloading during development.
    environment:
      # These variables are passed into the container.
      - SCORING_ENGINE_URL=http://scoring_engine:8002
      - ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY} # Loaded from the .env file
      - NEWS_API_KEY=${NEWS_API_KEY}                   # Loaded from the .env file
    depends_on:
      - scoring_engine
    networks:
      - credtech_network

  # The data ingestion service (can be run as a periodic task).
  # For this project, it runs once on startup as a demonstration.
  data_ingestion:
    build: ./services/data_ingestion
    container_name: data_ingestion
    environment:
      - ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY}
      - NEWS_API_KEY=${NEWS_API_KEY}
    networks:
      - credtech_network

  # The dedicated Machine Learning service for scoring and explanations.
  scoring_engine:
    build: ./services/scoring_engine
    container_name: scoring_engine
    volumes:
      - ./services/scoring_engine/app:/app
    networks:
      - credtech_network

# Defines the virtual network that allows containers to communicate with each other by name.
networks:
  credtech_network:
    driver: bridge