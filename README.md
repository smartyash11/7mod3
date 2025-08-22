# ======================================================================================
# FILE: README.md
# ======================================================================================
# 7mod3 - Real-Time Explainable Credit Intelligence Platform

**A Hackathon Submission by Yash Kamra & Tanya Goyanka**

## Our Aim

Our aim is to revolutionize credit risk analysis by replacing the slow, opaque, and lagging methodologies of traditional credit rating agencies. We leverage Artificial Intelligence and high-frequency, alternative data to build a real-time, transparent, and evidence-backed credit intelligence platform, empowering investors and analysts to make more informed and timely decisions.

## The Problem: Outdated Credit Ratings

In today's fast-paced global credit markets, decisions involving billions of dollars are still guided by traditional credit rating agencies. These legacy systems suffer from critical flaws:
* **Infrequent Updates:** Ratings are updated too slowly to reflect real-world events in a timely manner.
* **Opaque Methodologies:** The reasoning behind a given rating is often a "black box," making it difficult for investors to trust and verify.
* **Lagging Indicators:** They are often reactive, only changing a rating after a significant event has already impacted the market.

This inefficiency creates significant "mispricing opportunities" where the true risk is different from what the market perceives.

## Our Solution: A Real-Time Explainable Platform

We have built a **Real-Time Explainable Credit Intelligence Platform** that directly addresses these challenges. Our product continuously ingests a wide array of public data, from financial filings to real-time news, to produce dynamic creditworthiness scores that are both faster and fully transparent.

### Product Offerings & Core Functions

Our platform provides analysts and investors with a powerful, interactive web dashboard to:

* **Access Real-Time Credit Scores:** We generate up-to-the-minute creditworthiness scores for various companies ("issuers") that react faster to market events than traditional ratings.
* **Understand the "Why":** For every score, we provide clear, feature-level explanations, plain-language summaries, and trend insights. We show exactly which factors—from financial ratios to recent news—influenced the score.
* **Track Score Trends:** Through interactive visualizations, users can monitor a company's creditworthiness over time, identifying risks and opportunities as they emerge.
* **Integrate Unstructured Data:** Our system detects and interprets events from unstructured sources like news headlines and factors them directly into the credit score and its explanation, providing an early warning system for events like debt restructuring or warnings of declining demand.

### The Competitive Landscape

Our primary competitors are the traditional credit rating agencies (e.g., Moody's, S&P, Fitch).

| Feature                | Traditional Agencies                                      | 7mod3 Platform                                                                                             |
| ---------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Speed** | Ratings are updated infrequently and lag behind events. | Scores are generated in near-real-time and react faster to new information.                   |
| **Transparency** | Methodologies are opaque, creating a "black box".     | Fully explainable with feature contribution breakdowns and evidence-backed reasoning.        |
| **Data Sources** | Primarily rely on official company filings and reports.        | Fuses financial data with high-frequency, alternative, and unstructured data like real-time news. |
| **Adaptability** | Static models that are slow to change.                    | An adaptive scoring engine capable of frequent retraining to incorporate new data patterns. |

## Our Tech Stack

We chose a modern, scalable tech stack to meet the demands of a real-time data platform:

* **Frontend:** React.js, Chart.js
* **Backend:** Python, FastAPI
* **Machine Learning:** XGBoost, SHAP, Scikit-learn
* **Deployment:** Docker, Docker Compose, Nginx

## Project Structure Explained

The project is organized into a clean, microservices-based architecture to ensure scalability and separation of concerns.

```
credtech-7mod3/
├── .env                  # Stores private API keys securely, not committed to Git.
├── docker-compose.yml      # Orchestrates all Docker containers for easy setup.
├── README.md               # You are here! Project explanation and guide.
│
├── services/               # Contains all backend microservices.
│   │
│   ├── api_gateway/        # Handles all incoming requests from the frontend and routes them.
│   │   ├── app/
│   │   │   ├── main.py     # FastAPI application entry point for the gateway.
│   │   │   └── services.py # Logic for communicating with other backend services.
│   │   ├── Dockerfile      # Instructions to build the gateway's Docker image.
│   │   └── requirements.txt# Python dependencies for the gateway.
│   │
│   ├── data_ingestion/     # Service for fetching data from external APIs.
│   │   ├── app/
│   │   │   ├── main.py     # Main script demonstrating the ingestion process.
│   │   │   └── fetchers.py # Functions to fetch from Alpha Vantage and NewsAPI.
│   │   └── ...
│   │
│   └── scoring_engine/     # Dedicated ML service for scoring and explanations.
│       ├── app/
│       │   ├── main.py     # FastAPI application for serving the ML model.
│       │   ├── model/      # Stores the trained XGBoost model files.
│       │   └── ml_utils.py # Core logic for prediction and SHAP explanation.
│       ├── train.py          # Script to train the ML model.
│       └── ...
│
└── frontend/               # Contains the entire React user interface.
    │
    ├── public/             # Static assets like the main HTML page.
    ├── src/
    │   ├── components/     # Reusable React components (charts, tabs, watchlist).
    │   ├── App.css         # Main styling for the TradingView-inspired theme.
    │   ├── App.js          # Main application component and layout logic.
    │   └── index.js        # Entry point for the React application.
    ├── Dockerfile          # Builds the React app and sets up an Nginx server.
    ├── nginx.conf          # Nginx configuration for serving the app and proxying API calls.
    └── package.json        # Frontend dependencies (React, Chart.js, etc.).
```

## Step-by-Step Deployment Instructions

### A. Local Deployment (for Development)

**Prerequisites:**
* Docker and Docker Compose installed on your PC.
* API keys from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) and [NewsAPI](https://newsapi.org/register).

**Step 1: Clone the Repository**
```bash
git clone <your-repo-url>
cd credtech-7mod3
```

**Step 2: Create Environment File**
Create a file named `.env` in the project's root directory. This file will securely store your API keys.
```
# .env file
# Paste your actual keys here
NEWS_API_KEY=YOUR_NEWS_API_KEY_HERE
ALPHA_VANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_KEY_HERE
```
*Note: `docker-compose` automatically loads variables from a `.env` file.*

**Step 3: Train the Initial ML Model**
The `scoring_engine` needs a model file to start. Run the training script locally once. This script creates the model files inside `services/scoring_engine/app/model/`.
```bash
# On Mac/Linux
cd services/scoring_engine
pip3 install -r requirements.txt
python3 train.py
cd ../..

# On Windows
cd services/scoring_engine
pip install -r requirements.txt
python train.py
cd ../..
```

**Step 4: Build and Run the Application**
From the root directory (`credtech-7mod3/`), run the main Docker Compose command. This will build the images for each service and start the containers.
```bash
docker compose up --build -d
```
The `-d` flag runs the containers in detached mode (in the background).

**Step 5: Access the Platform**
* **Frontend Dashboard**: Open your browser and go to `http://localhost:3000`
* **API Gateway Docs**: To see the backend API documentation, go to `http://localhost:8000/docs`

### B. Cloud Deployment (for Public Demo)

**Prerequisites:** A cloud server (e.g., AWS EC2, DigitalOcean Droplet) with Docker and Docker Compose installed.

1.  **SSH into your server** and clone your Git repository.
2.  **Create the `.env` file** on the server with your production API keys, just like you did locally.
3.  **Run the `train.py` script** on the server to generate the model files.
4.  **Run Docker Compose:** `docker compose up --build -d`.
5.  **Configure Firewall/Security Groups:** In your cloud provider's dashboard, ensure port `3000` (or port `80` if you adjust the `docker-compose.yml` file) is open to public traffic.
6.  **Access your application** via your server's public IP address: `http://<your_server_ip>:3000`.

# ======================================================================================
# FILE: docker-compose.yml
# ======================================================================================
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

# ======================================================================================
# FILE: services/api_gateway/Dockerfile
# ======================================================================================
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ======================================================================================
# FILE: services/api_gateway/requirements.txt
# ======================================================================================
fastapi
uvicorn[standard]
requests
pydantic

# ======================================================================================
# FILE: services/api_gateway/app/__init__.py
# ======================================================================================
# This file is intentionally empty.

# ======================================================================================
# FILE: services/api_gateway/app/main.py
# ======================================================================================
from fastapi import FastAPI, HTTPException
from . import services
from pydantic import BaseModel
from typing import List, Dict, Any
import requests

app = FastAPI(
    title="7mod3 Credit Intelligence API",
    description="Made by Yash Kamra & Tanya Goyanka",
    version="1.0.0"
)

# Pydantic models define the structure of the API response for type safety and documentation.
class ScoreResponse(BaseModel):
    company: str
    symbol: str
    creditScore: int
    summary: str
    featureContributions: Dict[str, float]
    keyFinancials: Dict[str, Any]
    recentNews: List[Dict[str, str]]

@app.get("/api/score/{company_ticker}", response_model=ScoreResponse)
async def get_company_score(company_ticker: str):
    """
    Retrieves, processes, and scores data for a given company ticker.
    This is the primary endpoint for the frontend dashboard.
    """
    try:
        # 1. Fetch structured and unstructured data using the services module
        structured_data = services.get_structured_data(company_ticker)
        unstructured_data = services.get_unstructured_data(company_ticker)
        
        # 2. Prepare the feature set for the model, handling potential missing values with defaults.
        features = {
            "debtToEquity": float(structured_data.get('DebtToEquityRatio', 0.5)),
            "returnOnEquity": float(structured_data.get('ReturnOnEquityTTM', 0.1)),
            "news_sentiment_score": float(unstructured_data.get('sentiment_score', 0.0))
        }
        
        # 3. Get score and explanation from the internal ML service
        score_response = services.get_score_from_engine(features)

        # 4. Assemble the final response object that matches the ScoreResponse model
        return {
            "company": structured_data.get('Name', company_ticker.upper()),
            "symbol": company_ticker.upper(),
            "creditScore": score_response.get('creditScore'),
            "summary": score_response.get('summary'),
            "featureContributions": score_response.get('featureContributions'),
            "keyFinancials": structured_data,
            "recentNews": unstructured_data.get('headlines')
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch data from external APIs for ticker '{company_ticker}'. The ticker might be invalid or the API service is down.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# ======================================================================================
# FILE: services/api_gateway/app/services.py
# ======================================================================================
import os
import requests

# Service URLs and API Keys are loaded from environment variables for security and flexibility.
SCORING_ENGINE_URL = os.getenv("SCORING_ENGINE_URL")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_structured_data(ticker: str) -> dict:
    """
    Fetches company overview data (structured financial data) from Alpha Vantage.
    """
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == 'YOUR_ALPHA_VANTAGE_KEY_HERE':
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set or is a placeholder.")
    
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    
    if not data or "Note" in data or "Name" not in data:
        raise ValueError(f"Invalid or no structured data found for ticker '{ticker}'. The API limit might have been reached.")
        
    return data

def get_unstructured_data(ticker: str) -> dict:
    """
    Fetches news headlines from NewsAPI and calculates a simple sentiment score.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == 'YOUR_NEWS_API_KEY_HERE':
        raise ValueError("NEWS_API_KEY environment variable not set or is a placeholder.")
        
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
    
    r = requests.get(url)
    r.raise_for_status()
    articles = r.json().get("articles", [])
    
    headlines = [{"title": a["title"], "source": a["source"]["name"]} for a in articles]
    
    # Simple sentiment logic for demonstration: count positive/negative keywords
    sentiment_score = 0
    positive_words = ['strong', 'growth', 'profit', 'beats', 'up', 'launches', 'record', 'high', 'advances']
    negative_words = ['weak', 'loss', 'misses', 'down', 'investigation', 'concerns', 'fall', 'cuts', 'slumps']
    
    for item in headlines:
        title = item['title'].lower()
        for word in positive_words:
            if word in title:
                sentiment_score += 1
        for word in negative_words:
            if word in title:
                sentiment_score -= 1

    return {"headlines": headlines, "sentiment_score": float(sentiment_score)}
    
def get_score_from_engine(features: dict) -> dict:
    """
    Calls the internal scoring_engine microservice to get a score and explanation.
    """
    response = requests.post(f"{SCORING_ENGINE_URL}/score", json=features)
    response.raise_for_status()
    return response.json()

# ======================================================================================
# FILE: services/data_ingestion/Dockerfile
# ======================================================================================
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app /app
CMD ["python", "main.py"]

# ======================================================================================
# FILE: services/data_ingestion/requirements.txt
# ======================================================================================
requests

# ======================================================================================
# FILE: services/data_ingestion/app/__init__.py
# ======================================================================================
# This file is intentionally empty.

# ======================================================================================
# FILE: services/data_ingestion/app/main.py
# ======================================================================================
from fetchers import get_structured_data, get_unstructured_data
import time

def main():
    """
    This script simulates an ingestion cycle. In a real MLOps system, this would
    be a scheduled job that runs periodically to update a database.
    """
    print("--- [7mod3] Data Ingestion Service Cycle ---")
    
    # Example ticker to test the ingestion process
    ticker = "MSFT" 
    
    try:
        # Fetch structured financial data from Alpha Vantage
        structured_data = get_structured_data(ticker)
        print(f"Successfully fetched structured data for {ticker}. Market Cap: {structured_data.get('MarketCapitalization')}")
        
        # A small delay to be respectful of API rate limits
        time.sleep(1) 
        
        # Fetch unstructured news data from NewsAPI
        unstructured_data = get_unstructured_data(ticker)
        print(f"Successfully fetched {len(unstructured_data.get('articles', []))} news articles for {ticker}.")
        
    except Exception as e:
        print(f"An error occurred during the ingestion cycle for {ticker}: {e}")

    print("--- Ingestion cycle complete. In a real system, this data would now be written to a database. ---")

if __name__ == "__main__":
    main()

# ======================================================================================
# FILE: services/data_ingestion/app/fetchers.py
# ======================================================================================
import os
import requests

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_structured_data(ticker: str) -> dict:
    """Fetches company overview data (structured financial data) from Alpha Vantage."""
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set.")
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    if not data or "Note" in data:
        raise ValueError(f"No structured data found for ticker '{ticker}'.")
    return data

def get_unstructured_data(ticker: str) -> dict:
    """Fetches recent news articles (unstructured data) from NewsAPI."""
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY environment variable not set.")
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

# ======================================================================================
# FILE: services/scoring_engine/Dockerfile
# ======================================================================================
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

# ======================================================================================
# FILE: services/scoring_engine/requirements.txt
# ======================================================================================
fastapi
uvicorn[standard]
xgboost
scikit-learn
pandas
shap
pydantic

# ======================================================================================
# FILE: services/scoring_engine/train.py
# ======================================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os

print("--- Starting ML Model Training Script ---")

# Define the directory where the trained model and features will be saved.
MODEL_DIR = "app/model"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Step 1: Generate Synthetic Training Data ---
# In a real-world project, you would load historical data from a database here.
print("Generating synthetic training data...")
np.random.seed(42) # Use a seed for reproducibility
data_size = 500
X_train = pd.DataFrame({
    # Feature 1: A financial ratio, typically between 0.1 and 3.0
    'debtToEquity': np.random.uniform(0.1, 3.0, data_size),
    # Feature 2: A profitability metric, can be negative or positive
    'returnOnEquity': np.random.uniform(-0.5, 0.5, data_size),
    # Feature 3: A score from our unstructured data analysis
    'news_sentiment_score': np.random.uniform(-5, 5, data_size)
})

# Create a synthetic target variable (our "true" credit score, from 0 to 1).
# The formula defines the relationships: High ROE & positive sentiment increase the score,
# while high debt decreases it.
y_train = (0.4 * X_train['returnOnEquity']) - (0.2 * X_train['debtToEquity']) + (0.1 * X_train['news_sentiment_score'])
# Normalize the score to ensure it's always between 0 and 1.
y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())

print(f"Generated {data_size} training samples.")

# --- Step 2: Train the XGBoost Model ---
print("Training XGBoost Regressor model...")
# Initialize the model with chosen parameters.
model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,           # Number of boosting rounds
    learning_rate=0.1,          # Step size shrinkage
    max_depth=3,                # Maximum depth of a tree
    seed=42
)
# Fit the model to our training data.
model.fit(X_train, y_train)
print("Model training complete.")

# --- Step 3: Save the Model and Feature List ---
# These artifacts are required by the scoring service (ml_utils.py).
model_path = os.path.join(MODEL_DIR, "credit_score_model.json")
features_path = os.path.join(MODEL_DIR, "model_features.json")

# Save the trained model.
model.save_model(model_path)

# Save the list of feature names in the correct order.
with open(features_path, "w") as f:
    json.dump(list(X_train.columns), f)
    
print(f"Model and features saved successfully to '{MODEL_DIR}/'")
print("--- Training Script Finished ---")

# ======================================================================================
# FILE: services/scoring_engine/app/__init__.py
# ======================================================================================
# This file is intentionally empty.

# ======================================================================================
# FILE: services/scoring_engine/app/main.py
# ======================================================================================
from fastapi import FastAPI
from pydantic import BaseModel, Field
from . import ml_utils

# Pydantic model defines the expected input data structure,
# ensuring data validation and clear API documentation.
class CompanyFeatures(BaseModel):
    debtToEquity: float = Field(..., example=0.58)
    returnOnEquity: float = Field(..., example=0.15)
    news_sentiment_score: float = Field(..., example=0.7)

app = FastAPI(
    title="7mod3 Scoring Engine",
    description="A dedicated microservice for ML-based credit scoring and explainability."
)

@app.get("/")
def read_root():
    """A simple endpoint to confirm that the service is running."""
    return {"status": "Scoring Engine is online."}

@app.post("/score")
def get_score_and_explanation(features: CompanyFeatures):
    """
    Accepts company features, generates a credit score, and provides a SHAP-based explanation.
    This is the core endpoint of this service.
    """
    # Convert the Pydantic model to a dictionary
    input_data = features.dict()
    
    # Call the core logic from ml_utils to get the prediction and explanation
    result = ml_utils.predict_and_explain(input_data)
    
    return result

# ======================================================================================
# FILE: services/scoring_engine/app/ml_utils.py
# ======================================================================================
import xgboost as xgb
import shap
import json
import pandas as pd
import os

# Define the paths to the model and feature files.
# These files are created by the train.py script and must be present for this service to work.
MODEL_PATH = os.path.join("model", "credit_score_model.json")
FEATURES_PATH = os.path.join("model", "model_features.json")

# --- Load Model and Explainer on Service Startup ---
# This is a crucial optimization. The model and explainer are loaded into memory once
# when the container starts. This makes prediction requests much faster as the files
# don't need to be read from disk for every single API call.

# Load the trained XGBoost model from the file generated by train.py
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

# Load the list of feature names. It's critical that the input data
# for prediction uses the exact same feature order as the training data.
with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)

# Create a SHAP (SHapley Additive exPlanations) explainer object.
# This object is used to calculate the contribution of each feature to a specific prediction.
explainer = shap.TreeExplainer(model)

def predict_and_explain(input_data: dict) -> dict:
    """
    Generates a scaled credit score and a SHAP-based explanation for the input features.
    This is the main function called by the FastAPI endpoint.
    
    Args:
        input_data: A dictionary containing the feature values (e.g., {'debtToEquity': 0.5, ...}).

    Returns:
        A dictionary with the final credit score, a plain-language summary, and the
        raw feature contribution values.
    """
    # Convert the input dictionary to a pandas DataFrame.
    # Specifying the columns ensures the data is in the same order as the training data,
    # which is a strict requirement for the model.
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # --- Step 1: Generate a raw score ---
    # The model predicts a value between 0 and 1.
    raw_score = model.predict(input_df)[0]
    
    # --- Step 2: Scale the score ---
    # Convert the 0-1 score to a more familiar 300-850 range, similar to a FICO score.
    scaled_score = 300 + raw_score * 550
    
    # --- Step 3: Generate SHAP values ---
    # This is the core of the "Explainability Layer". It calculates how much each feature
    # pushed the prediction away from the average prediction.
    shap_values = explainer.shap_values(input_df)
    # The result is a numpy array; we convert it to a user-friendly dictionary.
    feature_contributions = dict(zip(feature_names, shap_values[0]))
    
    # --- Step 4: Generate a plain-language summary ---
    # This translates the numerical SHAP values into an understandable sentence.
    summary = generate_summary(feature_contributions)
    
    # --- Step 5: Return the complete result ---
    return {
        'creditScore': int(scaled_score),
        'summary': summary,
        'featureContributions': feature_contributions
    }

def generate_summary(contributions: dict) -> str:
    """Generates a plain-language summary from the feature contribution values."""
    # Sort features by the absolute magnitude of their impact to find the most influential ones.
    sorted_features = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    
    # Identify the primary driver of the score
    primary_driver = sorted_features[0]
    primary_feature_name = primary_driver[0].replace("_", " ").title()
    primary_impact = "positively" if primary_driver[1] > 0 else "negatively"
    
    summary = f"The score is primarily influenced by the '{primary_feature_name}' feature, which has a {primary_impact} impact. "
    
    # Identify the secondary driver, if it exists
    if len(sorted_features) > 1:
        secondary_driver = sorted_features[1]
        secondary_feature_name = secondary_driver[0].replace("_", " ").title()
        secondary_impact = "positive" if secondary_driver[1] > 0 else "negative"
        summary += f"The next most significant factor is '{secondary_feature_name}', which has a {secondary_impact} contribution."
        
    return summary.strip()

# ======================================================================================
# FILE: frontend/package.json
# ======================================================================================
{
  "name": "frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "axios": "^0.27.2",
    "chart.js": "^3.9.1",
    "react": "^18.2.0",
    "react-chartjs-2": "^4.3.1",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}

# ======================================================================================
# FILE: frontend/nginx.conf
# ======================================================================================
server {
  listen 80;
  server_name localhost;

  # Serve the static React files from the build directory
  location / {
    root /usr/share/nginx/html;
    index index.html index.htm;
    try_files $uri $uri/ /index.html; # This is crucial for single-page app routing
  }

  # Reverse proxy for API calls to the backend gateway
  # Any request to /api/... will be forwarded to the api_gateway service.
  location /api {
    proxy_pass http://api_gateway:8000; # The gateway service is on port 8000 internally
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}

# ======================================================================================
# FILE: frontend/Dockerfile
# ======================================================================================
# Stage 1: Build the React app
FROM node:16-alpine as build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . ./
RUN npm run build

# Stage 2: Serve the app with Nginx
FROM nginx:1.23-alpine
# Copy the static build output from the previous stage
COPY --from=build /app/build /usr/share/nginx/html
# Copy the custom Nginx configuration to handle client-side routing and proxying
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Expose port 80 for the Nginx server
EXPOSE 80
# Command to start the Nginx server
CMD ["nginx", "-g", "daemon off;"]
