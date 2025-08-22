# 7mod3 - Real-Time Explainable Credit Intelligence Platform

*A Hackathon Submission by Yash Kamra & Tanya Goyanka*

## Our Aim

Our aim is to revolutionize credit risk analysis by replacing the slow, opaque, and lagging methodologies of traditional credit rating agencies. We leverage **Artificial Intelligence** and **high-frequency, alternative data** to build a real-time, transparent, and evidence-backed credit intelligence platform, empowering investors and analysts to make more informed and timely decisions.

## The Problem: Outdated Credit Ratings

In today's fast-paced global credit markets, decisions involving billions of dollars are still guided by traditional credit rating agencies. These legacy systems suffer from critical flaws:

- **Infrequent Updates**: Ratings are updated too slowly to reflect real-world events in a timely manner
- **Opaque Methodologies**: The reasoning behind a given rating is often a "black box," making it difficult for investors to trust and verify
- **Lagging Indicators**: They are often reactive, only changing a rating after a significant event has already impacted the market

This inefficiency creates significant **"mispricing opportunities"** where the true risk is different from what the market perceives.

## Our Solution: A Real-Time Explainable Platform

We have built a **Real-Time Explainable Credit Intelligence Platform** that directly addresses these challenges. Our product continuously ingests a wide array of public data, from financial filings to real-time news, to produce dynamic creditworthiness scores that are both faster and fully transparent.

## Product Offerings & Core Functions

Our platform provides analysts and investors with a powerful, interactive web dashboard to:

### Access Real-Time Credit Scores
We generate up-to-the-minute creditworthiness scores for various companies ("issuers") that react faster to market events than traditional ratings.

### Understand the "Why"
For every score, we provide clear, feature-level explanations, plain-language summaries, and trend insights. We show exactly which factors—from financial ratios to recent news—influenced the score.

### Track Score Trends
Through interactive visualizations, users can monitor a company's creditworthiness over time, identifying risks and opportunities as they emerge.

### Integrate Unstructured Data
Our system detects and interprets events from unstructured sources like news headlines and factors them directly into the credit score and its explanation, providing an early warning system for events like debt restructuring or warnings of declining demand.

## The Competitive Landscape

Our primary competitors are the traditional credit rating agencies (e.g., Moody's, S&P, Fitch).

| Feature | Traditional Agencies | 7mod3 Platform |
|---------|---------------------|----------------|
| **Speed** | Ratings are updated infrequently and lag behind events | Scores are generated in near-real-time and react faster to new information |
| **Transparency** | Methodologies are opaque, creating a "black box" | Fully explainable with feature contribution breakdowns and evidence-backed reasoning |
| **Data Sources** | Primarily rely on official company filings and reports | Fuses financial data with high-frequency, alternative, and unstructured data like real-time news |
| **Adaptability** | Static models that are slow to change | An adaptive scoring engine capable of frequent retraining to incorporate new data patterns |

## 🛠 Our Tech Stack

We chose a modern, scalable tech stack to meet the demands of a real-time data platform:

- **Frontend**: React.js, Chart.js
- **Backend**: Python, FastAPI
- **Machine Learning**: XGBoost, SHAP, Scikit-learn
- **Deployment**: Docker, Docker Compose, Nginx

## 📁 Project Structure

The project is organized into a clean, microservices-based architecture to ensure scalability and separation of concerns.

```
credtech-7mod3/
├── .env                    # Stores private API keys securely, not committed to Git
├── docker-compose.yml      # Orchestrates all Docker containers for easy setup
├── README.md               # Project explanation and guide
│
├── services/               # Contains all backend microservices
│   │
│   ├── api_gateway/        # Handles all incoming requests from the frontend
│   │   ├── app/
│   │   │   ├── main.py     # FastAPI application entry point for the gateway
│   │   │   └── services.py # Logic for communicating with other backend services
│   │   ├── Dockerfile      # Instructions to build the gateway's Docker image
│   │   └── requirements.txt# Python dependencies for the gateway
│   │
│   ├── data_ingestion/     # Service for fetching data from external APIs
│   │   ├── app/
│   │   │   ├── main.py     # Main script demonstrating the ingestion process
│   │   │   └── fetchers.py # Functions to fetch from Alpha Vantage and NewsAPI
│   │   └── ...
│   │
│   └── scoring_engine/     # Dedicated ML service for scoring and explanations
│       ├── app/
│       │   ├── main.py     # FastAPI application for serving the ML model
│       │   ├── model/      # Stores the trained XGBoost model files
│       │   └── ml_utils.py # Core logic for prediction and SHAP explanation
│       ├── train.py        # Script to train the ML model
│       └── ...
│
└── frontend/               # Contains the entire React user interface
    │
    ├── public/             # Static assets like the main HTML page
    ├── src/
    │   ├── components/     # Reusable React components (charts, tabs, watchlist)
    │   ├── App.css         # Main styling for the TradingView-inspired theme
    │   ├── App.js          # Main application component and layout logic
    │   └── index.js        # Entry point for the React application
    ├── Dockerfile          # Builds the React app and sets up an Nginx server
    ├── nginx.conf          # Nginx configuration for serving the app and proxying API calls
    └── package.json        # Frontend dependencies (React, Chart.js, etc.)
```

## 🚀 Deployment Instructions

### A. Local Deployment (for Development)

#### Prerequisites
- Docker and Docker Compose installed on your PC
- API keys from Alpha Vantage and NewsAPI

#### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd credtech-7mod3
```

#### Step 2: Create Environment File
Create a file named `.env` in the project's root directory:

```bash
# .env file
# Paste your actual keys here
NEWS_API_KEY=YOUR_NEWS_API_KEY_HERE
ALPHA_VANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_KEY_HERE
```

> **Note**: docker-compose automatically loads variables from a .env file.

#### Step 3: Train the Initial ML Model
The scoring_engine needs a model file to start. Run the training script locally once:

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

#### Step 4: Build and Run the Application
From the root directory (`credtech-7mod3/`), run:

```bash
docker compose up --build -d
```

The `-d` flag runs the containers in detached mode (in the background).

#### Step 5: Access the Platform
- **Frontend Dashboard**: http://localhost:3000
- **API Gateway Docs**: http://localhost:8000/docs

### B. Cloud Deployment (for Public Demo)

#### Prerequisites
A cloud server (e.g., AWS EC2, DigitalOcean Droplet) with Docker and Docker Compose installed.

#### Steps
1. SSH into your server and clone your Git repository
2. Create the `.env` file on the server with your production API keys
3. Run the `train.py` script on the server to generate the model files
4. Run Docker Compose: `docker compose up --build -d`
5. Configure Firewall/Security Groups: Ensure port 3000 is open to public traffic
6. Access your application via your server's public IP: `http://<your_server_ip>:3000`

## 📝 License

This project is developed as a hackathon submission by Yash Kamra & Tanya Goyanka.

---

**Built with ❤️ for revolutionizing credit intelligence**
