import os
import requests

# Service URLs and API Keys are loaded from environment variables for security and flexibility.
SCORING_ENGINE_URL = os.getenv("SCORING_ENGINE_URL")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_structured_data(ticker: str) -> dict:
    """
    Fetches company overview data (structured financial data) from Alpha Vantage.
    
    Args:
        ticker: The stock symbol of the company (e.g., "MSFT").

    Returns:
        A dictionary containing the company's financial overview.
    
    Raises:
        ValueError: If the API key is not set or if no data is found for the ticker.
        requests.exceptions.RequestException: If the API call fails.
    """
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == 'YOUR_ALPHA_VANTAGE_KEY_HERE':
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set or is a placeholder.")
    
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    r = requests.get(url)
    r.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
    data = r.json()
    
    if not data or "Note" in data or "Name" not in data:
        raise ValueError(f"Invalid or no structured data found for ticker '{ticker}'. The API limit might have been reached.")
        
    return data

def get_unstructured_data(ticker: str) -> dict:
    """
    Fetches news headlines from NewsAPI and calculates a simple sentiment score.

    Args:
        ticker: The company name or stock symbol to search for in the news.

    Returns:
        A dictionary containing headlines and a calculated sentiment score.

    Raises:
        ValueError: If the API key is not set.
        requests.exceptions.RequestException: If the API call fails.
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
    
    Args:
        features: A dictionary with the feature values required by the ML model.

    Returns:
        A dictionary containing the response from the scoring engine.
    """
    response = requests.post(f"{SCORING_ENGINE_URL}/score", json=features)
    response.raise_for_status()
    return response.json()