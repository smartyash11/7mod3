import os
import requests

# API Keys are loaded from environment variables for security.
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
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set.")
    
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    r = requests.get(url)
    r.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
    data = r.json()
    
    if not data or "Note" in data or "Name" not in data:
        raise ValueError(f"Invalid or no structured data found for ticker '{ticker}'. The API limit might have been reached.")
        
    return data

def get_unstructured_data(ticker: str) -> dict:
    """
    Fetches recent news articles (unstructured data) from NewsAPI.

    Args:
        ticker: The company name or stock symbol to search for in the news.

    Returns:
        A dictionary containing the API response with news articles.

    Raises:
        ValueError: If the API key is not set.
        requests.exceptions.RequestException: If the API call fails.
    """
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY environment variable not set.")
        
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
    
    r = requests.get(url)
    r.raise_for_status()
    
    return r.json()