from fetchers import get_structured_data, get_unstructured_data
import time

def main():
    """
    This script simulates an ingestion cycle. In a real MLOps system, this would
    be a scheduled job (e.g., a cron job or an Airflow DAG) that runs periodically
    to fetch and store data in a database.
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