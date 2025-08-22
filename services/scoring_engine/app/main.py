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