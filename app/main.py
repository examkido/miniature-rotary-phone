from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os


# Define input data schema
class HouseData(BaseModel):
    CITY_NUMBER: int
    ADVERTISING_MONTH: float  
    POSTAL_CODE: float
    PROM: float   
    BEDROOMS: int   
    ADVERTISING_YEAR: float  
    MUNICIPALITY_AREA_SCORE: float

    
    class Config:
        json_schema_extra = {
            "example": {
                "CITY_NUMBER": 301,
                "ADVERTISING_MONTH": 0.05,
                "POSTAL_CODE": 0.06,
                "PROM": 0.02,
                "BEDROOMS": 4,
                "ADVERTISING_YEAR": -0.04,
                "MUNICIPALITY_AREA_SCORE": -0.02
            }
        }

# Initialize FastAPI app
app = FastAPI(
    title="House Monthly rent prediction",
    description="",
    version="1.0.0"
)
 
# Load the trained model
model_path = os.path.join("models", "house_model_01.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)



@app.post("/predict")
def predict_progression(house_rent: HouseData):
    """
    Predict monthly rent progression score
    """
    # Convert input to numpy array
    features = np.array([[
        house_rent.CITY_NUMBER, house_rent.ADVERTISING_MONTH, house_rent.POSTAL_CODE, house_rent.PROM,
        house_rent.BEDROOMS, house_rent.ADVERTISING_YEAR, house_rent.MUNICIPALITY_AREA_SCORE
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Return result with additional context
    return {
        "predicted_progression_score": round(prediction, 2),
        "interpretation": get_interpretation(prediction)
    }
 
def get_interpretation(score):
    """Provide human-readable interpretation of the score"""
    if score < 100:
        return "Below average progression"
    elif score < 150:
        return "Average progression"
    else:
        return "Above average progression"
    


@app.get("/")
def health_check():
    return {"status": "healthy", "model": "House Rent"}