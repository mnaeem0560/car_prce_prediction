from fastapi import FastAPI
from pydantic import BaseModel, PositiveFloat
from typing import Literal, List
from fastapi.middleware.cors import CORSMiddleware

from src.creat_df import load_data
from src.load_model import load_model
from src.preprocess import preprocess_data
from src.predict import predict

class InputData(BaseModel):
    Make : List[str]       
    Year : List[int]     
    Engine_Fuel_Type : List[str]       
    Engine_HP : List[PositiveFloat]   
    Engine_Cylinders : List[PositiveFloat]  
    Transmission_Type : List[str]  
    Driven_Wheels : List[str]  
    Number_of_Doors : List[PositiveFloat]   
    Vehicle_Size : List[str]  
    Vehicle_Style : List[str]   
    highway_MPG : List[int]
    city_mpg : List[int]
    Fuel_Efficiency : List[PositiveFloat]
    Automatic : List[int]

app = FastAPI(
    title="Car Price Prediction API",
    description="API for predicting Price of Car based on Given Specification.",
    version="1.0.0"
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/pred")
def pred(input_data: InputData):
    input_dict = input_data.model_dump()
    columns = ['Make', 'Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders',
       'Transmission Type', 'Driven_Wheels', 'Number of Doors', 'Vehicle Size',
       'Vehicle Style', 'highway MPG', 'city mpg', 'Fuel_Efficiency',
       'Automatic']
    input_dict_ = {columns[i]: value for i, value in enumerate(input_dict.values())}
    input_df = load_data(input_dict_)
    loaded_model = load_model("artifacts/car_price_prediction.joblib")

    preprocessed_df = preprocess_data(input_df, loaded_model)

    prediction = predict(preprocessed_df, loaded_model)

    return {"prediction": prediction}