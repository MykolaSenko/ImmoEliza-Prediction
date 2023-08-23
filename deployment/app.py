from fastapi import FastAPI
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, validator
import src.preprocessing as proc
import src.predict as pred


app = FastAPI()


class PropertyData(BaseModel):
    property_type: str
    floor: NonNegativeInt
    bedrooms_number: NonNegativeInt
    habitable_surface: NonNegativeFloat
    bathroom_number: NonNegativeInt
    condition: NonNegativeInt

    @validator('property_type')
    def validate_property_type(cls, value):
        if value.lower() not in ["house", "apartment"]:
            raise ValueError(
                "Property type must be either 'House' or 'Apartment'")
        return value

    @validator('condition')
    def validate_condition_range(cls, value):
        if value < 0 or value > 5:
            raise ValueError("Condition must be an integer between 0 and 5.")
        return value

class InputData(BaseModel):
    data: PropertyData


@app.get('/')
def whatever_func():
    text = '''This is an API for price prediction for Belgian real estate.
This model is built on base of trained data of more that 15000 Belgian real estate properties both apartments and 
houses using XGBoos regressor. To make a prediction of a property price you need to input data in JSON format as follows:
{"data":
    {"property_type": string. Expected "House" or "Apartment",
    "floor": positive integer,
    "bedrooms_number": positive integer,
    "habitable_surface": positive float,
    "bathroom_number": positive integer,
    "condition": integer. Expected from 0 to 5. O - very bad condition, to be done up, 5 - new.
        }
            } '''
    return text


@app.post('/calc')
async def predict_price(data: InputData):
    X_pred_pr = proc.processing(data)
    xgbr = pred.open_reg(X_pred_pr)
    y_pred = pred.predict_new(X_pred_pr, xgbr)
    return y_pred
