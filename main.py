from fastapi import FastAPI
from pydantic import BaseModel
import gzip
import pickle
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# CORS middleware for handling cross-origin requests
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model to define the input structure for heart disease prediction
class HeartDiseaseInput(BaseModel):
    Age: int
    Sex: int
    ChestPainType: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: int
    MaxHR: int
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope: int

# Load the saved heart disease prediction model
try:
    with gzip.open('heart_disease_model.sav.gz', 'rb') as f:
        heart_disease_model = pickle.load(f)
except Exception as e:
    print(f"Error loading the heart disease model: {e}")

# Endpoint for heart disease prediction
@app.post('/smart-symptomChecker/heart-disease')
async def heart_disease_predict(input_parameters: HeartDiseaseInput):
    try:
        # Convert input data to dictionary and list for prediction
        input_data = input_parameters.dict()
        input_list = [
            input_data['Age'],
            input_data['Sex'],
            input_data['ChestPainType'],
            input_data['RestingBP'],
            input_data['Cholesterol'],
            input_data['FastingBS'],
            input_data['RestingECG'],
            input_data['MaxHR'],
            input_data['ExerciseAngina'],
            input_data['Oldpeak'],
            input_data['ST_Slope']
        ]

        # Perform prediction
        prediction = heart_disease_model.predict([input_list])

        # Generate appropriate response
        if prediction[0] == 0:
            return JSONResponse(content={"result": "The person does not have heart disease"}, status_code=200)
        else:
            return JSONResponse(content={"result": "The person has heart disease"}, status_code=200)

    except Exception as e:
        # Error handling in case of prediction failure
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
