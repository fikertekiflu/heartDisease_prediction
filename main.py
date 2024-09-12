from fastapi import FastAPI
from pydantic import BaseModel
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
    ChestPain: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestECG: int
    MaxHeartRate: int
    ExerciseAngina: int
    OldPeak: float
    Slope: int
    MajorVessels: int
    Thalassemia: int

# Load the saved heart disease prediction model (no gzip, just pickle)
try:
    with open('heart_disease_model.sav', 'rb') as f:
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
            input_data['ChestPain'],
            input_data['RestingBP'],
            input_data['Cholesterol'],
            input_data['FastingBS'],
            input_data['RestECG'],
            input_data['MaxHeartRate'],
            input_data['ExerciseAngina'],
            input_data['OldPeak'],
            input_data['Slope'],
            input_data['MajorVessels'],
            input_data['Thalassemia']
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
