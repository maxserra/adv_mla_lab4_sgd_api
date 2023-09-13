import joblib
import json

import pandas as pd

from fastapi import FastAPI
from starlette.responses import JSONResponse


app = FastAPI()
sgd_pipe = joblib.load("../models/sgd_pipe.joblib")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health", status_code=200)
def healthcheck():
    return "SGDClassifier is all ready to go!"


def format_features(
    general_health,
    checkup,
    exercise,
    heart_disease,
    skin_cancer,
    other_cancer,
    depression,
    diabetes,
    arthritis,
    sex,
    age_category,
    height,
    weight,
    bmi,
    smoking_history,
    alcohol_consumption,
    fruit_consumption,
    green_vegetables_consumption,
    friedpotato_consumption,
):
    return {
        "General_Health": general_health,
        "Checkup": checkup,
        "Exercise": exercise,
        "Heart_Disease": heart_disease,
        "Skin_Cancer": skin_cancer,
        "Other_Cancer": other_cancer,
        "Depression": depression,
        "Diabetes": diabetes,
        "Arthritis": arthritis,
        "Sex": sex,
        "Age_Category": age_category,
        "Height_(cm)": height,
        "Weight_(kg)": weight,
        "BMI": bmi,
        "Smoking_History": smoking_history,
        "Alcohol_Consumption": alcohol_consumption,
        "Fruit_Consumption": fruit_consumption,
        "Green_Vegetables_Consumption": green_vegetables_consumption,
        "FriedPotato_Consumption": friedpotato_consumption,
    }


@app.get("/cvd/risks/prediction")
def predict(
    general_health,
    checkup,
    exercise,
    heart_disease,
    skin_cancer,
    other_cancer,
    depression,
    diabetes,
    arthritis,
    sex,
    age_category,
    height,
    weight,
    bmi,
    smoking_history,
    alcohol_consumption,
    fruit_consumption,
    green_vegetables_consumption,
    friedpotato_consumption,
):
    
    features_dict = format_features(general_health, checkup, exercise, heart_disease,
                                    skin_cancer, other_cancer, depression, diabetes,
                                    arthritis, sex, age_category, height, weight, bmi,
                                    smoking_history, alcohol_consumption, fruit_consumption,
                                    green_vegetables_consumption, friedpotato_consumption,)
    
    features_df = pd.DataFrame(features_dict, index=[0])

    pred = sgd_pipe.predict(features_df)[0]
    pred = {"Heart_Disease": pred}

    return JSONResponse(content=json.dumps(pred), media_type="application/json")
