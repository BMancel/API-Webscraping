from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import json
import os
from joblib import dump
from sklearn.preprocessing import StandardScaler
from src.api.routes.data import load_iris_data_as_dataframe, preprocess_iris_data, split_data

router = APIRouter()

@router.post("/train-iris-model", response_model=dict)
def train_iris_model():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(base_dir, "../../config/model_parameters.json")
        
        with open(params_path, "r") as f:
            params = json.load(f)

        df = load_iris_data_as_dataframe()
        X, y = preprocess_iris_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models_dir = os.path.join(base_dir, "../../models")
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        dump(scaler, scaler_path)

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"]
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "iris_classification_model_rf.joblib")
        dump(model, model_path)
        
        response = {
            "message": "Model trained and saved successfully",
            "accuracy": accuracy,
            "model_file": model_path
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
