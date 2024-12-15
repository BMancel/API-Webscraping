from fastapi import APIRouter, HTTPException
from src.schemas.message import MessageResponse
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

router = APIRouter()
data_dir = os.path.join(os.path.dirname(__file__), "../../data")

def load_iris_data_as_dataframe():
    iris_file = os.path.join(data_dir, "Iris.csv")
    if not os.path.exists(iris_file):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return pd.read_csv(iris_file)

def preprocess_iris_data(df: pd.DataFrame):
    X = df.drop(columns=["Id", "Species"])
    y = df["Species"]
    return X, y

@router.get("/load-iris-data", response_model=MessageResponse)
def load_iris_data():
    try:
        df = load_iris_data_as_dataframe()
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Iris dataset loaded successfully",
                "data": df.to_dict(orient="records")
            }
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "message": "Iris dataset file not found"
            }
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "The dataset file is empty"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"An unexpected error occurred: {str(e)}"
            }
        )

@router.get("/process-iris-data", response_model=MessageResponse)
def process_iris_data():
    try:
        df = load_iris_data_as_dataframe()
        X, y = preprocess_iris_data(df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Data processed successfully",
                "data": {
                    "X_scaled": X_scaled.tolist(),
                    "y": y.tolist()
                }
            }
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Error processing the data: {str(ve)}"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"An unexpected error occurred: {str(e)}"
            }
        )