from fastapi import APIRouter, HTTPException
from src.schemas.message import MessageResponse
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split
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

def split_data(X, y, test_size=0.2, random_state=42):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise Exception(f"Error during train-test split: {e}")

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
    
@router.get("/split-iris-data", response_model=MessageResponse)
def split_iris_data(test_size: float = 0.2, random_state: int = 42):
    try:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
            
        df = load_iris_data_as_dataframe()
        X, y = preprocess_iris_data(df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size, random_state)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Data split successfully",
                "data": {
                    "X_train": X_train.tolist(),
                    "X_test": X_test.tolist(),
                    "y_train": y_train.tolist(),
                    "y_test": y_test.tolist()
                }
            }
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Invalid parameters: {str(ve)}"
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