from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from firebase_admin import firestore, credentials, initialize_app
import os
import firebase_admin
from pydantic import BaseModel, Field
from google.cloud import exceptions

if not firebase_admin._apps:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(base_dir, "../../../../../private_key.json")
    print(f"Using key at: {key_path}")
    cred = credentials.Certificate(key_path)
    initialize_app(cred)

router = APIRouter()

class ParametersRequest(BaseModel):
    new_parameters: dict = Field(
        ..., 
        example={"key1": "value1", "key2": "value2", "key3": "value3"}
    )

@router.get("/retrieve-parameters", response_model=dict)
def retrieve_parameters():
    try:
        parameters_ref = firestore.client().collection("parameters").document("parameters")
        doc = parameters_ref.get()

        if doc.exists:
            parameters = doc.to_dict()
            return JSONResponse(content={"parameters": parameters})
        else:
            raise HTTPException(status_code=404, detail="Parameters document not found in Firestore")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-parameters", response_model=dict)
def add_parameters(request: ParametersRequest):
    try:
        parameters_ref = firestore.client().collection("parameters").document("parameters")
        
        doc = parameters_ref.get()
        
        if doc.exists:
            existing_parameters = doc.to_dict()
            merged_parameters = {**existing_parameters, **request.new_parameters}
            parameters_ref.set(merged_parameters)
            return JSONResponse(content={"message": "Parameters merged successfully", "parameters": merged_parameters})
        else:
            parameters_ref.set(request.new_parameters)
            return JSONResponse(content={"message": "New parameters document created successfully", "parameters": request.new_parameters})
    
    except exceptions.GoogleCloudError as e:
        raise HTTPException(status_code=500, detail=f"Firestore error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
