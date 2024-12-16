from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from firebase_admin import firestore, credentials, initialize_app
import os
import firebase_admin

if not firebase_admin._apps:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(base_dir, "../../../../../private_key.json")
    print(f"Using key at: {key_path}")
    cred = credentials.Certificate(key_path)
    initialize_app(cred)

router = APIRouter()

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