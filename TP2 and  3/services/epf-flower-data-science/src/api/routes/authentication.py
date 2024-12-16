from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from firebase_admin import auth, firestore
import firebase_admin

if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.client()

router = APIRouter()

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/register")
async def register_user(request: RegisterRequest):
    try:
        settings_ref = db.collection("settings").document("app_info")
        doc = settings_ref.get()

        if not doc.exists:
            user = auth.create_user(email=request.email, password=request.password)

            auth.set_custom_user_claims(user.uid, {"admin": True})

            settings_ref.set({"is_first_user": False})

            return {
                "message": "User registered successfully",
                "user_id": user.uid,
                "admin": True
            }

        user = auth.create_user(email=request.email, password=request.password)

        auth.set_custom_user_claims(user.uid, {"admin": False})

        return {
            "message": "User registered successfully",
            "user_id": user.uid,
            "admin": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating user: {e}")


