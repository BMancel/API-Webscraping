from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from firebase_admin import auth, firestore
import firebase_admin
from datetime import datetime, timedelta
import jwt
from secret_config import SECRET_KEY

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

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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

@router.post("/login")
async def login(request: LoginRequest):
    """
    Endpoint pour connecter un utilisateur et générer un token JWT.
    """
    try:
        user = auth.get_user_by_email(request.email)

        access_token = create_access_token(data={"sub": user.uid, "email": user.email})
        
        return {"access_token": access_token, "token_type": "bearer"}

    except auth.AuthError as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")