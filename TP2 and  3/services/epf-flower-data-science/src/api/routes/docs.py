from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import RedirectResponse
router = APIRouter()

# Redirect the root endpoint to the Swagger documentation
@router.get("/", include_in_schema=False)
def redirect_to_swagger(request: Request):
    return RedirectResponse(url="/docs")