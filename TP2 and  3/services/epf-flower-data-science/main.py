import uvicorn

from src.app import get_application

app = get_application()

@app.get("/openapi.json")
async def get_openapi():
    openapi_schema = app.openapi()
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    openapi_schema["security"] = [{"BearerAuth": []}]
    return openapi_schema

if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)
