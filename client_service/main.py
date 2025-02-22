from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import logging
import traceback
from typing import Optional

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.8

class GenerationResponse(BaseModel):
    generated_text: str

MODEL_SERVICE_URL = "http://model_service:8000/generate"

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        logger.info(f"Received request with prompt: {request.prompt}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Sending request to model service: {MODEL_SERVICE_URL}")
            response = await client.post(
                MODEL_SERVICE_URL,
                json=request.dict()
            )
            
            logger.info(f"Received response with status code: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info("Successfully received generated text")
                return GenerationResponse(generated_text=response_data["generated_text"])
            else:
                error_detail = response.json().get("detail", "Unknown error")
                logger.error(f"Model service error: {error_detail}")
                raise HTTPException(status_code=response.status_code, detail=error_detail)
                
    except httpx.HTTPError as e:
        error_msg = f"HTTP error occurred: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://model_service:8000/health")
            if response.status_code == 200:
                return {"status": "healthy", "model_service": "connected"}
            else:
                return {"status": "unhealthy", "reason": "model service not responding properly"}
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)} 