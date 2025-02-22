from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from smollm2_model import SmolLM2, SmolLM2Config
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.8

class GenerationResponse(BaseModel):
    generated_text: str

# Initialize model and tokenizer
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        config = SmolLM2Config()
        config.vocab_size = len(tokenizer)
        model = SmolLM2(config)
        
        # Load checkpoint
        checkpoint = torch.load('checkpoint.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load model globally
model, tokenizer = load_model()

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        input_ids = tokenizer.encode(request.prompt, return_tensors='pt')
        
        with torch.no_grad():
            current_ids = input_ids
            for _ in range(request.max_length - len(input_ids[0])):
                outputs = model(current_ids)
                next_token_logits = outputs[:, -1, :] / request.temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        generated_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        return GenerationResponse(generated_text=generated_text)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 