import gradio as gr
import torch
from smollm2_model import SmolLM2, SmolLM2Config
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model and tokenizer
def load_model():
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        
        # Initialize model configuration
        config = SmolLM2Config()
        config.vocab_size = len(tokenizer)
        
        # Create model
        model = SmolLM2(config)
        
        # Download and load trained weights from HuggingFace Hub
        logger.info("Downloading model checkpoint from HuggingFace Hub...")
        model_path = hf_hub_download(
            repo_id="amlanr66/smollm2",  # Replace with your model repository
            filename="checkpoint.pt"
        )
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_text(prompt, max_length=100, temperature=0.8):
    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            # Simple autoregressive generation
            current_ids = input_ids
            for _ in range(max_length - len(input_ids[0])):
                outputs = model(current_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Stop if we generate an end of text token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        generated_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return f"Error: {str(e)}"

# Load model and tokenizer globally
logger.info("Loading model and tokenizer...")
model, tokenizer = load_model()
logger.info("Model and tokenizer loaded successfully")

# Create Gradio interface
def predict(text, max_length, temperature):
    try:
        return generate_text(text, int(max_length), float(temperature))
    except Exception as e:
        return f"Error: {str(e)}"

# Define the interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Enter your prompt", placeholder="Type something here..."),
        gr.Slider(minimum=10, maximum=200, value=100, step=10, label="Maximum Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmolLM2 Text Generator",
    description="Enter a prompt and the model will continue generating text.",
    examples=[
        ["The quick brown fox", 100, 0.8],
        ["Once upon a time", 150, 0.9],
        ["In a galaxy far far", 120, 0.7]
    ]
)

# Launch the app
if __name__ == "__main__":
    iface.launch() 