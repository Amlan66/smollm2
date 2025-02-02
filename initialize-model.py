from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torchinfo import summary

def load_model():
    # Specify the 135M model checkpoint
    checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print("Downloading and loading the 135M model. This may take a moment...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,  # Use FP16 for performance optimization
        device_map="auto"  # Automatically map to GPU if available
    )
    
    # Display model summary
    print("\nModel Summary:")
    # Create a sample input for summary
    batch_size = 1
    sequence_length = 128
    input_shape = (batch_size, sequence_length)
    
    # Generate and display the model summary
    model_stats = summary(
        model,
        input_data=torch.zeros(input_shape, dtype=torch.long),
        verbose=1,  # Changed to 1 for more detailed console output
        col_names=["input_size", "output_size", "num_params", "trainable"],
    )
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()
