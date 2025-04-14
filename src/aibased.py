import torch
import os
from diffusers import StableDiffusionPipeline
from PIL import Image

def download_model():
    """
    Downloads the model for first-time use.
    This will be stored in the Hugging Face cache directory.
    """
    print("Downloading AI model for image generation (this might take a while on first run)...")
    return StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch.float32  
    )

def generate_aadhar_image(prompt, output_path="ai_generated_aadhar.jpg", apply_blur_effect=False, blur_strength=2):
    """
    Generate an Aadhar card-like image using AI model.
    
    Args:
        prompt: Text prompt describing the Aadhar card
        output_path: Path to save the generated image
        apply_blur_effect: Whether to apply blur to the final image
        blur_strength: Strength of blur effect (1-5)
        
    Returns:
        Path to the generated image
    """
    try:
        enhanced_prompt = f"An official Indian Aadhar card, government ID document, with details, {prompt}, highly detailed, realistic"
        
        device = "mps"
        print(f"Using device: {device}")
        
        pipe = download_model()
        pipe = pipe.to(device)
        
        print(f"Generating image with prompt: {enhanced_prompt}")
        with torch.no_grad():
            image = pipe(enhanced_prompt).images[0]
        
        if apply_blur_effect:
            print(f"Applying blur effect with strength {blur_strength}")
            image = image.filter(Image.FILTER.GaussianBlur(radius=blur_strength))
        
        image.save(output_path)
        print(f"AI-generated image saved as: {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error generating image with AI: {str(e)}")
        
        if os.path.exists("aadharme.jpeg"):
            print("Using fallback template image instead")
            fallback_image = Image.open("aadharme.jpeg")
            
            if apply_blur_effect:
                fallback_image = fallback_image.filter(Image.FILTER.GaussianBlur(radius=blur_strength))
                
            fallback_image.save(output_path)
            return output_path
        return None

if __name__ == "__main__":
    generate_aadhar_image("Person named John Doe, born on 15/05/1990")
