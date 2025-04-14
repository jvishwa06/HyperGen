import os
import re
import gradio as gr
from cvbased import main as edit_aadhar
from aibased import generate_aadhar_image

def extract_details_from_prompt(prompt):
    """
    Extract name, DOB, and Aadhar number from natural language prompt.
    
    Args:
        prompt: User's natural language prompt
        
    Returns:
        Tuple containing (name, dob, aadhar_number)
    """
    name = None
    dob = None
    aadhar_number = None
    
    name_patterns = [
        r'(?:name|called)\s+(?:is\s+)?([A-Za-z\s]+?)(?:\s+and|\s+with|\s+dob|\s+aadhar|$)',
        r'(?:name\s+)([A-Za-z\s]+?)(?:\s+and|\s+with|\s+dob|\s+aadhar|$)',
        r'(?:name|called)\s+(?:as\s+)?([A-Za-z\s]+?)(?:\s+and|\s+with|\s+dob|\s+aadhar|$)',
        r'(?:with|has)\s+(?:name\s+)?([A-Za-z\s]+?)(?:\s+and|\s+with|\s+dob|\s+aadhar|$)'
    ]
    
    for pattern in name_patterns:
        name_match = re.search(pattern, prompt.lower())
        if name_match:
            name = name_match.group(1).strip().title()
            break
    
    dob_patterns = [
        r'(?:dob|date\s+of\s+birth|born\s+on)\s+(?:is\s+)?(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})'
    ]
    
    for pattern in dob_patterns:
        dob_match = re.search(pattern, prompt)
        if dob_match:
            dob = dob_match.group(1).strip()
            dob = re.sub(r'[-\.]', '/', dob)
            break
    
    aadhar_patterns = [
        r'(?:aadhar|id|number)\s+(?:is\s+)?(\d{4}\s*\d{4}\s*\d{4})',
        r'(\d{4}\s*\d{4}\s*\d{4})'
    ]
    
    for pattern in aadhar_patterns:
        aadhar_match = re.search(pattern, prompt)
        if aadhar_match:
            aadhar_number = aadhar_match.group(1).strip()
            aadhar_number = re.sub(r'\s+', ' ', aadhar_number)
            if " " not in aadhar_number:
                aadhar_number = f"{aadhar_number[:4]} {aadhar_number[4:8]} {aadhar_number[8:12]}"
            break
    
    return name, dob, aadhar_number

def generate_aadhar_card(prompt, generation_method="cv", apply_blur=False, blur_strength=2):
    """
    Process natural language prompt and generate modified Aadhar card.
    
    Args:
        prompt: User's natural language prompt
        generation_method: Method to use for generation ('cv' or 'ai')
        apply_blur: Whether to apply blur effect to the image
        blur_strength: Strength of the blur effect (1-5)
        
    Returns:
        Path to the generated image or error message
    """
    try:
        name, dob, aadhar_number = extract_details_from_prompt(prompt)
        
        if not name and not dob and not aadhar_number:
            return "Could not identify any valid information in your prompt. Please include at least a name, DOB, or Aadhar number."
        
        details_message = []
        if name:
            details_message.append(f"Name: {name}")
        else:
            details_message.append("Name will be randomly generated")
            
        if dob:
            details_message.append(f"DOB: {dob}")
        else:
            details_message.append("DOB will be randomly generated")
            
        if aadhar_number:
            details_message.append(f"Aadhar Number: {aadhar_number}")
        else:
            details_message.append("Aadhar Number will be randomly generated")
        
        print("\n".join(details_message))
        
        input_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aadharme.jpeg")
        output_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_aadhar.jpg")
        
        if generation_method == "ai":
            print(f"Using AI-based generation method with blur={apply_blur}, blur_strength={blur_strength}")
            ai_prompt = f"An Aadhar card for person named {name or 'Random Person'}"
            if dob:
                ai_prompt += f" born on {dob}"
            if aadhar_number:
                ai_prompt += f" with ID number {aadhar_number}"
                
            result_path = generate_aadhar_image(
                ai_prompt,
                output_path=output_image_path,
                apply_blur_effect=apply_blur,
                blur_strength=blur_strength
            )
        else:
            print(f"Using CV-based generation method with blur={apply_blur}, blur_strength={blur_strength}")
            result_path = edit_aadhar(
                image_path=input_image_path,
                new_name=name,
                new_dob=dob,
                new_aadhar=aadhar_number,
                output_path=output_image_path,
                apply_blur_effect=apply_blur,
                blur_strength=blur_strength
            )
        
        if result_path:
            return result_path
        else:
            return "Failed to generate Aadhar card. Please try with a different prompt."
    
    except Exception as e:
        return f"Error: {str(e)}"

def setup_gradio_ui():
    """
    Set up and launch the Gradio UI.
    """
    with gr.Blocks(title="Aadhar Card Generator") as demo:
        gr.Markdown("# Aadhar Card Digital Editor")
        gr.Markdown("""
        Enter a prompt describing the details for the Aadhar card.
        
        Example: "Create a card with the name Vishwa and DOB 20/09/2003 and Aadhar number 7892 7654 1245"
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Create a card with the name..."
                )
                
                with gr.Row():
                    generation_method = gr.Radio(
                        ["CV-based", "AI-based"],
                        label="Generation Method",
                        value="CV-based"
                    )
                
                with gr.Row():
                    apply_blur = gr.Checkbox(
                        label="Apply Blur Effect",
                        value=False
                    )
                    blur_strength = gr.Slider(
                        minimum=0.5,
                        maximum=5.0,
                        step=0.5,
                        value=2.0,
                        label="Blur Strength"
                    )
                
                submit_btn = gr.Button("Generate Aadhar Card")
            
            with gr.Column(scale=3):
                output_image = gr.Image(label="Generated Aadhar Card")
                
        
        def process_inputs(prompt, method, blur, strength):
            generation_method = "cv" if method == "CV-based" else "ai"
            return generate_aadhar_card(prompt, generation_method, blur, int(strength))
        
        submit_btn.click(
            fn=process_inputs,
            inputs=[prompt_input, generation_method, apply_blur, blur_strength],
            outputs=output_image
        )
    
    demo.launch()

if __name__ == "__main__":
    setup_gradio_ui()
