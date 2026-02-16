from PIL import Image
import easyocr
import json
from langchain_ollama import OllamaLLM  

def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en']) 
    result = reader.readtext(image_path)
    
    extracted_text = " ".join([text[1] for text in result])
    return extracted_text

def structure_data_with_llm(extracted_text):
    llm = OllamaLLM(model="llama2:7b") 
    
    prompt_template = """
    You are given the following ID card data extracted from an image:

    {extracted_text}

    Please extract the following details and return them strictly in JSON format with the following fields in this exact order:
    - "name": "string"
    - "date_of_birth": "date"  
    - "12_digit_number": "string"  # Extract the 12-digit number even if it appears shuffled. Ensure the digits are in the correct order
    - "gender": "string"

    The JSON format should look exactly like this example:
    {{
        "name": "vishwa",
        "date_of_birth": "12/10/2004",
        "12_digit_number": "1943 6593 1261",
        "gender": "male"
    }}
    """
    
    prompt = prompt_template.format(extracted_text=extracted_text)
    
    structured_data = llm.invoke(prompt)
    
    print("Raw Output from LLM:", structured_data)
    
    try:
        json_data = json.loads(structured_data)
    except json.JSONDecodeError:
        return "Error parsing the data into JSON."
    
    return json_data

def main(image_path):
    extracted_text = extract_text_from_image(image_path)
    
    print("Extracted Text:", extracted_text)
    
    structured_data = structure_data_with_llm(extracted_text)
    
    print(json.dumps(structured_data, indent=4))

if __name__ == "__main__":
    image_path = "aadhar_card.png" 
    main(image_path)
