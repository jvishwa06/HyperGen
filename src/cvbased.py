import easyocr
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import re
import random
from datetime import datetime, timedelta


def load_image(image_path):
    """
    Load the image and prepare it for processing.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple containing (cv2 image, image height, image width)
    """
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image_height, image_width = image_cv.shape[:2]
    print(f"Image loaded: {image_path}, Dimensions: {image_width}x{image_height}")
    return image_cv, image_height, image_width


def initialize_ocr():
    """Initialize the OCR reader."""
    print("Initializing OCR reader...")
    return easyocr.Reader(['en'], gpu=False)


def detect_name_field(results, image_height):
    """
    Detect the name field in the OCR results.
    
    Args:
        results: OCR results from easyocr
        image_height: Height of the image
        
    Returns:
        Tuple containing (name_bbox, name_text)
    """
    name_bbox = None
    name_text = None
    
    for (bbox, text, prob) in results:
        y_top = bbox[0][1]
        
        is_proper_name = (
            text.replace(" ", "").isalpha() and
            len(text.split()) >= 1 and  
            prob > 0.5 and  
            y_top < image_height * 0.7 and  
            "government of india" not in text.lower() and
            len(text) > 3 
        )
        
        if is_proper_name:
            name_bbox = bbox
            name_text = text
            print(f"Likely Name Field Detected: {text}")
            print(f"Coordinates: {bbox}")
            break
    
    if name_bbox is None:
        print("No name found with strict criteria, trying with relaxed criteria...")
        
        results_sorted = sorted(results, key=lambda x: x[0][0][1])
        
        for (bbox, text, prob) in results_sorted[:5]:
            if len(text) > 3 and prob > 0.4:
                name_bbox = bbox
                name_text = text
                print(f"Using relaxed criteria - Likely Name Field: {text}")
                print(f"Coordinates: {bbox}")
                break
    
    return name_bbox, name_text


def detect_dob_field(results):
    """
    Detect the date of birth field in the OCR results.
    
    Args:
        results: OCR results from easyocr
        
    Returns:
        Tuple containing (dob_bbox, dob_text)
    """
    dob_bbox = None
    dob_text = None
    date_pattern = r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'
    dob_indicators = ["dob"]
    
    for (bbox, text, prob) in results:
        date_matches = re.findall(date_pattern, text)
        
        if date_matches or any(indicator in text.lower() for indicator in dob_indicators):
            if "DOB:" in text and date_matches:
                print(f"Found text with DOB label: {text}")
                
                label_start = text.find("DOB:")
                colon_pos = label_start + len("DOB")  
                label_end_index = colon_pos + 1       
                
                date_match = re.search(date_pattern, text)
                if date_match:
                    date_index = date_match.start()
                    date_length = date_match.end() - date_match.start()
                    
                    x0, y0 = bbox[0]
                    x1, y1 = bbox[1]
                    x2, y2 = bbox[2]
                    x3, y3 = bbox[3]
                    
                    char_width = (x2 - x0) / len(text) if len(text) > 0 else 0
                    
                    colon_x = x0 + ((label_start + len("DOB:")) * char_width)
                    
                    if date_index > 0 and text[label_end_index:date_index].strip() == "":
                        date_x0 = x0 + (date_index * char_width)
                    else:
                        date_x0 = colon_x + (char_width * 0.5)  
                        
                    date_x2 = date_x0 + (date_length * char_width)
                    
                    date_bbox = [
                        [date_x0, y0], 
                        [date_x2, y1],
                        [date_x2, y2],
                        [date_x0, y3]
                    ]
                    
                    dob_bbox = date_bbox
                    dob_text = date_match.group()
                    print(f"Target DOB field detected: {text} (contains {dob_text})")
                    print(f"Date portion coordinates: {dob_bbox}")
                    break
            elif date_matches:
                date_match = date_matches[0]
                date_index = text.find(date_match)
                date_length = len(date_match)
                
                if date_index > 0 or len(text) > date_length:
                    x0, y0 = bbox[0]
                    x1, y1 = bbox[1]
                    x2, y2 = bbox[2]
                    x3, y3 = bbox[3]
                    
                    char_width = (x2 - x0) / len(text) if len(text) > 0 else 0
                    
                    date_x0 = x0 + (date_index * char_width)
                    date_x2 = date_x0 + (date_length * char_width)
                    
                    date_bbox = [
                        [date_x0, y0], 
                        [date_x2, y1],
                        [date_x2, y2],
                        [date_x0, y3]
                    ]
                    
                    dob_bbox = date_bbox
                    print(f"Refined DOB coordinates for exact date position: {date_bbox}")
                else:
                    dob_bbox = bbox
                
                dob_text = date_match
                print(f"Target DOB field detected: {text} (contains {dob_text})")
                print(f"DOB Coordinates: {dob_bbox}")
                break
            else:
                dob_bbox = bbox
                dob_text = text
                print(f"DOB field detected: {text}")
                print(f"DOB Coordinates: {bbox}")
                break
    
    return dob_bbox, dob_text


def detect_aadhar_number(results):
    """
    Detect the Aadhar number field in the OCR results.
    
    Args:
        results: OCR results from easyocr
        
    Returns:
        Tuple containing (aadhar_bbox, aadhar_text)
    """
    aadhar_bbox = None
    aadhar_text = None
    
    for (bbox, text, prob) in results:
        aadhar_pattern = r'\d{4}\s*\d{4}\s*\d{4}'
        matches = re.findall(aadhar_pattern, text)
        
        if matches or (len(text.replace(" ", "")) >= 10 and text.replace(" ", "").isdigit()):
            aadhar_bbox = bbox
            aadhar_text = text
            print(f"Likely Aadhar Number detected: {text}")
            print(f"Aadhar Number coordinates: {bbox}")
            break
    
    return aadhar_bbox, aadhar_text


def get_system_font():
    """
    Get a suitable system font for use, with fallbacks.
    
    Returns:
        Font path or None if no suitable font is found
    """
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    if not os.path.exists(font_path):
        font_path = "/Library/Fonts/Arial.ttf"
    
    if not os.path.exists(font_path):
        print("System fonts not found, will use default font")
        return None
    
    print(f"Using font: {font_path}")
    return font_path


def adjust_font_size(draw, text, max_width, font_path, initial_size, min_size=10):
    """
    Adjust font size to fit text within a given width.
    
    Args:
        draw: PIL ImageDraw object
        text: Text to fit
        max_width: Maximum width in pixels
        font_path: Path to the font file
        initial_size: Initial font size to try
        min_size: Minimum allowable font size
        
    Returns:
        Tuple of (adjusted font, text width, text height)
    """
    font_size = initial_size
    font = None
    text_width = 0
    text_height = 0
    
    while True:
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
                break
                
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            if text_width > max_width:
                font_size -= 1
                if font_size < min_size:
                    font_size = min_size
                    font = ImageFont.truetype(font_path, font_size)
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    break
            else:
                break
        except Exception as e:
            print(f"Font adjustment error: {e}")
            font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            break
    
    return font, text_width, text_height


def modify_name(draw, name_bbox, original_name, new_name, font_path):
    """
    Modify the name field in the image.
    
    Args:
        draw: PIL ImageDraw object
        name_bbox: Bounding box coordinates for the name
        original_name: Original detected name text
        new_name: New name to insert
        font_path: Path to the font file
        
    Returns:
        The font used (for potential reuse)
    """
    if not name_bbox:
        print("No name bbox provided, skipping name modification")
        return None
        
    top_left = (int(name_bbox[0][0]), int(name_bbox[0][1]))
    bottom_right = (int(name_bbox[2][0]), int(name_bbox[2][1]))
    bbox_width = bottom_right[0] - top_left[0]
    bbox_height = bottom_right[1] - top_left[1]
    
    draw.rectangle([top_left, bottom_right], fill=(255, 255, 255))
    
    initial_font_size = int(bbox_height * 0.8)
    font, text_width, text_height = adjust_font_size(
        draw, new_name, bbox_width, font_path, initial_font_size
    )
    
    text_x = top_left[0] + 2
    text_y = top_left[1] + 2
    
    draw.text((text_x, text_y), new_name, font=font, fill=(0, 0, 0))
    print(f"Name modified from '{original_name}' to '{new_name}'")
    
    return font


def modify_dob(draw, dob_bbox, original_dob, new_dob, font_path, fallback_font=None):
    """
    Modify the date of birth field in the image.
    
    Args:
        draw: PIL ImageDraw object
        dob_bbox: Bounding box coordinates for the DOB
        original_dob: Original detected DOB text
        new_dob: New DOB to insert
        font_path: Path to the font file
        fallback_font: A font to use if the specified font fails to load
    """
    if not dob_bbox:
        print("No DOB bbox provided, skipping DOB modification")
        return
        
    dob_top_left = (int(dob_bbox[0][0]), int(dob_bbox[0][1]))
    dob_bottom_right = (int(dob_bbox[2][0]), int(dob_bbox[2][1]))
    dob_bbox_width = dob_bottom_right[0] - dob_top_left[0]
    dob_bbox_height = dob_bottom_right[1] - dob_top_left[1]
    
    initial_font_size = int(dob_bbox_height * 0.7)
    
    dob_font, new_dob_width, new_dob_height = adjust_font_size(
        draw, new_dob, dob_bbox_width, font_path, initial_font_size, 10
    )
    
    if dob_font is None and fallback_font is not None:
        dob_font = fallback_font
        new_dob_bbox = draw.textbbox((0, 0), new_dob, font=dob_font)
        new_dob_width = new_dob_bbox[2] - new_dob_bbox[0]
        new_dob_height = new_dob_bbox[3] - new_dob_bbox[1]
    
    dob_text_x = dob_top_left[0] + 10
    dob_text_y = max(dob_top_left[1] - 4, dob_top_left[1] + ((dob_bbox_height - new_dob_height) / 4) - 2)
    
    precise_bg_top_left = (dob_text_x, dob_text_y)
    precise_bg_bottom_right = (dob_text_x + new_dob_width, dob_text_y + new_dob_height + 5)
    
    draw.rectangle([precise_bg_top_left, precise_bg_bottom_right], fill=(255, 255, 255))
    
    draw.text((dob_text_x, dob_text_y), new_dob, font=dob_font, fill=(0, 0, 0))
    print(f"DOB modified from '{original_dob}' to '{new_dob}'")


def modify_aadhar_number(draw, aadhar_bbox, original_aadhar, new_aadhar, font_path, fallback_font=None):
    """
    Modify the Aadhar number field in the image.
    
    Args:
        draw: PIL ImageDraw object
        aadhar_bbox: Bounding box coordinates for the Aadhar number
        original_aadhar: Original detected Aadhar number
        new_aadhar: New Aadhar number to insert
        font_path: Path to the font file
        fallback_font: A font to use if the specified font fails to load
    """
    if not aadhar_bbox:
        print("No Aadhar number bbox provided, skipping Aadhar modification")
        return
        
    aadhar_top_left = (int(aadhar_bbox[0][0]), int(aadhar_bbox[0][1]))
    aadhar_bottom_right = (int(aadhar_bbox[2][0]), int(aadhar_bbox[2][1]))
    aadhar_bbox_width = aadhar_bottom_right[0] - aadhar_top_left[0]
    aadhar_bbox_height = aadhar_bottom_right[1] - aadhar_top_left[1]
    
    draw.rectangle([aadhar_top_left, aadhar_bottom_right], fill=(255, 255, 255))
    
    initial_font_size = int(aadhar_bbox_height * 0.7)
    
    aadhar_font, new_aadhar_width, new_aadhar_height = adjust_font_size(
        draw, new_aadhar, aadhar_bbox_width - 4, font_path, initial_font_size, 10
    )
    
    if aadhar_font is None and fallback_font is not None:
        aadhar_font = fallback_font
        new_aadhar_bbox = draw.textbbox((0, 0), new_aadhar, font=aadhar_font)
        new_aadhar_width = new_aadhar_bbox[2] - new_aadhar_bbox[0]
        new_aadhar_height = new_aadhar_bbox[3] - new_aadhar_bbox[1]
    
    aadhar_text_x = aadhar_top_left[0] + ((aadhar_bbox_width - new_aadhar_width) / 2)
    aadhar_text_y = aadhar_top_left[1] + ((aadhar_bbox_height - new_aadhar_height) / 2)
    
    draw.text((aadhar_text_x, aadhar_text_y), new_aadhar, font=aadhar_font, fill=(0, 0, 0))
    print(f"Aadhar number modified from '{original_aadhar}' to '{new_aadhar}'")


def save_and_display_image(image_pil, output_path="aadharcard_modified.jpg", apply_blur_effect=False, blur_strength=2):
    """
    Convert PIL image back to CV2 format and save.
    
    Args:
        image_pil: PIL Image object
        output_path: Path where to save the modified image
        apply_blur_effect: Whether to apply a blur effect
        blur_strength: Strength of the blur effect (1-5)
        
    Returns:
        Path to the saved image
    """
    if apply_blur_effect:
        image_pil = apply_blur(image_pil, blur_strength)
    
    image_modified = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, image_modified)
    print(f"Modified image saved as: {output_path}")
    
    return output_path


def generate_random_name():
    """
    Generate a random Indian name.
    
    Returns:
        A randomly generated name
    """
    first_names = ["Rahul", "Amit", "Vijay", "Suresh", "Raj", "Anil", "Rakesh", "Ramesh", 
                  "Priya", "Neha", "Pooja", "Sunita", "Anjali", "Meena", "Kavita", "Sanjay", 
                  "Vikram", "Deepak", "Rajesh", "Ravi", "Sunil", "Manoj"]
    
    last_names = ["Kumar", "Singh", "Sharma", "Verma", "Patel", "Shah", "Mehta", 
                 "Gupta", "Joshi", "Desai", "Patil", "Reddy", "Nair", "Rao", "Mishra"]
    
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def generate_random_dob():
    """
    Generate a random date of birth (between 18 and 60 years ago).
    
    Returns:
        A random date in DD/MM/YYYY format
    """
    today = datetime.now()
    min_age = 18
    max_age = 60
    
    days_to_subtract = random.randint(min_age * 365, max_age * 365)
    random_date = today - timedelta(days=days_to_subtract)
    
    return random_date.strftime("%d/%m/%Y")


def generate_random_aadhar():
    """
    Generate a random 12-digit Aadhar number with proper spacing.
    
    Returns:
        A random Aadhar number in XXXX XXXX XXXX format
    """
    digits = [str(random.randint(0, 9)) for _ in range(12)]
    
    aadhar = ''.join(digits[:4]) + ' ' + ''.join(digits[4:8]) + ' ' + ''.join(digits[8:12])
    
    return aadhar


def apply_blur(image_pil, blur_strength=2.0):
    """
    Apply a Gaussian blur effect to the image.
    
    Args:
        image_pil: PIL Image object
        blur_strength: Strength of the blur effect (0.5-5.0, in 0.5 increments)
        
    Returns:
        Blurred PIL Image object
    """
    blur_strength = max(0.5, min(5.0, float(blur_strength)))
    
    blurred_image = image_pil.filter(ImageFilter.GaussianBlur(radius=blur_strength))
    print(f"Applied blur effect with strength {blur_strength}")
    
    return blurred_image


def main(image_path, new_name=None, new_dob=None, new_aadhar=None, output_path="aadharcard_modified.jpg", apply_blur_effect=False, blur_strength=2):
    """
    Main function that orchestrates the Aadhar card modification process.
    
    Args:
        image_path: Path to the original image
        new_name: New name to be inserted (if None, will be randomly generated)
        new_dob: New date of birth to be inserted (if None, will be randomly generated)
        new_aadhar: New Aadhar number to be inserted (if None, will be randomly generated)
        output_path: Path where to save the modified image
        apply_blur_effect: Whether to apply a blur effect to the final image
        blur_strength: Strength of the blur effect (1-5)
        
    Returns:
        Path to the saved image
    """
    image_cv, image_height, image_width = load_image(image_path)
    reader = initialize_ocr()
    
    results = reader.readtext(image_path)
    name_bbox, name_text = detect_name_field(results, image_height)
    dob_bbox, dob_text = detect_dob_field(results)
    aadhar_bbox, aadhar_text = detect_aadhar_number(results)
    
    if name_bbox is None:
        print("No valid Name field detected. Exiting.")
        return None
    
    if new_name is None:
        new_name = generate_random_name()
        print(f"Name not provided, using randomly generated name: {new_name}")
        
    if new_dob is None:
        new_dob = generate_random_dob()
        print(f"DOB not provided, using randomly generated DOB: {new_dob}")
        
    if new_aadhar is None:
        new_aadhar = generate_random_aadhar()
        print(f"Aadhar number not provided, using randomly generated number: {new_aadhar}")
    
    if dob_bbox is None:
        print("Warning: No DOB field detected in image, continuing with name change only.")
    
    if aadhar_bbox is None:
        print("Warning: No Aadhar number detected in image, continuing without changing ID.")
    
    font_path = get_system_font()
    
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    fallback_font = modify_name(draw, name_bbox, name_text, new_name, font_path)

    modify_dob(draw, dob_bbox, dob_text, new_dob, font_path, fallback_font)
    modify_aadhar_number(draw, aadhar_bbox, aadhar_text, new_aadhar, font_path, fallback_font)
    return save_and_display_image(image_pil, output_path, apply_blur_effect, blur_strength)


if __name__ == "__main__":
    IMAGE_PATH = 'original.jpg'
    NEW_NAME = "Vijay Kumar"  
    NEW_DOB = "18/09/1999"  
    NEW_AADHAR = "4276 5933 7812"
    
    main(IMAGE_PATH, NEW_NAME, NEW_DOB, NEW_AADHAR)