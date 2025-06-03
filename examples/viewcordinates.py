import easyocr
import cv2
import re
import numpy as np

def load_image(image_path):
    """
    Load the image and prepare it for processing.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        cv2 image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    return image


def initialize_ocr():
    """Initialize the OCR reader."""
    print("Initializing OCR reader...")
    return easyocr.Reader(['en'], gpu=False)


def detect_and_annotate_date(image, ocr_results):
    """
    Detect date in the format DD/MM/YYYY and annotate it with a bounding box.
    
    Args:
        image: Original image
        ocr_results: OCR results with bounding boxes
    
    Returns:
        Annotated image with date bounding box
    """
    date_pattern = r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b'
    
    for (bbox, text, prob) in ocr_results:
        print(f"Detected text: {text} with probability: {prob}")
        
        date_text = re.sub(r'.*DOB[:\-]?\s*', '', text).strip()
        
        date_matches = re.match(date_pattern, date_text)
        
        if date_matches:
            pts = [tuple(b) for b in bbox]
            cv2.polylines(image, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

            text_position = (int(bbox[0][0]), int(bbox[0][1]) - 10)  
            cv2.putText(image, date_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            print(f"Date detected: {date_text}")
    
    return image


def save_annotated_image(image, output_path="annotated_image.jpg"):
    """
    Save the annotated image with bounding boxes.
    
    Args:
        image: Annotated image
        output_path: Path where the modified image will be saved
    """
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved as: {output_path}")


def main(image_path, output_path="annotated_image.jpg"):
    """
    Main function to load image, detect date, annotate bounding boxes, and save image.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image
    """
    image = load_image(image_path)

    reader = initialize_ocr()

    ocr_results = reader.readtext(image_path)

    print(f"OCR Results: {ocr_results}")
    
    annotated_image = detect_and_annotate_date(image, ocr_results)

    save_annotated_image(annotated_image, output_path)


if __name__ == "__main__":
    IMAGE_PATH = 'original.jpg'  
    OUTPUT_PATH = 'output_image.jpg' 
    main(IMAGE_PATH, OUTPUT_PATH)
