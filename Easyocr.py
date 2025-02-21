import cv2
import numpy as np
import easyocr
from pdf2image import convert_from_path
import math
from typing import List, Tuple
import os
from scipy import ndimage
from skimage import exposure, filters, morphology

def advanced_preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Advanced image preprocessing for better OCR results
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 1. Initial denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # 2. Adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # 3. Binarization using Otsu's method
    _, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Remove small noise using morphological operations
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 5. Edge enhancement
    edge_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    edge_enhanced = cv2.filter2D(cleaned, -1, edge_kernel)
    
    # 6. Deskew text lines
    coords = np.column_stack(np.where(edge_enhanced > 0))
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    deskewed = ndimage.rotate(edge_enhanced, angle)
    
    # 7. Remove borders if present
    contours, _ = cv2.findContours(deskewed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_content = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(deskewed)
        cv2.drawContours(mask, [main_content], -1, (255), thickness=cv2.FILLED)
        deskewed = cv2.bitwise_and(deskewed, mask)
    
    # 8. Final cleaning
    kernel_final = np.ones((2,2), np.uint8)
    final_image = cv2.morphologyEx(deskewed, cv2.MORPH_CLOSE, kernel_final)
    
    return final_image

def detect_rotation_angle(image: np.ndarray) -> float:
    """
    Advanced rotation detection using text line analysis
    """
    # Convert to binary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find text contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours
            rect = cv2.minAreaRect(contour)
            angle = rect[-1]
            
            # Normalize angle
            if angle < -45:
                angle = 90 + angle
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Use the most common angle
    angle_counts = np.bincount(np.array(angles, dtype=int))
    most_common_angle = float(np.argmax(angle_counts))
    
    return most_common_angle

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image with border handling
    """
    # Get image dimensions
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Calculate new dimensions
    radians = math.radians(abs(angle))
    new_width = int(width * abs(math.cos(radians)) + height * abs(math.sin(radians)))
    new_height = int(width * abs(math.sin(radians)) + height * abs(math.cos(radians)))
    
    # Adjust rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotation_matrix[0, 2] += (new_width - width) // 2
    rotation_matrix[1, 2] += (new_height - height) // 2
    
    # Perform rotation with border replication
    rotated = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated

def extract_text_from_pdf(
    pdf_path: str,
    output_path: str,
    languages: List[str] = ['en']
) -> List[str]:
    """
    Extract text from PDF using EasyOCR with advanced preprocessing
    """
    # Initialize EasyOCR
    reader = easyocr.Reader(languages, gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    
    # Process each page
    all_text = []
    
    for i, image in enumerate(images):
        print(f"Processing page {i+1}...")
        
        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect and correct rotation
        angle = detect_rotation_angle(cv_image)
        if abs(angle) > 0.5:
            cv_image = rotate_image(cv_image, angle)
        
        # Apply advanced preprocessing
        processed_image = advanced_preprocess_image(cv_image)
        
        # Save temporary processed image
        temp_path = f'temp_processed_page_{i}.jpg'
        cv2.imwrite(temp_path, processed_image)
        
        # Perform OCR with confidence filtering
        results = reader.readtext(temp_path)
        
        # Extract text with position information
        page_text = []
        if results:
            # Sort results by vertical position (top to bottom)
            sorted_results = sorted(results, key=lambda x: x[0][0][1])  # Sort by y-coordinate
            
            for (bbox, text, conf) in sorted_results:
                if conf > 0.5:  # Filter low confidence results
                    page_text.append(text)
        
        # Join text and add page marker
        full_page_text = f"\n=== Page {i+1} ===\n" + "\n".join(page_text)
        all_text.append(full_page_text)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        print(f"Page {i+1} completed")
    
    # Save all text to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_text))
    
    return all_text

def main():
    # Example usage
    pdf_path = "sample.pdf"
    output_path = "output_text.txt"
    languages = ['en']  # Add more languages if needed, e.g., ['en', 'fr', 'de']
    
    try:
        print("Starting OCR processing...")
        extracted_text = extract_text_from_pdf(pdf_path, output_path, languages)
        print(f"OCR completed successfully. Output saved to: {output_path}")
        
        # Print first few lines of each page as preview
        for page_text in extracted_text:
            preview = "\n".join(page_text.split("\n")[:5])
            print(f"\nPreview:{preview}\n...")
            
    except Exception as e:
        print(f"Error during OCR processing: {str(e)}")

if __name__ == "__main__":
    main()
