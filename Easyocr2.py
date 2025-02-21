import cv2
import numpy as np
import easyocr
from pdf2image import convert_from_path
from typing import List, Tuple
import os
from PIL import Image
from fpdf import FPDF
import pytesseract
from skimage import exposure

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
    
    # 6. Final cleaning
    kernel_final = np.ones((2,2), np.uint8)
    final_image = cv2.morphologyEx(edge_enhanced, cv2.MORPH_CLOSE, kernel_final)
    
    return final_image

def check_rotation_needed(image: np.ndarray) -> bool:
    """
    Check if 90-degree rotation is needed based on text orientation
    Uses simple width/height ratio and text detection
    """
    height, width = image.shape[:2]
    
    # If image is portrait (height > width), check for horizontal text
    if height > width:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Use horizontal text detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count non-zero pixels after horizontal detection
        horizontal_pixels = cv2.countNonZero(detected_lines)
        
        # If very few horizontal lines detected, rotation might be needed
        return horizontal_pixels < (width * height * 0.01)
    
    return False

def rotate_90_clockwise(image: np.ndarray) -> np.ndarray:
    """
    Rotate image 90 degrees clockwise
    """
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def create_pdf_page(text: str, page_number: int, output_dir: str) -> str:
    """
    Create a PDF page from extracted text
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Set font
    pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)
    
    # Add text
    pdf.multi_cell(0, 10, text)
    
    # Save page
    output_path = os.path.join(output_dir, f'page_{page_number}.pdf')
    pdf.output(output_path)
    return output_path

def merge_pdfs(pdf_files: List[str], output_path: str):
    """
    Merge multiple PDF files into one
    """
    from PyPDF2 import PdfMerger
    
    merger = PdfMerger()
    for pdf in pdf_files:
        merger.append(pdf)
    
    merger.write(output_path)
    merger.close()

def process_pdf(
    input_pdf: str,
    output_dir: str,
    final_pdf: str,
    languages: List[str] = ['en']
) -> None:
    """
    Process PDF with OCR and create final merged PDF
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize EasyOCR
    reader = easyocr.Reader(
        languages,
        gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False
    )
    
    # Convert PDF to images
    images = convert_from_path(input_pdf)
    pdf_pages = []
    
    # Process each page
    for i, image in enumerate(images):
        print(f"Processing page {i+1}...")
        
        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Check if rotation is needed
        if check_rotation_needed(cv_image):
            cv_image = rotate_90_clockwise(cv_image)
            print(f"Page {i+1} rotated 90 degrees clockwise")
        
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
            sorted_results = sorted(results, key=lambda x: x[0][0][1])
            
            for (bbox, text, conf) in sorted_results:
                if conf > 0.5:  # Filter low confidence results
                    page_text.append(text)
        
        # Create page text with markers
        full_page_text = f"Page {i+1}\n\n" + "\n".join(page_text)
        
        # Create PDF for this page
        pdf_path = create_pdf_page(full_page_text, i+1, output_dir)
        pdf_pages.append(pdf_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        print(f"Page {i+1} completed")
    
    # Merge all PDF pages into final PDF
    merge_pdfs(pdf_pages, final_pdf)
    print(f"Final PDF created: {final_pdf}")
    
    # Clean up individual page PDFs
    for page_pdf in pdf_pages:
        os.remove(page_pdf)

def main():
    # Example usage
    input_pdf = "sample.pdf"
    output_dir = "processed_pages"
    final_pdf = "final_output.pdf"
    languages = ['en']  # Add more languages if needed
    
    try:
        print("Starting PDF processing...")
        process_pdf(input_pdf, output_dir, final_pdf, languages)
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
