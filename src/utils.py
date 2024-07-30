# src/utils.py
import os
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using Poppler and Tesseract OCR.
    
    Args:
        pdf_path (str): The path to the PDF file.
    
    Returns:
        str: The extracted text.
    """
    # Convert PDF to images
    images = convert_from_path(pdf_path, poppler_path='/path/to/poppler/bin')
    
    # Extract text from images
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    
    return text
