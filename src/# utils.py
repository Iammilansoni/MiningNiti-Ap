# utils.py
import os
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_pdf(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path, 500)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text
