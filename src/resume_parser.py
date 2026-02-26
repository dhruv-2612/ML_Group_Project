import PyPDF2
import io
import logging
from typing import List, Dict, Any
from src.utils import extract_skills, extract_education

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_stream) -> str:
    """Extracts text from a PDF file stream."""
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def parse_resume(file_obj, file_type: str) -> Dict[str, Any]:
    """
    Parses a resume file (PDF/TXT) and extracts skills and basic info.
    Returns: Dictionary with 'text', 'skills', 'education', and inferred 'role'.
    """
    text = ""
    try:
        if file_type == "application/pdf" or file_obj.name.endswith(".pdf"):
            text = extract_text_from_pdf(file_obj)
        elif file_type == "text/plain" or file_obj.name.endswith(".txt"):
            text = file_obj.read().decode("utf-8")
        else:
            logger.warning("Unsupported file type")
            return {}
            
        if not text:
            return {}
            
        # Extract Skills using our centralized skill extractor
        detected_skills = extract_skills(text)
        detected_edu = extract_education(text)
        
        return {
            "text": text,
            "skills": detected_skills,
            "education": detected_edu,
            "char_count": len(text)
        }
        
    except Exception as e:
        logger.error(f"Resume parsing failed: {e}")
        return {}
