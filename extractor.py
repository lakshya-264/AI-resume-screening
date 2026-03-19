import os
from pypdf import PdfReader

def extract_text_from_file(file_path: str) -> str:
    """Extracts text from a given valid PDF or TXT file."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    elif ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
            
    else:
        raise ValueError(f"Unsupported file type: {ext}")
