import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
import re

resume_text_list = []

def read_pdf(file_path):
    loader = PyPDFLoader(str(file_path))
    data = loader.load()
    text = "\n".join([doc.page_content for doc in data])
    return text



def read_docx(file_path):
    loader = Docx2txtLoader(str(file_path))
    data = loader.load()
    text = "\n".join([doc.page_content for doc in data])
    return text

def clean_text(text):
    if not text:
        return ""

    # 1. remove non-ascii characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 2. fix spaced letters like "h t t p s"
    text = re.sub(r'(?<=\b\w)\s(?=\w\b)', '', text)

    # 3. fix broken words like "Pr oduct"
    text = re.sub(r'(?<=\w)\s(?=\w)', '', text)

    # 4. restore URLs
    text = re.sub(r'h\s*t\s*t\s*p\s*s?\s*:\s*/\s*/', 'https://', text)

    # 5. normalize emails
    text = re.sub(r'\s*@\s*', '@', text)
    text = re.sub(r'\s*\.\s*', '.', text)

    # 6. normalize phone numbers
    text = re.sub(r'\s*-\s*', '-', text)

    # 7. remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # 8. remove duplicate punctuation
    text = re.sub(r'([.,])\1+', r'\1', text)

    return text.strip()



def process_resume(file_path):
    local_resume_list = [] 

    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".pdf":
        text = read_pdf(file_path)
    elif ext == ".docx":
        text = read_docx(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return None

    cleaned_text = clean_text(text)

    result = {
        "file": os.path.basename(file_path),
        "text": cleaned_text
    }

    
    local_resume_list.append(result)
    
    # Return the dictionary to stage1.py
    return result




 




