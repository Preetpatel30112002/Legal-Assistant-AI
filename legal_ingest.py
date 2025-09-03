import os
import fitz
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tempfile
import pdfplumber

def extract_text_from_pdfs(pdf_path: str):
    print(f"[INFO] Extracting and chunking PDFs from: {pdf_path}")
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    if pdf_path.lower().endswith(".pdf"):
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text("text") + "\n"
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(Document(
                page_content = chunk,
                metadata = {
                    "source": os.path.basename(pdf_path),
                    "case_name": os.path.basename(pdf_path).replace(".PDF", "").replace("_", " ")
                }
            ))
        
    print(f"[info] Total chunks extracted: {len(all_chunks)}")
    return all_chunks

def extract_text_from_uploaded_file(uploaded_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    try:
        with pdfplumber.open(tmp_file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    finally:
        os.unlink(tmp_file_path)

    return text