import json
import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from legal_ingest import extract_text_from_pdfs, extract_text_from_uploaded_file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import glob
from config import (PINECONE_API_KEY, 
                    PINECONE_INDEX_METRIC, 
                    PINECONE_CLOUD, 
                    EMBEDDING_MODEL_NAME, 
                    PINECONE_INDEX_NAME, 
                    PINECONE_ENVIRONMENT)

def embed_uploaded_pdf(uploaded_file, embeddings, vectorstore):
    try:
        text = extract_text_from_uploaded_file(uploaded_file)

        if not text.strip():
            return False, "No text found or extracted from the pdf [:(]"

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        documents = [
            Document(
                page_content = chunk,
                metadata = {
                    "source": uploaded_file.name,
                    "case_name": uploaded_file.name.replace(".pdf", "").replace(".PDF", ""),
                    "upload_type": "user_uploaded"
                }
            ) for chunk in chunks
        ]

        vectorstore.add_documents(documents)
        return True, f"Successfully processed {len(documents)} chunks from {uploaded_file.name}"

    except Exception as e:
        print(f"Error processing uploaded PDF: {str(e)}")
        return False, f"Error processing pdf: {str(e)}"

def load_and_embed_single_pdf(pdf_file_path: str):
    print(f"Loading HuggingFace embedding model: {EMBEDDING_MODEL_NAME}")
    model_kwargs = {"device": 'cuda' if torch.cuda.is_available() else 'cpu'}

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
    )

    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        sample_embedding = embeddings.embed_query("Sample Text")
        embedding_dimension = len(sample_embedding)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dimension,
            metric=PINECONE_INDEX_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_ENVIRONMENT
            )
        )
        print("Waiting for index to be ready...")
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            pass
        print("Index ready.")
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    if not os.path.isfile(pdf_file_path) or not pdf_file_path.lower().endswith('.pdf'):
        print(f"Invalid PDF file path: {pdf_file_path}")
        return None, None

    print(f"Processing PDF file: {os.path.basename(pdf_file_path)}")

    try:
        documents = extract_text_from_pdfs(pdf_file_path)
        if not documents:
            print("No documents extracted from PDF.")
            return None, None

        print(f"Successfully extracted {len(documents)} chunks from: {os.path.basename(pdf_file_path)}")

        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
        )

        vectorstore.add_documents(documents)
        print(f"Success! Ingested {len(documents)} chunks into Pinecone index '{PINECONE_INDEX_NAME}'.")
        print("Your legal knowledge base is now live!")
        return vectorstore, embeddings
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None, None
    
def load_and_embed_documents_from_folder(folder_path: str):
    print("Loading HuggingFace embedding model: {EMBEDDING_MODEL_NAME}")
    model_kwargs = {"device": 'cuda' if torch.cuda.is_available() else 'cpu'}\
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
    )

    print('Initializing Pinecone...')
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print("Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        sample_embedding = embeddings.embed_query("sample text")
        embedding_dimension = len(sample_embedding)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dimension,
            metric=PINECONE_INDEX_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_ENVIRONMENT
            )
        )
        print("Waiting for index to be ready...")
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            pass
        print("Index ready.")
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    if os.path.isfile(folder_path) and folder_path.lower().endswith(".pdf"):
        pdf_files = [folder_path]
    elif os.path.isdir(folder_path):
        pdf_files = glob.glob(os.path.join(folder_path, "*pdf")) + glob.glob(os.path.join(folder_path, "*.PDF"))
    else:
        print(f"Invalid path: {folder_path}")
        return None, None
    
    if not pdf_files:
        print("No PDF files found in the specific path.")
        return None, None
    
    print(f"Found {len(pdf_files)} PDF files to process: ")
    for pdf in pdf_files:
        print(f"   - {os.path.basename(pdf)}")

    all_documents = []
    for pdf_file in pdf_files:
        try:
            documents = extract_text_from_pdfs(pdf_file)
            if documents:
                all_documents.extend(documents)
                print(f"Successfully processed: {os.path.basename(pdf_file)} ({len(documents)} chunks)")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    if not all_documents:
        print("No documents to ingest. Exiting.")
        return None, None
    
    print(f"Ingesting {len(all_documents)} total documents into pinecone...")
    
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )

    try:
        vectorstore.add_documents(all_documents)
        print(f'Success! Ingested {len(all_documents)} chunks into Pinecone Index "{PINECONE_INDEX_NAME}".')
        print("Your legal knowledge base is now live!")
        return vectorstore, embeddings
    except Exception as e:
        print(f"Error ingesting documents: {e}")
        return None, None
