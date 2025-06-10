import os
import pdfplumber
import pytesseract
import camelot
from PIL import Image

# Updated imports from LangChain Community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Imports for Multi-Query and Retrieval QA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI


import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-v4")


# ✅ Define System Prompt for Document Analysis
SYSTEM_PROMPT = """
You are a highly skilled document analysis assistant. Your task is to analyze documents and answer questions strictly using the provided documents.
If the answer is not found in the documents, inform the user instead of making assumptions.

Guidelines:
- Extract key insights and summarize the content.
- Identify important facts, entities, and relationships.
- If structured data (tables, lists) exist, summarize them.
- Maintain a professional, concise, and informative tone.

If an answer is not in the documents, respond with:
"I couldn't find the answer in the provided documents. Would you like me to try answering from general knowledge?"
"""

# ============================
# Extraction Functions
# ============================

# Function to extract tables from PDFs using Camelot
def extract_tables_with_camelot(pdf_path):
    tables_text = ""
    try:
        # Try stream flavor first, then lattice if no tables found
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        if len(tables) == 0:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        for table in tables:
            df = table.df
            tables_text += "\nTable:\n" + df.to_string(index=False) + "\n"
    except Exception as e:
        print(f"Error extracting tables with Camelot: {e}")
    return tables_text

# Function to extract text from images inside PDFs using OCR
def extract_ocr_from_images(pdf_path):
    ocr_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Convert entire page to an image
                pil_img = page.to_image(resolution=300).original.convert("RGB")
                # Apply OCR
                text = pytesseract.image_to_string(pil_img)
                if text.strip():
                    ocr_text += "\nOCR Extracted:\n" + text + "\n"
    except Exception as e:
        print(f"Error in OCR extraction: {e}")
    return ocr_text

# ============================
# Document Processing and Retrieval Setup
# ============================

# Function to load and process PDFs
def load_and_process_pdfs(pdf_files):
    all_documents = []
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        
        # Extract text using PyPDFLoader
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        
        # Extract tables and OCR text
        tables_text = extract_tables_with_camelot(pdf_file)
        ocr_text = extract_ocr_from_images(pdf_file)
        
        # Combine extracted content
        extracted_content = tables_text + "\n" + ocr_text
        if extracted_content.strip():
            # Append as a new Document object
            documents.append(Document(page_content=extracted_content))
        
        # Split the documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        all_documents.extend(split_docs)
    
    return all_documents

# Function to create retrievers using FAISS and BM25
def create_retrievers(docs):
    # Explicitly specify model name for embeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-v4")
    
    # FAISS for dense (semantic) retrieval
    vectorstore = FAISS.from_documents(docs, embeddings)
    # BM25 for keyword-based retrieval
    bm25_retriever = BM25Retriever.from_documents(docs)
    return vectorstore, bm25_retriever

# Function to initialize the LLM (Google Gemini) with a custom system prompt
def initialize_llm():
    return GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        system_message=SYSTEM_PROMPT
    )


# Function to create a multi-query retriever to expand and refine queries
def create_multi_query_retriever(vectorstore, llm):
    return MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

# Function to set up the Retrieval-Augmented Generation (RAG) pipeline
def setup_rag_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # Set to True if you want to access sources; otherwise, set to False
    )

# ============================
# Chatbot Interface
# ============================

# Function to query Gemini with fallback to general knowledge if no document-based answer is found
def ask_gemini_rag(question, rag_chain, llm):
    response = rag_chain.invoke(question)
    answer = response["result"]
    
    if "I couldn't find the answer in the provided documents." in answer:
        print("\n🤖 I couldn't find the answer in the provided documents.")
        fallback = input("Would you like me to try answering from general knowledge? (yes/no): ").strip().lower()
        if fallback == "yes":
            general_answer = llm.invoke(question)
            print("\n🔹 **General Knowledge Answer:**", general_answer)
        else:
            print("\n🤖 Okay, let me know if you need anything else.")
    else:
        print("\n🔹 **Answer:**", answer)
        # If you want to hide source documents from output, do not print them.
        # Otherwise, uncomment the next block to print source excerpts.
        """
        print("\n🔹 **Source Documents:**")
        for doc in response["source_documents"]:
            print(f"🔹 {doc.page_content[:300]}...")
        """

# Chatbot interactive loop
def chatbot(rag_chain, llm):
    print("\n🤖 Chatbot Ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("\n🤖 Goodbye!")
            break
        ask_gemini_rag(query, rag_chain, llm)

# ============================
# Main Function
# ============================
def main():
    # PDF file paths - update with your actual PDF paths
    pdf_files = [
        r"C:\Users\Arjun\Downloads\Tata Chemicals InvestorDeck Q3 FY25.pdf",
    ]
    
    # Load and process PDFs (including table and OCR extraction)
    docs = load_and_process_pdfs(pdf_files)
    
    # Create retrievers
    vectorstore, bm25_retriever = create_retrievers(docs)
    
    # Initialize LLM (Google Gemini)
    llm = initialize_llm()
    
    # Create a multi-query retriever for improved retrieval
    multi_query_retriever = create_multi_query_retriever(vectorstore, llm)
    
    # Set up the RAG pipeline
    rag_chain = setup_rag_chain(llm, multi_query_retriever)
    
    print("Processing complete. Documents are ready for retrieval.")
    
    # Start the interactive chatbot loop
    chatbot(rag_chain, llm)

# Run the script
if __name__ == "__main__":
    main()
