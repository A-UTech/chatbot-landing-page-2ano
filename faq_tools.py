from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_faq_context(pergunta: str) -> str:
    loader = PyPDFLoader(os.getenv("PDF_PATH"))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY, transport="rest")

    db = FAISS.from_documents(chunks, embeddings)

    results = db.similarity_search(pergunta, k=6)
    return results  


if __name__ == "__main__":
    sample_question = "Quais são as habilidades de Rafael Cruz no projeto? E qual é a filosofia da A&U Tech?"
    context_docs = get_faq_context(sample_question)
    for i, doc in enumerate(context_docs):
        print(f"Document {i+1}:\n{doc.page_content}\n")