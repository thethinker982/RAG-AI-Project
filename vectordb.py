from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

pdfs_directory = "pdfs/"

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def create_chunks(documents):
    """
    Split the document by 'Article' so that each article
    is stored as a separate chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\nArticle", "Article"],
        chunk_size=1500,
        chunk_overlap=100,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

file_path = "udhv.pdf"
documents = load_pdf(file_path)
text_chunks = create_chunks(documents)

ollama_model_name = "deepseek-r1:1.5b"

def get_embedding_model(model_name):
    return OllamaEmbeddings(model=model_name)

FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)

