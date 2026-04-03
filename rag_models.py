from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from dotenv import load_dotenv
from uuid import uuid4
import os

load_dotenv()  # loads .env file

# LLM model
def llm_model():
    # Gemini LLM
    # model = ChatGoogleGenerativeAI(
    #     model=os.getenv("LLM_MODEL"),   # has quota on your account
    #     google_api_key=os.getenv("LLM_API_KEY")
    # )

    # Ollama LLM
    model = ChatOllama(
        model=os.getenv("LLM_MODEL2"),   # name of the model in Ollama
        temperature=0.1     # conservative model
    )
    return model

# Embeddings
def embedding_model():
    # Ollama embedding model
    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING")) # 768 dimensions
    return embeddings


# Vectorstore
def vector_store_db(embeddings,session_id):
    # Chroma vector_space
    vector_store = Chroma(
        collection_name=f"pdf_collection_{uuid4()}",
        embedding_function=embeddings,
        persist_directory=f"./embeddings_db_{session_id}",  # create local folder for embeddings
    )
    return vector_store