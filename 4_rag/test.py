# Récupération de la clé Mistral
import os
import time
from colors import color
import chromadb
import ollama
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "Peirce_Theory_of_Signs.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_mistralai")

#################################################################################
#  Retrieval avec la collection chromadb avec mistral et ollama
#################################################################################

# generate an embedding for the prompt and retrieve the most relevant doc
client = chromadb.PersistentClient(path=persistent_directory)
collection = client.get_collection(name="docs")

#print(collection.peek())
result = collection.get()

print(result['metadatas'])
print(len(result['metadatas']))
print(result.keys())