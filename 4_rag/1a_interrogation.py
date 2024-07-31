
# Récupération de la clé Mistral
import os
import time
from colors import color
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI

# Identification des path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "Peirce_Theory_of_Signs.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_mistralai")

print("\n______________________________________________________________\n")
print(f"Persistent directory : {persistent_directory}")

#################################################################################
# Chargement d'un fichier txt
#################################################################################

from langchain_community.document_loaders import TextLoader
loader = TextLoader(file_path, encoding='utf-8')

# La méthode load crée une liste documents de un élément
documents=loader.load()


#################################################################################
# Splitting en utilisant Recursivecharacteretextsplitter
#################################################################################

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
     # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
    # Existing args par défaut la liste des séparateurs
    separators=[ # par défaut
        "\n\n",
        "\n",
    ],
)
texts = text_splitter.create_documents([documents[0].page_content])

print(f"Nombre de chunks avec tous les separators : {len(texts)}")
print(texts[0])


#################################################################################
#  Embedding et storing in chromadb avec mistral et ollama
#################################################################################
import chromadb
import ollama
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# test si la base existe déjà

if not os.path.exists(persistent_directory):
  print("\nPersistent directory does not exist. Initializing vector store...\n")
  # Ensure the text file exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(
          f"The file {file_path} does not exist. Please check the path."
      )
  client = chromadb.PersistentClient(path=persistent_directory)
  print("Création de la base chromadb : chroma_db_mistralai")
  liste_chunks = [chunk.page_content for chunk in texts]

  collection = client.create_collection(name="docs")
  print("Création de la collection : docs")
  documents=liste_chunks

  #store each document in a vector embedding database
  try:
    time_debut = time.time()
    for i, d in enumerate(documents):
      response = ollama.embeddings(model="nomic-embed-text", prompt=d)
      embedding = response["embedding"]
      collection.add(
        ids=[i],
        embeddings=[embedding],
        documents=[d]
      )
    print(f"Durée de l'opération d'embeddings : {time.time() - time_debut:.1f} secs")
    print("Opération terminée")
  except:
    print("Le serveur ollama n'est pas actif")

print("\n______________________________________________________________\n\n")

