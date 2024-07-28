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
prompt="Can you detail me the signification of 'interpretant' ?"

response = ollama.embeddings(
  prompt=prompt,
  model="nomic-embed-text"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=15
)
datatotal = results['documents']
color_list = [color.WARNING,color.OKGREEN,color.OKBLUE]
index_couleur=0
for i,data in enumerate(results['documents'][0]):
  couleur = color_list[index_couleur]
  print(f"{couleur}Chunk {i+1} {len(results['documents'][0][i])}\n {results['documents'][0][i]}\n{color.ENDC}")
  if index_couleur==2 :
    index_couleur=0
  else :
    index_couleur+=1
    
  
#################################################################################
# Generation of a response combining the prompt and data we retrieved in step 2
#################################################################################
time_debut = time.time()
output = ollama.generate(
  model="llama3.1",
  prompt=f"Using this data: {datatotal}. Respond to this prompt: {prompt}"
)
print(time.time()-time_debut)
print(f"{color.OKCYAN}{output['response']}{color.ENDC}")