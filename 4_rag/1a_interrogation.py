
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
print(persistent_directory)

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

print("Nombre de chunks avec tous les separators", len(texts))
print(type(texts))
#for i in range(len(texts)):
  #print(i, "\n","Longueur du chunk",len(texts[i].page_content),texts[i],"\n\n")


#################################################################################
#  Embedding et storing in chromadb avec mistral et ollama
#################################################################################
import chromadb
import ollama
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# test si la base existe déjà
# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
  print("Persistent directory does not exist. Initializing vector store...")
  print(persistent_directory)
  # Ensure the text file exists
  if not os.path.exists(file_path):
      raise FileNotFoundError(
          f"The file {file_path} does not exist. Please check the path."
      )
  client = chromadb.PersistentClient(path=persistent_directory)
  # ollama_embeddings = OllamaEmbeddingFunction(
  #     model_name="nomic-embed-text",
  #     url="http://localhost:11434/api/embeddings",
  # )
  liste_chunks = [chunk.page_content for chunk in texts]
  time_debut = time.time()
  print(time_debut)
  # Insert embeddings into ChromaDB
  #chroma_db = Chroma.from_documents(documents=liste_chunks, embedding=ollama_embeddings)
  #client = chromadb.Client()
  collection = client.create_collection(name="docs")
  documents=liste_chunks

  #store each document in a vector embedding database
  for i, d in enumerate(documents):
    response = ollama.embeddings(model="nomic-embed-text", prompt=d)
    embedding = response["embedding"]
    collection.add(
      ids=[str(i)],
      embeddings=[embedding],
      documents=[d]
    )
  print(time.time()-time_debut)
  print("Vectore store completed")

#################################################################################
#  Retrieval avec la collection chromadb avec mistral et ollama
#################################################################################

# generate an embedding for the prompt and retrieve the most relevant doc
client = chromadb.PersistentClient(path=persistent_directory)
collection = client.get_collection(name="docs")
prompt="Can detail me the signification of 'interpretant' ?"

response = ollama.embeddings(
  prompt=prompt,
  model="nomic-embed-text"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=15
)
color_list = [color.WARNING,color.OKGREEN,color.OKBLUE]
index_couleur=0
for i,data in enumerate(results['documents'][0]):
  couleur = color_list[index_couleur]
  print(f"{couleur}Chunk {i+1} {len(results['documents'][0][i])}\n {results['documents'][0][i]}\n{color.ENDC}")
  if index_couleur==2 :
    index_couleur=0
  else :
    index_couleur+=1
    
  


#generate a response combining the prompt and data we retrieved in step 2
time_debut = time.time()
output = ollama.generate(
  model="llama3.1",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)
print(time.time()-time_debut)
print(f"{color.OKCYAN}{output['response']}{color.ENDC}")