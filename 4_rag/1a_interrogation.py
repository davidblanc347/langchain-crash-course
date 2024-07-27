
# Récupération de la clé Mistral
import os
import time
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

# Chargement d'un fichier txt
from langchain_community.document_loaders import TextLoader
loader = TextLoader(file_path, encoding='utf-8')

# La méthode load crée une liste documents de un élément
documents=loader.load()


# Splitting  documents
# from langchain_text_splitters import CharacterTextSplitter
# text_splitter = CharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=1000,
#     chunk_overlap=100,
#     length_function=len,
#     is_separator_regex=False,
# )
#texts = text_splitter.split_documents(documents)
# on affiche le premier éléemnt de liste 

#print("nombre de chunks", len(texts))


# Splitting en utilisant Recursivecharacteretextsplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
     # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
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
for i in range(len(texts)):
  print(i, "\n","Longueur du chunk",len(texts[i].page_content),texts[i],"\n\n")

# Embedding avec mistral et ollama
import chromadb
import ollama
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

client = chromadb.PersistentClient(path=persistent_directory)
# ollama_embeddings = OllamaEmbeddingFunction(
#     model_name="nomic-embed-text",
#     url="http://localhost:11434/api/embeddings",
# )
liste_chunks = [chunk.page_content for chunk in texts]
time_debut = time.time()
# Insert embeddings into ChromaDB
#chroma_db = Chroma.from_documents(documents=liste_chunks, embedding=ollama_embeddings)
#client = chromadb.Client()
collection = client.get_collection(name="docs")
# documents=liste_chunks

# # store each document in a vector embedding database
# for i, d in enumerate(documents):
#   response = ollama.embeddings(model="nomic-embed-text", prompt=d)
#   embedding = response["embedding"]
#   collection.add(
#     ids=[str(i)],
#     embeddings=[embedding],
#     documents=[d]
#   )
# print(time.time()-time_debut)



# generate an embedding for the prompt and retrieve the most relevant doc
prompt="can you sumarize the first chapter ?"

response = ollama.embeddings(
  prompt=prompt,
  model="nomic-embed-text"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=10
)
for i,data in enumerate(results['documents'][0]):
  print("Chunk ",i+1,"\n",results['documents'][0][i],"\n")



# generate a response combining the prompt and data we retrieved in step 2
# output = ollama.generate(
#   model="llama3.1",
#   prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
# )

# print(output['response'])