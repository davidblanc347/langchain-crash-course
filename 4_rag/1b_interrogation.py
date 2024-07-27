import os
import chromadb
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "Peirce_Theory_of_Signs.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_mistrl=al")

client = chromadb.PersistentClient(path=persistent_directory)
collection = client.get_collection(
    name="docs",
    embedding_function="nomic-embed-text"
    )

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt="qui est cs peirce",
  model="nomic-embed-text"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)
data = results['documents'][0][0]

print(data)