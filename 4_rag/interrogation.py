
# Récupération de la clé Istral
import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI
#print(MISTRAL_API_KEY)

# Identification des path
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "Peirce_Theory_of_Signs.txt")
#print(current_dir)
#print(file_path)

# Chargement d'un fichier txt
from langchain_community.document_loaders import TextLoader
loader = TextLoader(file_path, encoding='utf-8')

# La méthode load crée une liste documents de un élément
documents=loader.load()


# Splitting  documents
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
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
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "", #875
    ],
)
texts = text_splitter.create_documents([documents[0].page_content])

print("Nombre de chunks avec aucun separator", len(texts))

# Splitting en utilisant Recursivecharacteretextsplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
     # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n", # 179 chunks
    ],
    # Existing args
)
texts = text_splitter.create_documents([documents[0].page_content])

print("Nombre de chunks avec separator", len(texts))
print(type(texts))
print(texts[0].page_content)

# Embedding avec mistral et ollama
import chromadb
import ollama


client = chromadb.Client()
collection = client.create_collection(name="docs")

text_to_embed=[texts[index].page_content for index in range(len(texts))]
print(type(text_to_embed))
print(len(text_to_embed))

# store each document in a vector embedding database
for i, d in enumerate(texts):
  print (i, d)
  response = ollama.embeddings(model="mxbai-embed-large", prompt=d.page_content)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    texts=[d]
  )

print(type(embeddings))
print(len(embeddings))



