
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
file_name = "Peirce_Theory_of_Signs.txt"
file_path = os.path.join(current_dir, "books", file_name)
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
# Attention l'objet documents est une liste d'un seul élément avec deux attributs page_content et metadata (comm d'hab)
#for index in range(len(documents)):
    #print("Le premier document.metadata est :", documents[index].metadata)
    #print("Le premier document.page_content est :",documents[index].page_content)

#################################################################################
# Splitting en utilisant Recursivecharacteretextsplitter
#################################################################################

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
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
texts[0].metadata["source"]=file_name
print("Affiche le premier élément de la liste texts : " , texts[0].metadata)


liste_chunks = [chunk.page_content for chunk in texts]

for index in range(len(texts)):
    texts[index].metadata["source"]=file_name
    texts[index].metadata["idx"]=index
    texts[index].metadata["length"]=len(texts[index].page_content)
    print(f"Affiche l'élément {index} de la liste texts : " , texts[index].metadata, texts[index].page_content)
    print("_________________")