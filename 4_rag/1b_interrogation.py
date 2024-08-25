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

# Quelle est la question ?
question_initiale=input("Quelle est la question : ")
if question_initiale == "":
  question_initiale = "Can you detail me the signification of the 'dynamic interpretant' ?"
prompt=question_initiale

response = ollama.embeddings(
  prompt=prompt,
  model="nomic-embed-text"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=5
)

tous_les_chunks_retrieved = ''
color_list = [color.WARNING,color.OKGREEN,color.OKBLUE]
index_couleur=0
for i,data in enumerate(results['documents'][0]):
  couleur = color_list[index_couleur]
  print(f"{couleur}Chunk {i+1} {len(results['documents'][0][i])}\n {results['documents'][0][i]}\n{color.ENDC}")
  tous_les_chunks_retrieved+=f"Chunk {i+1} \n{results['documents'][0][i]}\n\n"
  if index_couleur==2 :
    index_couleur=0
  else :
    index_couleur+=1


#################################################################################
#  Choix du llm
#################################################################################
  
llm_choisi = input("Choisir un llm : '1' local, '2' mistral API : ") 
time_debut = time.time()
match llm_choisi:
  case '1': # Generation with local ollama3.1 ###################################  
    output = ollama.generate(
      model="llama3.1",
      prompt=f"Using this data: {results['documents']}. Respond to this prompt: {prompt}"
    )
    reponse = output['response']
  case '2':
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    chat = ChatMistralAI(api_key=MISTRAL_API_KEY)
    prompt=f"Using this data: {results['documents']}. Respond to this prompt: {prompt}"
    output = chat.invoke(prompt)
    reponse = output.content

print(f"{color.OKCYAN}{reponse}{color.ENDC}")
print(f"Durée de la recherche : {time.time() - time_debut:.1f} secs")

#################################################################################
# Enregistrement de la question, des chunks, et de la réponse dans un fichier txt
#################################################################################

import unicodedata
import re
from datetime import datetime

def create_filename(title):
    # Convert to lowercase
    title = title.lower()
    
    # Remove accents
    title = unicodedata.normalize('NFKD', title).encode('ASCII', 'ignore').decode('ASCII')
    
    # Replace non-alphanumeric characters with underscores
    title = re.sub(r'\W+', '_', title)
    
    # Remove leading or trailing underscores
    title = title.strip('_')
    
    # Get the current date and time
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    
    # Combine the transformed title with the date and time
    final_title = f"{date_time_str}_{title}[20].md"
    return final_title

def save_file(content, title):
    # Get the current directory
    current_dir = os.getcwd()
    
    # Define the target directory
    target_dir = os.path.join(current_dir, '4_rag', 'responses')
    
    # Create the target directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Create the filename
    filename = create_filename(title)
    
    # Define the full file path
    file_path = os.path.join(target_dir, filename)
    
    # Write the content to the file
    with open(file_path, 'w', encoding="utf-8") as file:
        file.write(content)
    
    print(f"File saved to {file_path}")
    
    
# Enregistrement de la question, de la réponse et des chunks dans un fichier txt

filename = create_filename(question_initiale)
print(f"Réponse enregistrée dans {filename} du dossier responses.")
question_response_chunks = f"# {question_initiale}\n({filename})\n\n## {reponse}\n\n## Retrieved chunks\n{tous_les_chunks_retrieved}"
save_file(question_response_chunks, filename)





