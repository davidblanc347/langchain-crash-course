import os
import pandas as pd
import fitz  # PyMuPDF
# Identification des path
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "books", "Peirce_collected_papers.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_mistralai")
print(persistent_directory)


# def pdf_to_text(pdf_path, txt_path):
#     # Ouvrir le fichier PDF
#     document = fitz.open(pdf_path)
    
#     # Créer ou ouvrir le fichier texte
#     with open(txt_path, 'w', encoding='utf-8') as text_file:
#         # Parcourir chaque page du PDF
#         for page_num in range(len(document)):
#             page = document.load_page(page_num)
#             text = page.get_text()
#             text_file.write(text)
    
#     print(f"Le contenu du fichier PDF a été enregistré dans {txt_path}")

# # Chemin du fichier PDF et du fichier texte de sortie
# txt_path = os.path.join(current_dir, "books", "Peirce_collected_papers.txt")

# output = pdf_to_text(pdf_path, txt_path)
# print(output)
txt_path = os.path.join(current_dir, "books", "Peirce_collected_papers.txt")
import re
chiffres_premieres_lignes = []
with open(txt_path, "r", encoding='utf-8') as f :# Séparer le texte en morceaux à partir de chaque occurrence de "Peirce: CP"
    texte = f.read()
    morceaux = texte.split("Peirce: CP")
    print("nb morceaux :", morceaux[1200:1202])
    # Enlever les espaces blancs au début et à la fin de chaque morceau
    morceaux = [morceau.strip() for morceau in morceaux if morceau.strip()]
max =0
liste_longueur=[]
liste_volume=[]
liste_chunk=[]
# Afficher la liste
for idx, text in enumerate(morceaux):
    liste_volume.append(text[0:1])
    liste_longueur.append(len(text))
    #print(f"Index {idx}, longueur : {len(text)}, volume : {text[0:1]} ,  idx  et section  {text[0:5]}      \n\n")
print("nb de morceaux", len(morceaux))
#print(liste_volume[0:10])
