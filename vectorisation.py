import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


def charger_documents():

    # Charge les documents préparés depuis le fichier JSON

    with open("documents.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    return documents


def embedder_chunks(chunks, modele):

    # Transforme les textes des films en vecteurs numériques (embeddings) à l'aide du modèle de sentence-transformers
    # Ces vecteurs seront utilisés par FAISS pour la recherche de similarité.

    embeddings = modele.encode(chunks, show_progress_bar=True)

    # Faiss travaille mieux avec des vecteurs en float32
    return np.array(embeddings, dtype=np.float32)

def creer_index_faiss(vecteurs):
    # Crée un index FAISS à partir des vecteurs.

    dimension = vecteurs.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)

    return index

def sauvegarder_index(index, chemin):
    # Sauvegarde l'index FAISS sur le disque.
    # Cela évite de recalculer les embeddings à chaque lancement du RAG.

    faiss.write_index(index, chemin)

if __name__ == "__main__":
    # Charge les documents crées par indexation.py 
    documents = charger_documents()

    print("Nombre de documents :", len(documents))

    # Modèle d'embedding adapté au contenu en anglais

    modele = SentenceTransformer("all-MiniLM-L6-v2")

    # On récupere uniquement le texte de chaque document
    chunks = [doc["content"] for doc in documents]

    # Création des embeddings
    embeddings = embedder_chunks(chunks, modele)

    print("Nombre de vecteurs :", embeddings.shape[0])
    print("Dimension des vecteurs :", embeddings.shape[1])
    print("Shape embeddings :", embeddings.shape)

    # Création de l'index FAISS 
    index = creer_index_faiss(embeddings)

    print("Nombre de vecteurs dans FAISS :", index.ntotal)

    # Sauvegarde de l'index sur le disque

    sauvegarder_index(index, "index.faiss")

    print("Index Faiss sauvegardé avec succès.")

    # Vérification du chargement de l'index

    index_recharge = faiss.read_index("index.faiss")

    print("Index Faiss rechargé avec succès.")
    print("Nombre de vecteurs dans l'index rechargé :", index_recharge.ntotal)