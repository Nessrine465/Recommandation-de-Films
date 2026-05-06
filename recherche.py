import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def charger_documents():

    # Charge les documents préparés depuis le fichier JSON

    with open("documents.json", "r", encoding="utf-8") as f:
        return json.load(f)


def charger_index():

    # Recharge l'index FAISS sauvegardé sur le disque

    return faiss.read_index("index.faiss")


def rechercher(question, modele, index, documents, k=5):

    # Transforme la question utilisateur en embedding

    question_embedding = modele.encode([question])
    question_embedding = np.array(question_embedding, dtype=np.float32)

    # Recherche des k vecteurs les plus proches dans FAISS

    distances, indices = index.search(question_embedding, k)

    resultats = []

    # Parcourt les résultats trouvés

    for distance, indice in zip(distances[0], indices[0]):

        # Filtre les résultats peu pertinents
        # Plus la distance est faible, plus le résultat est similaire

        if distance > 1.2:
            continue

        document = documents[indice]

        resultats.append({
            "score": float(distance),
            "content": document["content"],
            "metadata": document["metadata"]
        })

    return resultats

if __name__ == "__main__":

    # Charge les documents et l'index FAISS

    documents = charger_documents()
    index = charger_index()

    # Charge le même modèle utilisé pendant l'indexation

    modele = SentenceTransformer("all-MiniLM-L6-v2")

    # Question de test 

    question =  "romentic movies"

    # Lancer la recherche vectorielle

    resultats = rechercher(question, modele, index, documents, k=5)

    print("\nQuestion :", question)
    print("\nRésultats trouvés :\n")

    # Affiche les résultats trouvés

    for i, resultat in enumerate(resultats, start=1):
        print(f"--- Résultat {i} ---")
        print("Score :", resultat["score"])
        print("Title :", resultat["metadata"]["title"])
        print("Genres :", resultat["metadata"]["genres"])
        print("Rating :", resultat["metadata"]["rating"])
        print("Release date :", resultat["metadata"]["release_date"])
        print() 