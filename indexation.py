import pandas as pd
import json


def charger_donnees():
     
    #charge le fichier CSV contenant les films TMDB
    df = pd.read_csv("data/tmdb_5000_movies.csv")
    return df

def parse_genres(genres_str):

    #Convertit la colonne genres du CSV.
    #Dans le fichier TMDB, les genres sont stockés au format JSON.
    #Exemple :[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]
    #cette fonction retourne :Action, Adventure

    try:
        genres = json.loads(genres_str)
        return ", ".join([genre["name"] for genre in genres])
    except Exception:
        return ""
    
def chunker(text, taille_max=500, overlap=50):
    
    # Découpe un texte en chunks.
    # Pour ce projet films, chaque document correspond à un seul film.
    
    return [text]

def build_documents(df):

     # Transforme chaque ligne du CSV en document texte exploitable par le RAG.

    documents = []

    for _, row in df.iterrows():
        genres = parse_genres(row["genres"])
        content = f"""
Title: {row["title"]}
Release date: {row["release_date"]}
Overview: {row["overview"]}
Rating: {row["vote_average"]}/10
Original language: {row["original_language"]}
Genres: {genres}
""".strip()

        chunks = chunker(content)

        for i, chunk in enumerate(chunks):
            document = {
                "id": f"{row['id']}",
                "content": chunk,
                "metadata": {
                    "source": "tmdb_5000_movies.csv",
                    "title": row["title"],
                    "release_date": row["release_date"],
                    "rating": row["vote_average"],
                    "original_language": row["original_language"],
                    "genres": genres,
                }
            }

            documents.append(document)

    return documents

def sauvegarder_documents(documents, chemin="documents.json"):

    # Sauvegarde les documents préparés dans un fichier JSON.
    # Ce fichier sera ensuite utilisé pour créer les embeddings et l'index FAISS.
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":

    df = charger_donnees()
    print("Nombre de films :", len(df))

    documents = build_documents(df)
    print("\nNombre de documents créés :", len(documents))

    print("\nPremier document transformé :")
    print(f"\nId : {documents[0]['id']}")

    print("\nContenu du document :\n")
    print(documents[0]["content"])

    print("\nMetadata :\n")
    print(documents[0]["metadata"])
    
    print("\nTest du chunking sur un document :\n")
    test_chunks = chunker(documents[0]["content"])

    for i, chunk in enumerate(test_chunks):
        print(f"\n--- Chunk {i} ---\n")
        print(chunk)
    sauvegarder_documents(documents)
    print("\nFichier documents.json créé avec succès.")