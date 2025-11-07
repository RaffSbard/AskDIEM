import os
import time
import requests
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Carica le chiavi API dal tuo file .env locale
load_dotenv()

# --- 1. CONFIGURAZIONE ---
QDRANT_CLOUD_URL = "https://e542824d-6590-4005-91db-6dd34bf8f471.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_CLOUD_API_KEY = os.getenv("QDRANT__API_KEY")
COLLECTION_NAME = "diem_chatbot3_v2"

# Percorso di salvataggio (corrisponde al volume del compose)
LOCAL_SNAPSHOT_DIR = "qdrant_snapshots"
os.makedirs(LOCAL_SNAPSHOT_DIR, exist_ok=True)
SNAPSHOT_FILENAME = "migration_snapshot.snapshot"
SNAPSHOT_FILE_PATH_LOCAL = os.path.join(LOCAL_SNAPSHOT_DIR, SNAPSHOT_FILENAME)

def create_and_download_snapshot():
    """
    Si collega al Qdrant Cloud, crea uno snapshot 
    e lo scarica nella cartella locale 'qdrant_snapshots'.
    """
    
    print(f"Connessione a Qdrant Cloud: {QDRANT_CLOUD_URL}")
    client_cloud = QdrantClient(
        url=QDRANT_CLOUD_URL, 
        api_key=QDRANT_CLOUD_API_KEY
    )
    
    try:
        print(f"Creazione snapshot per '{COLLECTION_NAME}'...")
        snapshot = client_cloud.create_snapshot(collection_name=COLLECTION_NAME, wait=True)
        snapshot_name = snapshot.name
        print(f"Snapshot '{snapshot_name}' creato sul cloud.")
        
        print(f"Download snapshot '{snapshot_name}' in corso...")
        snapshot_url = f"{QDRANT_CLOUD_URL}/collections/{COLLECTION_NAME}/snapshots/{snapshot_name}"
        headers = {"api-key": QDRANT_CLOUD_API_KEY}
        
        response = requests.get(snapshot_url, headers=headers, stream=True)
        response.raise_for_status() 
        
        with open(SNAPSHOT_FILE_PATH_LOCAL, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)
        
        print(f"Download completato!")
        print(f"Snapshot salvato in: {SNAPSHOT_FILE_PATH_LOCAL}")

    except Exception as e:
        print(f"ERRORE durante il download dal cloud: {e}")

if __name__ == "__main__":
    if os.path.exists(SNAPSHOT_FILE_PATH_LOCAL):
        print(f"Il file di snapshot '{SNAPSHOT_FILE_PATH_LOCAL}' esiste già.")
        risposta = input("Vuoi scaricarlo di nuovo (sovrascrivendolo)? (s/n): ")
        if risposta.lower() == 's':
            create_and_download_snapshot()
        else:
            print("Operazione annullata.")
    else:
        create_and_download_snapshot()