#!/bin/sh

# Funzione per popolare i volumi se sono vuoti
populate_volume() {
    SOURCE_DIR=$1
    TARGET_DIR=$2
    
    # Se la cartella target è vuota (o contiene solo file nascosti)
    if [ -z "$(ls -A $TARGET_DIR)" ]; then
        echo "inizializzazione volume: Copia dati da $SOURCE_DIR a $TARGET_DIR..."
        cp -R $SOURCE_DIR/* $TARGET_DIR/
    else
        echo "Volume $TARGET_DIR già popolato. Salto inizializzazione."
    fi
}

# --- POPOLAMENTO DEI VOLUMI ---
# Copiamo i dati salvati nell'immagine verso i volumi gestiti da Portainer
echo "Controllo stato volumi..."
populate_volume "/app/init_data/data" "/app/data"
populate_volume "/app/init_data/nodes" "/app/nodes"
populate_volume "/app/init_data/urls_lists" "/app/urls_lists"
populate_volume "/app/init_data/snapshots" "/app/snapshots"
# ------------------------------

# 1. Avvia l'app Streamlit in background
echo "Avvio dell'applicazione Streamlit..."
streamlit run app-docker.py --server.port 8501 --server.address 0.0.0.0 &

# 2. Avvia il loop di aggiornamento in PRIMO PIANO
# Questo script terrà il container attivo.
echo "Avvio del processo di aggiornamento periodico (si esegue ora e poi ogni 24 ore)..."
while true; do
    echo "-------------------------------------"
    echo "Esecuzione dello script di aggiornamento (update.py)..."
    # Esegui il tuo script Python
    python update.py
    echo "Aggiornamento completato. In attesa di 24 ore (86400 secondi)..."
    echo "-------------------------------------"
    sleep 86400
done