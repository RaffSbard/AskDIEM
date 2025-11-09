#!/bin/sh

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