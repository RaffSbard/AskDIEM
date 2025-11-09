# Usa un'immagine Python ufficiale
FROM python:3.11-slim

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Installa le dipendenze di sistema (necessarie per Selenium/WebDriver)
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    # Pulisci la cache per ridurre la dimensione dell'immagine
    && rm -rf /var/lib/apt/lists/*

# Copia i file dei requisiti
# Usiamo 'requirements-docker.txt' perché contiene TUTTE le dipendenze (app + updater)
COPY requirements-docker.txt .

# Installa TUTTE le dipendenze
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copia tutto il resto del codice sorgente nell'immagine
COPY . .

# Crea i file di log/stato vuoti se non esistono (necessario per il volume)
RUN mkdir -p data nodes urls_lists
RUN touch data/page_update_state.json
RUN touch urls_lists/urls_html_master_list.txt
RUN touch urls_lists/urls_pdf_master_list.txt
RUN touch urls_lists/urls_pdf_downloaded_list.txt
RUN touch nodes/nodes_final_enriched.pkl

# Crea ed rendi eseguibile lo script di avvio
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Esponi la porta 8501 (per Streamlit)
EXPOSE 8501

# Definisci il comando che avvierà il container
ENTRYPOINT ["./entrypoint.sh"]