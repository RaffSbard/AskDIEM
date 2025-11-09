# 🤖 AskDIEM: Assistente Conversazionale RAG per il DIEM

**AskDIEM** è un assistente conversazionale basato su Large Language Models (LLM) e Retrieval-Augmented Generation (RAG), sviluppato come progetto di tesi magistrale. Il suo scopo è fornire agli studenti del **Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica Applicata (DIEM)** dell'Università di Salerno un accesso intuitivo, rapido e affidabile alle informazioni accademiche e amministrative, basandosi esclusivamente su documenti e pagine web ufficiali.

L'applicazione è accessibile pubblicamente (per la versione deployata) ed è stata progettata per essere bilingue (Italiano/Inglese) e completamente responsiva.

## ✨ Caratteristiche Principali

  * **Architettura RAG:** Fornisce risposte basate su un contesto recuperato da documenti ufficiali, riducendo drasticamente il rischio di "allucinazioni" (informazioni inventate).
  * **Interfaccia Chat Intuitiva:** Un'applicazione [Streamlit](https://streamlit.io/) che permette agli utenti di porre domande in linguaggio naturale.
  * **Supporto Bilingue:** Rileva la lingua della domanda (Italiano o Inglese) e risponde di conseguenza, adattando anche l'interfaccia.
  * **Trasparenza delle Fonti:** Mostra sempre un elenco puntato degli URL specifici utilizzati per generare la risposta, permettendo all'utente di verificare l'informazione all'origine.
  * **Consapevolezza Temporale:** Il system prompt viene aggiornato dinamicamente con la data corrente (`{current_date}`) per fornire risposte contestualizzate al periodo accademico (es. sessioni d'esame, scadenze).
  * **Pipeline di Dati Completo:** Include un notebook (`preparation.ipynb`) per l'intero ciclo di vita dei dati: crawling, estrazione, arricchimento con LLM, indicizzazione e valutazione rigorosa.
  * **Containerizzazione Docker:** L'intera applicazione (Streamlit UI + script di aggiornamento in background) è containerizzata per un deployment semplice e multipiattaforma.

## 🛠️ Stack Tecnologico e Architettura

L'intero progetto è diviso in due componenti principali: il **Pipeline di Preparazione** e l'**Applicazione Chatbot**.

| Componente | Tecnologia Chiave | Scopo |
| :--- | :--- | :--- |
| **Framework RAG** | [LlamaIndex](https://www.llamaindex.ai/) | Orchestrazione dell'intero flusso RAG: ingestion, indicizzazione, retriever e chat engine. |
| **Applicazione Web** | [Streamlit](https://streamlit.io/) | Creazione e deployment dell'interfaccia utente interattiva. |
| **Container** | [Docker](https://www.docker.com/) | Containerizzazione dell'app, dello script di aggiornamento e del database per il deployment. |
| **Modelli LLM** | Google Gemini 2.5 Flash e Flash Lite | Usati rispettivamente per la generazione delle risposte (`app.py`) e l'arricchimento dei metadati (`preparation.ipynb`). |
| **Modello di Embedding** | BAAI/bge-m3 | Trasforma i documenti in vettori. Usato localmente (`preparation.ipynb`) e tramite API HuggingFace nel deployment (`app-public.py`). |
| **Vector Database** | [Qdrant Cloud](https://qdrant.tech/) | Database vettoriale cloud per l'archiviazione e la ricerca ad alta velocità dei nodi. |
| **Reranker** | Cohere ReRank | Modello di post-processing che riordina i nodi recuperati per massimizzare la pertinenza prima di inviarli all'LLM. |
| **Data Crawling** | Selenium, BeautifulSoup | Usati per navigare il sito DIEM, gestire contenuti dinamici (Javascript) ed estrarre l'HTML. |
| **Estrazione Contenuti** | `MCER.py`, `pypdf` | Classi personalizzate per estrarre il contenuto principale (testo e link) dalle pagine HTML e PDF. |
| **Valutazione** | LlamaIndex Eval, Pandas, Seaborn | Utilizzati per generare dataset Q&A sintetici e valutare rigorosamente le prestazioni del sistema (Response e Retrieval). |

## 📁 Struttura del Progetto

```
.
├── app.py                   # App Streamlit per esecuzione locale (usa .env)
├── app-public.py            # App Streamlit per deployment (usa st.secrets)
├── app-docker.py            # App Streamlit per esecuzione nel container Docker (usa QDrant su Server)
├── preparation.ipynb        # Notebook Jupyter per l'intera pipeline (dati, nodi, eval)
├── MCE.py                   # Classe custom MainContentExtractor
├── MCER.py                  # Classe custom MainContentExtractorReader
├── migrate.py               # Script per scaricare lo snapshot da Qdrant Cloud
├── update.py                # Script per l'aggiornamento del vector store
│
├── Dockerfile               # Istruzioni per costruire l'immagine dell'app
├── entrypoint.sh            # Script di avvio per il container dell'app
├── docker-compose.yml       # File per avviare l'ambiente di sviluppo Docker
├── docker-compose.prod.yml  # File per avviare l'ambiente di produzione Docker
│
├── requirements-app.txt     # Dipendenze Python per l'app Streamlit
├── requirements-docker.txt  # Dipendenze Python per il container docker
├── requirements-prep.txt    # Dipendenze Python per il notebook di preparazione
├── requirements.txt         # Dipendenze Python per il deployment
├── askdiem.png              # Logo
├── .env.example             # Template per le chiavi API
│
├── data/                    # Dati generati e di stato
│   ├── extracted_metadata.json
│   ├── generated_rag_answers.json
│   └── page_update_state.json
│
├── documents/               # Documenti HTML pre-processati
│   └── processed_documents_final.pkl
│
├── nodes/                   # Nodi (chunk) pronti per l'indicizzazione
│   ├── nodes_metadata_sentence_x8.pkl
│   ├── nodes_metadata_sentence_x16.pkl
│   ├── nodes_metadata_hierarchical_x8x2x1.pkl
│   ├── nodes_metadata_hierarchical_x16x4x1.pkl
│   └── nodes_metadata_update.pkl
│
├── qdrant_snapshots/        # Snapshot dei vector store
│   ├── tmp/
│   │    └── upload/
│   └── migration_snapshots.snapshots
│
├── results/                 # Risultati della valutazione
│   ├── response/
│   ├── retrieval/
│   └── nodes/
│
└── urls_lists/              # Liste di URL indicizzati
    ├── urls_html_master_list.txt
    ├── urls_pdf_master_list.txt
    └── urls_pdf_downloaded_list.txt
```

## 🚀 Setup e Installazione

Ci sono due modi per eseguire il progetto: tramite Docker o manualmente in ambienti virtuali Python.

---

### Metodo 1: Esecuzione con Docker

Questo è il metodo più semplice e affidabile per eseguire l'intera applicazione (interfaccia web e script di aggiornamento).

**Prerequisiti:**
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installato e in esecuzione.
* Aver clonato questo repository.

#### 1. Creare il file `.env`
Copia `.env.example` in un nuovo file chiamato `.env` e inserisci le tue chiavi API.

```bash
cp .env.example .env
# Ora modifica il file .env con le tue chiavi
```

#### 2. Scaricare lo Snapshot del Database

Il database Qdrant nel container parte vuoto. Dobbiamo "riempirlo" con i dati vettoriali scaricando uno snapshot dal Qdrant Cloud (dove i dati sono stati preparati tramite `preparation.ipynb`).

  * **(Se necessario) Installa i requisiti per lo script di migrazione:**

    ```bash
    pip install qdrant-client python-dotenv requests
    ```

  * **Esegui lo script `migrate.py`:**

    ```bash
    python migrate.py
    ```

    Questo scaricherà `migration_snapshot.snapshot` nella cartella `qdrant_snapshots/`. Lo script `update.py` all'interno del container lo rileverà al primo avvio e ripristinerà il database.

#### 3. Avviare Docker Compose

Hai due opzioni:

  * **A. Per lo Sviluppo (Build Locale):**
    Questo comando costruisce l'immagine Docker localmente usando i tuoi file sorgente. Usalo se hai modificato il codice (`app-docker.py`, `update.py`, ecc.).

    ```bash
    docker compose up --build -d
    ```

  * **B. Per la Produzione (Immagine Pubblica):**
    Questo comando scarica l'immagine `rafsbard/askdiem_app:latest` già pronta da Docker Hub. È più veloce e non richiede la build.

    ```bash
    docker compose -f docker-compose.prod.yml up -d
    ```

#### 4. Accedere all'App

Una volta avviati i container, apri il tuo browser e vai su:
**`http://localhost:8501`**

#### 5. Fermare l'Applicazione

Per fermare ed eliminare i container e il volume del database:

```bash
docker compose down -v
```

-----

### Metodo 2: Setup Manuale (Senza Docker)

Questo metodo è utile per eseguire il notebook di preparazione o per testare l'app Streamlit in un ambiente Python locale. Richiede due ambienti: uno completo per la preparazione/valutazione dei dati (`requirements-prep.txt`) e uno più leggero per eseguire l'app (`requirements-app.txt`).

### Prerequisiti

  * Python (versione 3.10+)
  * Un file `.env` nella root del progetto con le seguenti chiavi API (vedi `.env.example`):
  
    ```ini
    GOOGLE_API_KEY="AIzaSy..."
    COHERE_API_KEY="..."
    HUGGINGFACE_API_KEY="hf_..."
    QDRANT__API_KEY="..."
    ```

#### Parte 1: Preparazione Dati e Valutazione (usando `preparation.ipynb`)

Questo notebook è il "cervello" del progetto. Gestisce l'intero ciclo di vita dei dati.

1.  **Crea un ambiente virtuale e installa le dipendenze:**

    ```bash
    python -m venv venv_prep
    source venv_prep/bin/activate  # Su Windows: .\venv_prep\Scripts\activate
    pip install -r requirements-prep.txt
    pip install ipykernel
    ```

2.  **Esegui le celle in `preparation.ipynb`:**
    Il notebook è diviso in sezioni che eseguono le seguenti operazioni:

      * **FASE 1-2 (Crawler):** Esegue il crawling dei siti web DIEM usando `requests` e `Selenium` (per pagine dinamiche). Salva la lista degli URL in `urls_lists/`.
      * **FASE 3 (Estrazione):** Carica gli URL, estrae il contenuto principale usando `MCER.py` e `pypdf`, e salva i `Document` LlamaIndex in `documents/processed_documents.pkl`.
      * **FASE 4 (Arricchimento):** Itera sui documenti, chiama Gemini per estrarre metadati (titolo, sommario, parole chiave) e li salva in `data/extracted_metadata.json`.
      * **FASE 5 (Creazione Nodi):** Applica le strategie di *chunking* (es. `SentenceSplitter`) ai documenti arricchiti e salva i nodi finali in `nodes/nodes_final_enriched.pkl`.
      * **FASE 6 (Indicizzazione):** Carica i nodi e li indicizza sul database vettoriale Qdrant Cloud. (Se questa fase risulta troppo lenta caricare il notebook su ambienti online (es. Google Colab / Kaggle) e usufruire della GPU offerta)
      * **FASE 7 (Valutazione):** Genera dataset Q\&A sintetici (`data/generated_rag_answers.json`) e valuta rigorosamente il pipeline RAG, salvando i report in `results/`.

    In alternativa, per le fasi 1-6, puoi anche eseguire unicamente il **main_workflow**.

#### Parte 2: Esecuzione dell'App Chatbot (Locale)

Questo avvia l'interfaccia utente per interagire con l'indice **Qdrant Cloud**.

1.  **Crea un ambiente virtuale (leggero) e installa le dipendenze:**

    ```bash
    python -m venv venv_app
    source venv_app/bin/activate  # Su Windows: .\venv_app\Scripts\activate
    pip install -r requirements-app.txt
    ```

2.  **Avvia l'app Streamlit:**
    Assicurati che il file `.env` sia presente.

    ```bash
    streamlit run app.py
    ```

3.  Apri il tuo browser all'indirizzo `http://localhost:8501`.


## 🌐 Deployment

La versione pubblica dell'app è deployata su Streamlit Community Cloud. Utilizza il file `app-public.py`, che è identico a `app.py` ma differisce in due punti chiave:

1.  **Non usa `dotenv`**: Non carica il file `.env`.
2.  **Usa `st.secrets`**: Legge le chiavi API direttamente dai segreti di Streamlit Cloud (`GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]`), come richiesto dalla piattaforma.