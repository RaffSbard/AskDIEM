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

## 🛠️ Stack Tecnologico e Architettura

L'intero progetto è diviso in due componenti principali: il **Pipeline di Preparazione** e l'**Applicazione Chatbot**.

| Componente | Tecnologia Chiave | Scopo |
| :--- | :--- | :--- |
| **Framework RAG** | [LlamaIndex](https://www.llamaindex.ai/) | Orchestrazione dell'intero flusso RAG: ingestion, indicizzazione, retriever e chat engine. |
| **Applicazione Web** | [Streamlit](https://streamlit.io/) | Creazione e deployment dell'interfaccia utente interattiva. |
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
├── preparation.ipynb        # Notebook Jupyter per l'intera pipeline (dati, nodi, eval)
├── MCE.py                   # Classe custom MainContentExtractor
├── MCER.py                  # Classe custom MainContentExtractorReader
├── requirements-app.txt     # Dipendenze Python per l'app Streamlit
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
│   └── nodes_metadata_hierarchical_x16x4x1.pkl
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

Il progetto richiede due ambienti: uno completo per la preparazione/valutazione dei dati (`requirements-prep.txt`) e uno più leggero per eseguire l'app (`requirements-app.txt`).

### Prerequisiti

  * Python (versione 3.10+)
  * Un file `.env` nella root del progetto con le seguenti chiavi API (vedi `.env.example`):
  
    ```ini
    GOOGLE_API_KEY="AIzaSy..."
    COHERE_API_KEY="..."
    HUGGINGFACE_API_KEY="hf_..."
    QDRANT__API_KEY="..."
    ```

-----

### Parte 1: Preparazione Dati e Valutazione (usando `preparation.ipynb`)

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

-----

### Parte 2: Esecuzione dell'App Chatbot (Locale)

Questo avvia l'interfaccia utente per interagire con l'indice creato nella Parte 1.

1.  **Crea un ambiente virtuale (leggero) e installa le dipendenze:**

    ```bash
    python -m venv venv_app
    source venv_app/bin/activate  # Su Windows: .\venv_app\Scripts\activate
    pip install -r requirements-app.txt
    ```

2.  **Avvia l'app Streamlit:**
    Assicurati che il file `.env` sia presente nella root.

    ```bash
    streamlit run app.py
    ```

3.  Apri il tuo browser all'indirizzo `http://localhost:8501`.

## 🌐 Deployment

La versione pubblica dell'app è deployata su Streamlit Community Cloud. Utilizza il file `app-public.py`, che è identico a `app.py` ma differisce in due punti chiave:

1.  **Non usa `dotenv`**: Non carica il file `.env`.
2.  **Usa `st.secrets`**: Legge le chiavi API direttamente dai segreti di Streamlit Cloud (`GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]`), come richiesto dalla piattaforma.