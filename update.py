# ==============================================================================
# --- SEZIONE 0: IMPORTAZIONI E CONFIGURAZIONE GLOBALE ---
# ==============================================================================
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from collections import deque
import re
import json
import hashlib
import os
from dotenv import load_dotenv
import pickle
import random
from tqdm import tqdm
import io
from pypdf import PdfReader

# Import per Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# Import per LlamaIndex
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import per Qdrant
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from MCER import MainContentExtractorReader
from MCE import MainContentExtractor

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
os.environ["QDRANT__API_KEY"] = os.getenv("QDRANT__API_KEY")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3", 
    token=os.getenv("HUGGINGFACE_API_KEY")
)
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
)

# File dove salvare lo stato delle pagine (ETag, Last-Modified, e hash)
STATE_FILE = "data/page_update_state.json"
ALL_URLS_FILE = "urls_lists/urls_html_master_list.txt"
ALL_URLS_PDF_FILE = "urls_lists/urls_pdf_master_list.txt"
DOWNLOADED_PDF_URLS_FILE = "urls_lists/urls_pdf_downloaded_list.txt"
NODES_OUTPUT_FILE = "nodes/nodes_metadata_sentence_x16.pkl"
NEW_NODES_OUTPUT_FILE = "nodes/nodes_metadata_update.pkl"

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "diem_chatbot3_v2"

# Configurazione per l'estrazione metadati
MIN_DELAY_SECONDS = 1
MAX_DELAY_SECONDS = 2

# Header da inviare per simulare un browser reale
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ==============================================================================
# --- SEZIONE 1: FUNZIONI DI SUPPORTO ---
# ==============================================================================

def save_to_pickle(data, filepath):
    """Salva qualsiasi oggetto Python in un file pickle."""
    print(f"Salvataggio di {len(data)} oggetti in '{filepath}'...")
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print("Salvataggio completato.")

def load_from_pickle(filepath):
    """Carica qualsiasi oggetto Python da un file pickle."""
    if not os.path.exists(filepath):
        print(f"File di cache '{filepath}' non trovato.")
        return []
        
    print(f"Caricamento oggetti dalla cache '{filepath}'...")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"Caricati {len(data)} oggetti.")
    return data

def load_state(filepath):
    """Carica in modo sicuro lo stato precedente da un file JSON."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Attenzione: impossibile leggere il file di stato '{filepath}'. Parto da uno stato vuoto. Errore: {e}")
        return {}

def save_state(filepath, state):
    """Salva lo stato corrente in un file JSON."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        print(f"\nStato aggiornato salvato con successo in '{filepath}'.")
    except IOError as e:
        print(f"Errore critico: impossibile salvare il file di stato '{filepath}'. Errore: {e}")

def get_content_hash(content):
    """Calcola l'hash SHA256 del contenuto testuale di una pagina."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_clean_content_hash(html_content):
    """
    Estrae il contenuto principale (come Markdown) usando MainContentExtractor
    e calcola il suo hash.
    """
    try:
        markdown_content = MainContentExtractor.extract(
            html=html_content,
            output_format="markdown",
            include_links=True
        )
        
        if not markdown_content:
            # Se l'estrattore non trova nulla, per sicurezza
            # eseguiamo l'hash di una stringa vuota.
            return get_content_hash("")

        return get_content_hash(markdown_content)
    
    except Exception as e:
        print(f"   Avviso: Fallita estrazione 'clean' del contenuto: {e}. Eseguo hash su testo grezzo.")
        # Se l'estrattore fallisce, esegue il fallback sull'hash del contenuto grezzo
        return get_content_hash(html_content)

def clean_and_validate_url(url):
    """
    Pulisce un URL rimuovendo parametri specifici e il fragment.
    Restituisce l'URL pulito e un flag booleano che è True se la struttura
    è quella del DIEM (300638) o se non è specificata.
    """
    default_params_to_remove = {'bando', 'progetto', 'lettera', 'avvisi', 'coorte', 'schemaid', 'schemaId', 'adCodFraz', 'adCodRadice', 'annoOfferta', 'annoOrdinamento', 'teamId'}
    DIEM_STRUCTURE_ID = '300638'
    
    # Scomponi l'URL e la sua query in un dizionario
    parsed_url = urlparse(url)
    query_dict = parse_qs(parsed_url.query)

    is_diem_structure = True # Assumiamo che sia valido di default
    
    params_to_remove_this_time = default_params_to_remove.copy()
    # Controlla il parametro 'struttura'
    if 'struttura' in query_dict:
        # Se mi trovo nella sezione strutture della rubrica allora rimuovo anche "struttura"
        if "https://rubrica.unisa.it/strutture" in url:
            params_to_remove_this_time.add('struttura')
        # Se il parametro esiste ma il suo valore non è quello corretto
        elif query_dict['struttura'][0] != DIEM_STRUCTURE_ID:
            is_diem_structure = False
            
    # Controlla il parametro 'cdsStruttura'
    elif 'cdsStruttura' in query_dict:
        # Se il parametro esiste ma il suo valore non è quello corretto
        if query_dict['cdsStruttura'][0] != DIEM_STRUCTURE_ID:
            is_diem_structure = False

    # Controlla l'url base per casi speciali
    elif 'https://www.diem.unisa.it/home/bandi' in url and 'modulo' in query_dict and query_dict['modulo'][0] != '226':
        is_diem_structure = False

    # Aggiungi il parametro 'anno' se non esiste
    if 'https://www.diem.unisa.it/home/bandi' in url and 'modulo' in query_dict and 'anno' not in query_dict:
        query_dict['anno'] = ['2025']  

    # Logica di pulizia dei parametri
    if 'bando' in query_dict and 'idConcorso' in query_dict:
        params_to_remove_this_time.remove('bando')
    
    # Rimuovi le chiavi indesiderate
    for param in params_to_remove_this_time:
        query_dict.pop(param, None)
        
    # Ricostruisci la stringa di query e l'URL
    new_query_string = urlencode(query_dict, doseq=True)
    clean_parsed_url = parsed_url._replace(query=new_query_string, fragment="")
    cleaned_url = clean_parsed_url.geturl()
    
    # Restituisce sia l'URL pulito che il flag
    return cleaned_url, is_diem_structure

def get_insegnamento_signature(url):
    """
    Se l'URL è di tipo 'insegnamenti', calcola la sua "firma" unica
    rimuovendo il penultimo segmento del percorso. Altrimenti, restituisce None.
    """
    if "insegnamenti" in url and "corsi" not in url:
        try:
            path_segments = urlparse(url).path.strip('/').split('/')
            if len(path_segments) < 2:
                return None
            signature = tuple(path_segments[:-2] + path_segments[-1:])
            return signature
        except Exception:
            return None
    else:
        return None
    
def make_markdown_links_absolute(markdown_text, base_url):
    def replacer(match):
        link_text = match.group(1)
        link_url = match.group(2)
        absolute_url = urljoin(base_url, link_url)
        return f"[{link_text}]({absolute_url})"
    markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    return re.sub(markdown_link_pattern, replacer, markdown_text)

# ==============================================================================
# --- SEZIONE 2: CONTROLLO DEGLI AGGIORNAMENTI ---
# ==============================================================================

def check_for_updates_robust(urls_to_check, last_state):
    """
    Controlla una lista di URL usando una strategia ibrida basata solo su richieste GET.
    """
    updated_urls = []
    current_state = {}
    
    print(f"Controllo di {len(urls_to_check)} URL per aggiornamenti...")
    
    for url in urls_to_check:
        print(f"\n-> Controllando: {url}")
        previous_data = last_state.get(url, {})
        request_headers = HEADERS.copy()

        # Aggiungi gli header di caching se li abbiamo salvati
        if previous_data.get("etag"):
            request_headers["If-None-Match"] = previous_data["etag"]
        if previous_data.get("last_modified"):
            request_headers["If-Modified-Since"] = previous_data["last_modified"]

        try:
            with requests.get(url, headers=request_headers, timeout=10, allow_redirects=True, stream=True) as response:
                time.sleep(0.5)

                # 1. CONTROLLO EFFICIENTE TRAMITE HEADER
                if response.status_code == 304: # 304 Not Modified
                    print("   Stato: Non modificato (304 via GET).")
                    current_state[url] = previous_data
                    continue
                
                # Se il server risponde 200 OK e fornisce header di caching, li usiamo
                # senza scaricare l'intero contenuto.
                if response.status_code == 200 and (response.headers.get("ETag") or response.headers.get("Last-Modified")):
                    print("   Stato: Aggiornato (rilevato via header GET). Salvo nuovi ETag/Last-Modified.")
                    updated_urls.append(url)
                    current_state[url] = {
                        "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                        "content_hash": None # Resettiamo l'hash
                    }
                    continue

                # 2. FALLBACK SU HASH DEL CONTENUTO
                # Solo se il server risponde 200 OK ma non fornisce header di caching,
                # procediamo a scaricare l'intero contenuto.
                if response.status_code == 200:
                    print("   Info: Il server non supporta caching efficiente. Eseguo fallback su hash del contenuto.")
                    
                    # Scarica il contenuto del body
                    content = response.text
                    new_hash = get_clean_content_hash(content)
                    old_hash = previous_data.get("content_hash")
                    
                    if new_hash != old_hash:
                        print(f"   Stato: Aggiornato (rilevato via hash). Hash: {new_hash[:10]}... (precedente: {str(old_hash)[:10]}...)")
                        updated_urls.append(url)
                        current_state[url] = {
                            "etag": None,
                            "last_modified": None,
                            "content_hash": new_hash
                        }
                    else:
                        print("   Stato: Non modificato (hash identico).")
                        current_state[url] = previous_data
                else:
                    # Gestisce altri status code (es. 403, 404, 500)
                    response.raise_for_status()

        except requests.RequestException as e:
            print(f"   ERRORE: Impossibile controllare l'URL. Errore: {e}")
            if url in last_state:
                current_state[url] = last_state[url]
            
    return updated_urls, current_state

# ==============================================================================
# --- SEZIONE 3: CRAWLER ---
# ==============================================================================

def run_crawler(start_urls, pre_visited_urls):
    """
    Esegue il crawler partendo da una lista di URL fornita.
    """
    if not start_urls:
        print("\n--- FASE 2: Nessuna pagina aggiornata da cui partire. Crawling non avviato. ---")
        return

    print(f"\n--- FASE 2: Inizio crawling da {len(start_urls)} pagine aggiornate ---")

    # Imposta e avvia il browser automatizzato
    options = webdriver.ChromeOptions()
    options.add_argument('--headless') # Esegue Chrome in background, senza aprire una finestra
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # --- IMPOSTAZIONI ---
    ALLOWED_DOMAINS = ["www.diem.unisa.it", "rubrica.unisa.it", "docenti.unisa.it", "easycourse.unisa.it", "web.unisa.it", "corsi.unisa.it", "unisa.coursecatalogue.cineca.it"]
    DOMAINS_REQUIRING_JS = {"unisa.coursecatalogue.cineca.it"}
    EXCLUDED_LANGUAGES = {'en', 'es', 'de', 'fr', 'zh'}

    # --- STRUTTURE DATI ---
    # La coda mantiene la logica del percorso per la stampa a schermo
    urls_to_visit = deque([(url, [url]) for url in start_urls])
    visited_urls = pre_visited_urls # Inizializza con gli URL già visitati
    seen_insegnamenti_signatures = set() # Per evitare duplicati in "insegnamenti"

    for url in pre_visited_urls:
        if "unisa.coursecatalogue.cineca.it" in url:
            signature = get_insegnamento_signature(url)
            if signature and signature not in seen_insegnamenti_signatures:
                seen_insegnamenti_signatures.add(signature)

    for url in start_urls:
        if "unisa.coursecatalogue.cineca.it" in url:
            signature = get_insegnamento_signature(url)
            if signature and signature not in seen_insegnamenti_signatures:
                seen_insegnamenti_signatures.add(signature)

    newly_found_urls = set()

    while urls_to_visit:
        current_url, current_path = urls_to_visit.popleft()

        if current_url in visited_urls:
            continue

        path_str = " -> ".join(current_path)
        print(f"-> Visitando: {path_str}")
        
        visited_urls.add(current_url)
        page_content = None
        try:
            time.sleep(1)
            current_domain = urlparse(current_url).netloc

            if current_domain in DOMAINS_REQUIRING_JS:
                try:
                    # Se il dominio richiede JS, usiamo Selenium
                    driver.get(current_url)
                    wait = WebDriverWait(driver, 10) # Aspetta massimo 10 secondi
                    wait.until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, "main.app-main-container"))
                    )
                    page_content = driver.page_source
                except TimeoutException:
                    print(f"Timeout durante l'attesa del contenuto dinamico per {current_url}")
                    page_content = None # Se non carica, non abbiamo contenuto da analizzare
                except Exception as e:
                    print(f"Un altro errore di Selenium è occorso per {current_url}: {e}")
                    page_content = None
            else:
                # Altrimenti, usiamo 'requests'
                response = requests.get(current_url, timeout=10)
                if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                    page_content = response.content
                
            if page_content:
                newly_found_urls.add(current_url)

                soup = BeautifulSoup(page_content, 'html.parser')

                # Gestione dei link relativi senza slash iniziale
                parsed_current_url = urlparse(current_url)
                base_for_join = current_url
                if "EasyCourse" in current_url:
                    if not current_url.endswith('/') and '.' not in parsed_current_url.path.split('/')[-1]:
                        base_for_join += '/'

                for link in soup.find_all('a', href=True):

                    href = link['href']
                    absolute_url = urljoin(base_for_join, href)
                    parsed_url = urlparse(absolute_url)  
                    path_segments = parsed_url.path.split('/')
                    clean_url = parsed_url._replace(fragment="").geturl()
                    param_cleaned_url, is_diem_structure = clean_and_validate_url(clean_url)
                    new_domain = parsed_url.netloc
                    path = urlparse(clean_url).path
                    module_count = path.count("/module/")
                    row_count = path.count("/row/")

                    if new_domain not in ALLOWED_DOMAINS or EXCLUDED_LANGUAGES.intersection(path_segments) or "sitemap" in clean_url or \
                    ("unisa-rescue-page" in clean_url and ((module_count > 1 and row_count > 1) or "/uploads/rescue/" in clean_url)) or not is_diem_structure or \
                        clean_url.endswith(('.pdf', '.doc', '.docx', '.jpg', '.png', '.htm')) or clean_url.startswith("http://"): 
                        continue

                    should_add = False
                    if new_domain == "www.diem.unisa.it" and current_domain == "www.diem.unisa.it":
                        should_add = True
                    elif new_domain == "rubrica.unisa.it" and current_domain == "www.diem.unisa.it":
                        should_add = True
                    elif new_domain == "docenti.unisa.it" and (current_domain == "rubrica.unisa.it" or (current_domain == "docenti.unisa.it" and (("curriculum" in clean_url and not clean_url.endswith("/")) or ("didattica" in clean_url and "didattica" not in current_url)))) and clean_url != "https://docenti.unisa.it" and "simona.mancini" not in clean_url:
                        should_add = True
                    elif new_domain == "easycourse.unisa.it" and ("Dipartimento_di_Ingegneria_dellInformazione_ed_Elettrica_e_Matematica_Applicata" in clean_url or "Facolta_di_Ingegneria_-_Esami" in clean_url):
                        if ("index" in current_url and ("ttCdlHtml" not in clean_url and "index" not in clean_url)) or "ttCdlHtml" in current_url: 
                            should_add = False
                        else:
                            should_add = True
                    elif new_domain == "web.unisa.it" and (current_domain == "www.diem.unisa.it" or current_domain == "web.unisa.it") and "servizi-on-line" in clean_url:
                        should_add = True
                    elif new_domain == "corsi.unisa.it" and (current_domain == "www.diem.unisa.it" or current_domain == "corsi.unisa.it") and clean_url != "https://corsi.unisa.it" and clean_url != "http://corsi.unisa.it" and "unisa-rescue-page" not in clean_url and "news" not in clean_url and "occupazione-spazi" not in clean_url and "information-Engineering-for-digital-medicine" not in clean_url and not re.search(r"^https://corsi\.unisa\.it/\d{5,}", clean_url):
                        should_add = True
                    elif new_domain == "unisa.coursecatalogue.cineca.it" and (current_domain == "corsi.unisa.it" or current_domain == "unisa.coursecatalogue.cineca.it") and clean_url != "https://unisa.coursecatalogue.cineca.it/" and "gruppo" not in clean_url and "cerca-" not in clean_url and "support.apple.com" not in clean_url and "WWW.ESSE3WEB.UNISA.IT" not in clean_url:
                        signature = get_insegnamento_signature(param_cleaned_url)
                        if not signature or signature in seen_insegnamenti_signatures:
                            continue # Salta se la firma non è valida o è già stata vista
                        
                        seen_insegnamenti_signatures.add(signature)
                        should_add = True
                    
                    if should_add:
                        if param_cleaned_url not in visited_urls:
                            new_path = current_path + [param_cleaned_url]
                            urls_to_visit.append((param_cleaned_url, new_path))

        except requests.RequestException as e:
            print(f"Errore durante la richiesta a {current_url}: {e}")

    driver.quit()

    print("\nCrawling completato.")
    newly_found_urls = [url for url in newly_found_urls if "rubrica.unisa.it" not in url]
    return newly_found_urls

# ==============================================================================
# --- SEZIONE 4: LOGICA DI ELABORAZIONE DEL CONTENUTO ---
# ==============================================================================

def process_urls_to_documents(urls_to_process):
    """
    Prende una lista di URL, estrae il contenuto principale, lo elabora
    e salva il risultato in un file pickle.
    """
    if not urls_to_process:
        print("\nFASE 3: Nessun nuovo documento da elaborare.")
        return

    print(f"\nFASE 3: Elaborazione del contenuto di {len(urls_to_process)} pagine...")

    # 1. Estrazione del blocco HTML principale
    loader = MainContentExtractorReader()
    html_documents = loader.load_data(urls=urls_to_process)

    # 2. Aggiunta dei metadati
    for doc, url in zip(html_documents, urls_to_process):
        doc.metadata["source_url"] = url
    print(f"Metadato 'source_url' aggiunto a {len(html_documents)} documenti.")

    # 3. Elaborazione del testo e creazione dei documenti finali
    processed_documents = []
    for doc in html_documents:
        base_url = doc.metadata.get("source_url", "")
        clean_text_with_links = make_markdown_links_absolute(doc.text, base_url)
        processed_documents.append(
            Document(text=clean_text_with_links, metadata=doc.metadata, id_=base_url)
        )

    # 4. Salvataggio del risultato
    print(f"Elaborati {len(processed_documents)} documenti.")
    return processed_documents

def process_pdfs(url_list_file, url_list_downloaded_file):
    """
    Legge gli URL dei PDF, li scarica in memoria, estrae il testo 
    e restituisce una lista di Document di LlamaIndex.
    """
    
    with open(url_list_file, "r") as f:
        urls_to_process = [line.strip() for line in f if line.strip()]
    
    print(f"Trovati {len(urls_to_process)} PDF da processare in memoria.")

    with open(url_list_downloaded_file, "r") as f:
        already_downloaded_urls = [line.strip() for line in f if line.strip()]

    pdfs_to_process = [url for url in urls_to_process if url not in already_downloaded_urls]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    pdf_documents = []

    for url in pdfs_to_process:
        try:
            print(f"-> Processando in memoria: {url}")
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status() # Controlla errori HTTP

            # Assicurati che sia un PDF prima di continuare
            if 'application/pdf' not in response.headers.get('content-type', ''):
                print(f"  [SKIPPATO] L'URL non è un PDF, ma: {response.headers.get('content-type')}")
                continue

            pdf_bytes = io.BytesIO(response.content)
            
            reader = PdfReader(pdf_bytes)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or "" # Aggiungi "" se la pagina è vuota
            
            if not text.strip():
                print("  [SKIPPATO] Il PDF è vuoto o contiene solo immagini (no testo).")
                continue

            doc = Document(
                text=text,
                metadata={
                    "source_url": url,
                }
            )
            pdf_documents.append(doc)

            # Aggiungi l'URL alla lista dei già scaricati
            with open(url_list_downloaded_file, "a") as f:
                f.write(url + "\n")

        except requests.RequestException as e:
            print(f"  [ERRORE DOWNLOAD]: {e}")
        except Exception as e:
            print(f"  [ERRORE LETTURA PDF]: {e}")

    print(f"Processati {len(pdf_documents)} documenti PDF in memoria.")
    return pdf_documents

# ==============================================================================
# --- SEZIONE 5: ARRICCHIMENTO METADATI E CREAZIONE NODI ---
# ==============================================================================

def enrich_documents_with_metadata(documents):
    """
    Arricchisce una lista di documenti con metadati generati da un LLM.
    """
    if not documents:
        print("\nFASE 4: Nessun documento da arricchire con metadati.")
        return []

    print(f"\nFASE 4: Inizio arricchimento metadati per {len(documents)} documenti...")

    for document in tqdm(documents, desc="Arricchendo documenti"):
        if not isinstance(document.text, str) or not document.text.strip():
            continue
        
        # Estratto dal tuo codice di estrazione
        prompt = (
            "Analizza il seguente documento, inclusa la sua URL di origine, per estrarre i metadati richiesti. "
            "Fornisci l'output esclusivamente in formato JSON, seguendo la struttura e le regole specificate.\n\n"
            "--- DOCUMENTO ---\n"
            f'"""{document.text}"""\n'
            "--- FINE DOCUMENTO ---\n\n"
            
            "REGOLE GENERALI:\n"
            "- Se il documento è troppo breve o non contiene informazioni sufficienti per generare un campo specifico (title, summary, etc.), lascia quel campo vuoto (es. `\"title\": \"\"` o `\"questions\": []`).\n"
            "- **Keywords**: Estrai un numero di parole chiave proporzionale alla lunghezza del testo, fino a un massimo di 10. Per documenti molto brevi, poche parole chiave (o nessuna) sono accettabili.\n"
            "- **Domande**: Genera un numero di domande proporzionale alla lunghezza del testo, fino a un massimo di 3. Per documenti molto brevi, una sola domanda o nessuna sono accettabili.\n\n"

            "REGOLE PER 'years':\n"
            "- Identifica l'anno o gli anni accademici (es. '2024/2025') o solari (es. '2023') *PRINCIPALI* del documento. Controlla sia il testo che l'URL di origine.\n"
            "- Se il documento tratta un singolo anno, inserisci solo quello. Esempio: [\"2023\"].\n"
            "- Se nell'URL è presente il parametro 'anno', considera il suo valore corrispondente come UNICO anno principale, NON considerare il parametro quando 'anno=0'.\n"
            "- Se tratta più anni, IDENTIFICA QUELLO PRINCIPALE ed inseriscilo, altrimenti inseriscili tutti. Esempio: [\"2022\", \"2023\", \"2024\"].\n"
            "- Se non riesci ad identificare l'anno principale dal testo, ma è presente nell'URL, usa quello. Esempio: [\"2023\"] se l'URL contiene 'anno=2023' (sempre escludendo il caso in cui sia 'anno=0').\n"
            "- Se non ha un anno di riferimento, lascia la lista vuota. Esempio: [].\n\n"

            "Formato JSON richiesto:\n"
            "{\n"
            '  "title": "Un titolo conciso e descrittivo del documento",\n'
            '  "summary": "Un riassunto di 2-3 frasi del contenuto principale",\n'
            '  "questions": [],\n'  # Da 0 a 3 domande in base alla lunghezza del testo
            '  "keywords": [],\n'   # Da 0 a 10 parole chiave in base alla lunghezza del testo
            '  "years": []\n'
            "}\n\n"
            "Output JSON:"
        )
        
        try:
            response = Settings.llm.complete(prompt)
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            metadata = json.loads(cleaned_response)
            document.metadata.update(metadata) # Aggiorna direttamente i metadati del documento
        except Exception as e:
            print(f"\nErrore durante l'arricchimento del documento {document.metadata.get('source_url', 'N/A')}: {e}")
        
        time.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
        
    print("Arricchimento metadati completato.")
    return documents

def create_nodes_from_documents(documents, output_filepath):
    """
    Prende una lista di documenti arricchiti e li trasforma in nodi.
    Prima di salvare, rimuove i nodi obsoleti dal file .pkl esistente
    per evitare duplicati.
    """
    if not documents:
        print("\nFASE 5: Nessun documento da trasformare in nodi.")
        return [] # Restituisce una lista vuota se non ci sono nuovi nodi

    print(f"\nFASE 5: Creazione di nodi da {len(documents)} nuovi documenti...")
    
    sentence_splitter = SentenceSplitter(chunk_size=512*16, chunk_overlap=512*2)
    pipeline = IngestionPipeline(transformations=[sentence_splitter])
    
    # 1. Crea i nuovi nodi
    new_nodes = pipeline.run(documents=documents, show_progress=True)
    print(f"Creati {len(new_nodes)} nuovi nodi.")

    # 2. Ottieni gli ID dei documenti che stiamo aggiornando
    doc_ids_to_update = set(doc.id_ for doc in documents if doc.id_)
    
    if not doc_ids_to_update:
        print("ATTENZIONE: I documenti in input non hanno un 'id_'.")
        print("La rimozione dei duplicati potrebbe fallire. Controllo 'source_url' nei metadati...")
        # Fallback nel caso in cui l'id_ non sia stato impostato, ma 'source_url' sì
        doc_ids_to_update = set(doc.metadata.get("source_url") for doc in documents if doc.metadata.get("source_url"))
        if not doc_ids_to_update:
            print("ERRORE: Impossibile determinare gli ID dei documenti da aggiornare. L'unione dei nodi conterrà duplicati.")
            
    # 3. Carica i nodi esistenti
    existing_nodes = []
    if os.path.exists(output_filepath):
        try:
            print(f"File '{output_filepath}' esistente. Caricamento nodi...")
            existing_nodes = load_from_pickle(output_filepath)
            
            if not isinstance(existing_nodes, list):
                print(f"Attenzione: il file '{output_filepath}' non conteneva una lista. Sarà sovrascritto.")
                existing_nodes = []
            else:
                print(f"Caricati {len(existing_nodes)} nodi esistenti.")
                
                if doc_ids_to_update:
                    print(f"Filtraggio dei nodi obsoleti (documenti da aggiornare: {len(doc_ids_to_update)})...")
                    
                    filtered_existing_nodes = []
                    nodes_removed_count = 0
                    
                    for node in existing_nodes:
                        if node.ref_doc_id not in doc_ids_to_update:
                            filtered_existing_nodes.append(node)
                        else:
                            nodes_removed_count += 1
                            
                    print(f"Rimossi {nodes_removed_count} nodi obsoleti.")
                    existing_nodes = filtered_existing_nodes # Sovrascrivi la lista

        except Exception as e:
            print(f"Errore nel caricamento di '{output_filepath}': {e}. Il file sarà sovrascritto.")
            existing_nodes = []
    
    # 4. Combina i nodi esistenti (filtrati) con i nuovi nodi
    all_nodes = existing_nodes + new_nodes
    
    # 5. Salva la lista completa
    save_to_pickle(all_nodes, output_filepath)
    print(f"Salvataggio completato. Totale nodi in '{output_filepath}': {len(all_nodes)}")

    # Restituisce solo i nodi appena creati
    return new_nodes

# ==============================================================================
# --- SEZIONE 6: INDICIZZAZIONE SU QDRANT ---
# ==============================================================================

def index_nodes_to_qdrant(nodes_to_index, urls_to_delete):
    """
    Indicizza una lista di nodi in una collezione Qdrant.
    Crea la collezione se non esiste, altrimenti aggiunge i nodi.
    """
    if not nodes_to_index:
        print("\nFASE 6: Nessun nuovo nodo da indicizzare.")
        return

    print(f"\nFASE 6: Inizio indicizzazione di {len(nodes_to_index)} nodi su Qdrant Locale...")

    # 1. Connettiti a Qdrant
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION_NAME)

    # 2. Controlla se la collezione esiste già
    try:
        # Questo comando fallisce se la collezione non esiste
        client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        collection_exists = True
        print(f"La collezione '{QDRANT_COLLECTION_NAME}' esiste già. Aggiungo i nuovi nodi.")
    except Exception:
        collection_exists = False
        print(f"La collezione '{QDRANT_COLLECTION_NAME}' non esiste. Verrà creata.")

    # 3. Indicizza i nodi
    if collection_exists:
        # Carica l'indice esistente
        index = VectorStoreIndex.from_vector_store(vector_store)

        # Rimuovi i vecchi nodi corrispondenti agli URL da eliminare
        doc_ids_to_delete = list(urls_to_delete)
        for doc_id in tqdm(doc_ids_to_delete, desc="Eliminazione vecchi nodi"):
            # delete_ref_doc cerca e rimuove tutti i nodi con questo ref_doc_id
            index.delete_ref_doc(doc_id, delete_from_docstore=True)

        # Inserisci i nuovi nodi
        index.insert_nodes(nodes_to_index, show_progress=True)
    else:
        # Crea l'indice da zero con i nuovi nodi
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes_to_index, storage_context=storage_context, show_progress=True)
    
    print("Indicizzazione su Qdrant completata con successo.")

# ==============================================================================
# --- SEZIONE 7: ESECUZIONE DEL FLUSSO INTEGRATO ---
# ==============================================================================

def main_workflow():
    """ Esegue il flusso completo di controllo aggiornamenti e crawling. """
    print(f"--- AVVIO PROCESSO DI AGGIORNAMENTO ({time.ctime()}) ---")
    
    # 1. Carica la lista master di URL da monitorare
    try:
        with open(ALL_URLS_FILE, "r", encoding="utf-8") as f:
            urls_to_monitor = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File {ALL_URLS_FILE} non trovato. Inizio con la sitemap di default.")
        urls_to_monitor = ["https://www.diem.unisa.it/home?sitemap"]
    
    # 2. Controlla gli aggiornamenti
    last_known_state = load_state(STATE_FILE)
    updated_pages, new_state = check_for_updates_robust(urls_to_monitor, last_known_state)
    
    # 3. Usa le pagine aggiornate come punto di partenza per il crawler
    # Se non ci sono pagine aggiornate, il crawler non parte.
    # Se è la prima esecuzione (last_known_state è vuoto), tutte le pagine sono "nuove".
    start_points_for_crawler = updated_pages
    if not last_known_state:
        print("\nPrima esecuzione: tutte le pagine verranno considerate 'da visitare'.")
        start_points_for_crawler = urls_to_monitor
        # Alla prima esecuzione, non ci sono pagine da ignorare
        unchanged_urls = set()
    else:
        print(f"Rilevati {len(updated_pages)} URL HTML aggiornati.")
        unchanged_urls = set(urls_to_monitor) - set(updated_pages)

    crawled_urls = run_crawler(start_points_for_crawler, unchanged_urls)

    # 4. Aggiorna la lista master degli URL HTML
    if crawled_urls:
        print(f"\nAggiornamento della lista URL master con {len(crawled_urls)} nuove pagine.")
        # Unisci i vecchi URL con i nuovi trovati, rimuovendo duplicati
        os.makedirs(os.path.dirname(ALL_URLS_FILE), exist_ok=True)
        final_url_set = set(urls_to_monitor).union(crawled_urls)
        with open(ALL_URLS_FILE, "w", encoding="utf-8") as f:
            for url in sorted(list(final_url_set)):
                f.write(f"{url}\n")
        print(f"Lista URL master aggiornata con {len(crawled_urls)} nuovi URL.")
    
    # 5. Estrai il contenuto
    newly_processed_htmls = process_urls_to_documents(crawled_urls)
    newly_processed_pdfs = process_pdfs(ALL_URLS_PDF_FILE, DOWNLOADED_PDF_URLS_FILE)

    newly_processed_documents = newly_processed_htmls + newly_processed_pdfs
    if not newly_processed_documents:
        print("Nessun nuovo documento (HTML o PDF) da elaborare.")
        save_state(STATE_FILE, new_state) # Salva comunque lo stato
        print(f"--- PROCESSO DI AGGIORNAMENTO TERMINATO ({time.ctime()}) ---")
        return

    # 6. Arricchisci con metadati
    enriched_documents = enrich_documents_with_metadata(newly_processed_documents)
    
    # 7. Crea e salva i nodi
    new_nodes = create_nodes_from_documents(enriched_documents, NODES_OUTPUT_FILE)
    save_to_pickle(new_nodes, NEW_NODES_OUTPUT_FILE)

    # 8. Indicizza i nodi su Qdrant
    index_nodes_to_qdrant(new_nodes, start_points_for_crawler)
    
    # 9. Salva lo stato aggiornato delle pagine per il prossimo controllo
    save_state(STATE_FILE, new_state)
    print(f"--- PROCESSO DI AGGIORNAMENTO TERMINATO ({time.ctime()}) ---")

if __name__ == "__main__":
    main_workflow()