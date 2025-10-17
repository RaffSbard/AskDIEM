import streamlit as st
import datetime
import locale
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from qdrant_client import QdrantClient
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank
from typing import Any
import pickle
import os

# def load_from_pickle(filepath: str) -> Any:
#     """Carica qualsiasi oggetto Python da un file pickle."""
#     if not os.path.exists(filepath):
#         print(f"File di cache '{filepath}' non trovato.")
#         return []
        
#     print(f"Caricamento oggetti dalla cache '{filepath}'...")
#     with open(filepath, "rb") as f:
#         data = pickle.load(f)
#     print(f"Caricati {len(data)} oggetti.")
#     return data

# --- 0. DIZIONARIO PER LE TRADUZIONI ---
TRANSLATIONS = {
    "Italiano": {
        "page_title": "AskDIEM",
        "title": "🤖 AskDIEM",
        "caption": "Un assistente AI basato sui documenti ufficiali del Dipartimento di Ingegneria dell'Informazione e Elettrica e Matematica Applicata.",
        "initial_message": "Ciao! Sono il chatbot del DIEM. Chiedimi informazioni sui corsi, gli orari, i docenti o i regolamenti.",
        "chat_input_placeholder": "Scrivi qui la tua domanda...",
        "spinner_message": "Caricamento dei documenti e creazione dell'indice... L'operazione potrebbe richiedere alcuni minuti.",
        "thinking_message": "Sto pensando...",
        "sources_expander": "Mostra le fonti utilizzate",
        "no_sources_message": "Nessuna fonte specifica è stata recuperata per questa risposta.",
        "source_label": "Fonte",
        "relevance_score_label": "Punteggio di pertinenza"
    },
    "English": {
        "page_title": "AskDIEM",
        "title": "🤖 AskDIEM",
        "caption": "An AI assistant based on the official documents of the Department of Information and Electrical Engineering and Applied Mathematics.",
        "initial_message": "Hi! I'm the DIEM chatbot. Ask me about courses, schedules, professors, or regulations.",
        "chat_input_placeholder": "Write your question here...",
        "spinner_message": "Loading documents and creating the index... This may take a few minutes.",
        "thinking_message": "Thinking...",
        "sources_expander": "Show sources used",
        "no_sources_message": "No specific sources were retrieved for this answer.",
        "source_label": "Source",
        "relevance_score_label": "Relevance score"
    }
}

# --- 1. CONFIGURAZIONE DELLA PAGINA E SCELTA DELLA LINGUA ---

# Inizializza la lingua nello stato della sessione se non è presente
if "language" not in st.session_state:
    st.session_state.language = "Italiano" # Lingua predefinita

# Selettore della lingua nella sidebar
st.sidebar.image("./askdiem.png", width="stretch", caption="AskDIEM")
st.sidebar.title("Settings / Impostazioni")
selected_language = st.sidebar.selectbox(
    label="Language / Lingua",
    options=["Italiano", "English"],
    index=["Italiano", "English"].index(st.session_state.language)
)

# Se la lingua cambia, aggiorna lo stato e resetta i messaggi
if st.session_state.language != selected_language:
    st.session_state.language = selected_language
    if "messages" in st.session_state and len(st.session_state.messages) <= 1:
        st.session_state.messages = None 
    st.rerun()

# Carica i testi dell'interfaccia nella lingua corretta
ui_texts = TRANSLATIONS[st.session_state.language]

# Imposta la configurazione della pagina
st.set_page_config(
    page_title=ui_texts["page_title"],
    page_icon="./askdiem.png",
    layout="centered",
    # initial_sidebar_state="auto"
)

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT__API_KEY"]

# Imposta la lingua per la data in base alla scelta
try:
    if st.session_state.language == "Italiano":
        locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')
    # else:
    #     locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
except locale.Error as e:
    print(f"Errore nell'impostare il locale: {e}")

@st.cache_resource(show_spinner=False)
def load_index() -> VectorStoreIndex | StorageContext:
    """Carica i dati, inizializza i modelli e costruisce l'indice."""
    # Nota: i messaggi di caricamento qui non possono essere multilingua
    # perché la lingua viene scelta dopo l'avvio della funzione.
    with st.spinner(ui_texts["spinner_message"]):
        Settings.llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.5
        )
        print("Modello LLM impostato su Google Gemini 2.5 Flash.")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        print("Modello di embedding impostato su BAAI/bge-m3.")

        qdrant_client = QdrantClient(
            url="https://e542824d-6590-4005-91db-6dd34bf8f471.eu-west-2-0.aws.cloud.qdrant.io:6333", 
            api_key=QDRANT_API_KEY,
        )
        print("Client Qdrant inizializzato.")

        vector_store = QdrantVectorStore(client=qdrant_client, collection_name="diem_chatbot3")
        print("Vector store Qdrant collegato alla collezione 'diem_chatbot3'.")

        # docstore = SimpleDocumentStore()
        # nodes = load_from_pickle("./nodes/nodes_metadata_hierarchical_x16x4x1.pkl")
        # docstore.add_documents(nodes)

        # storage_context = StorageContext.from_defaults(docstore=docstore)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        print("Indice vettoriale caricato da Qdrant.")
        return vector_index, storage_context

vector_index, storage_context = load_index()

# --- 2. GESTIONE DELLA CHAT ---

st.title(ui_texts["title"])
st.caption(ui_texts["caption"])

# Il system prompt e il context prompt rimangono in italiano come da tua logica
# Se vuoi renderli multilingua, dovrai aggiungerli al dizionario TRANSLATIONS
SYSTEM_PROMPT_TEMPLATE = (
    """Sei un assistente virtuale dell'Università di Salerno, specializzato nell'aiutare gli studenti del Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica Applicata (DIEM).

    Il tuo obiettivo è fornire risposte accurate basandoti esclusivamente sulle informazioni ufficiali che ti vengono fornite.
    Tieni presente che oggi è: {current_date}.

    REGOLE GENERALI:
    - Rispondi nella lingua in cui ti viene posta la domanda, con un tono formale, chiaro e professionale.
    - A meno che nella domanda non venga specificato un anno o una data in particolare, rispondi sempre tenendo presente la data di oggi.
    - Se nomini un evento, adegua i tempi verbali in base alla data attuale.
    - Se non disponi delle informazioni necessarie per rispondere a una domanda, dichiara chiaramente: "Non dispongo delle informazioni necessarie per rispondere a questa domanda."
    - Non inventare mai informazioni, contatti, date o procedure. La tua priorità è l'accuratezza."""
)

if "chat_engine" not in st.session_state:
    print("Creazione di una nuova istanza del Chat Engine.")
    context_prompt = (
        """Date le seguenti informazioni estratte dai documenti ufficiali e la domanda dell'utente, fornisci una risposta chiara ed esaustiva.

        Contesto:
        {context_str}

        Istruzioni per la risposta:
        - Basa la tua risposta esclusivamente sul contesto fornito.
        - Se nel contesto è presente un link a una risorsa rilevante (come un PDF di un bando, una graduatoria o una pagina web), citalo esplicitamente alla fine della tua risposta.
        - Non includere link che non siano presenti nel contesto.

        Domanda: {query_str}
        Risposta:
        """
    )
    st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=vector_index.as_retriever(similarity_top_k=10),
        # retriever = AutoMergingRetriever(vector_index.as_retriever(similarity_top_k=10), storage_context),
        memory=ChatMemoryBuffer.from_defaults(token_limit=50000),
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
        context_prompt=context_prompt,
        node_postprocessors=[CohereRerank(api_key=COHERE_API_KEY, top_n=5)],
        verbose=True,
    )

if "messages" not in st.session_state or st.session_state.messages is None:
    st.session_state.messages = [{
        "role": "assistant",
        "content": ui_texts["initial_message"]
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Se il messaggio è dell'assistente E contiene fonti, mostrale
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander(ui_texts["sources_expander"]):
                for i, node in enumerate(message["sources"]):
                    # Costruisce il titolo dell'expander usando i testi tradotti
                    expander_title = (
                        f"{ui_texts['source_label']} {i+1} "
                        f"({ui_texts['relevance_score_label']}: {node.score:.2f})"
                    )
                    with st.expander(expander_title):
                        source_info = node.metadata.get('source_url') or node.metadata.get('file_name')
                        if source_info:
                            st.caption(f"Da: {source_info}")
                        st.info(node.text)

if prompt := st.chat_input(ui_texts["chat_input_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner(ui_texts["thinking_message"]):
            current_date_str = datetime.datetime.now().strftime("%A, %d %B %Y")
            chat_engine = st.session_state.chat_engine
            chat_engine.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(current_date=current_date_str)
            response = chat_engine.chat(prompt)
            st.write(response.response)

            with st.expander(ui_texts["sources_expander"]):
                if response.source_nodes:
                    for i, node in enumerate(response.source_nodes):
                        expander_title = (
                            f"{ui_texts['source_label']} {i+1} "
                            f"({ui_texts['relevance_score_label']}: {node.score:.2f})"
                        )
                        with st.expander(expander_title):
                            source_info = node.metadata.get('source_url') or node.metadata.get('file_name')
                            if source_info:
                                st.caption(f"Da: {source_info}")
                            st.info(node.text)
                else:
                    st.info(ui_texts["no_sources_message"])
    
    # Aggiungi la risposta E le fonti dell'assistente alla cronologia
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response.response,
        "sources": response.source_nodes
    })