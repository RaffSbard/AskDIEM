import streamlit as st
import datetime
from babel.dates import format_datetime
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from qdrant_client import QdrantClient
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from typing import List

# --- CLASSE POST-PROCESSOR PERSONALIZZATA ---
class KeepAtLeastOneNodePostprocessor(BaseNodePostprocessor):
    """
    Un post-processore personalizzato che "avvolge" altri post-processori 
    per garantire che, se il retriever aveva originariamente trovato dei nodi,
    almeno uno venga sempre restituito per evitare che il ChatEngine fallisca.
    """
    postprocessors: List[BaseNodePostprocessor]

    def _postprocess_nodes(self, nodes, query_str):
        """
        Applica la logica di post-processing.
        """
        if not nodes:
            # Se il retriever non ha trovato nulla, restituisce una lista vuota.
            return []
        
        # Salviamo un riferimento al nodo migliore prima di applicare i filtri.
        best_node = nodes[0] 
        
        # Applica tutti i post-processori "avvolti"
        processed_nodes = nodes
        for pp in self.postprocessors:
            processed_nodes = pp.postprocess_nodes(processed_nodes, query_str=query_str)
        
        if not processed_nodes:
            return [best_node]
        
        return processed_nodes

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
)

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT__API_KEY"]

@st.cache_resource(show_spinner=False)
def load_index():
    """Carica i dati, inizializza i modelli e costruisce l'indice."""

    with st.spinner(ui_texts["spinner_message"]):
        Settings.llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.5
        )

        # Alternativa a bge-m3 poichè troppo pesante per esser caricato in cloud
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name="gemini-embedding-001", 
            api_key=GOOGLE_API_KEY
        )

        qdrant_client = QdrantClient(
            url="https://e542824d-6590-4005-91db-6dd34bf8f471.eu-west-2-0.aws.cloud.qdrant.io:6333", 
            api_key=QDRANT_API_KEY,
        )

        vector_store = QdrantVectorStore(client=qdrant_client, collection_name="diem_chatbot_final")

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return vector_index, storage_context

vector_index, storage_context = load_index()

# --- 2. GESTIONE DELLA CHAT ---

st.title(ui_texts["title"])
st.caption(ui_texts["caption"])

SYSTEM_PROMPT_TEMPLATE = (
    """Sei AskDIEM, un assistente virtuale dell'Università di Salerno, specializzato nell'aiutare gli studenti del Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica Applicata (DIEM).

    Il tuo obiettivo è fornire risposte accurate basandoti esclusivamente sulle informazioni ufficiali che ti vengono fornite.
    Tieni presente che oggi è: {current_date}.

    REGOLE GENERALI:
    - *IMPORTANTE*: Se la domanda ti viene posta in inglese rispondi in inglese, a prescindere dalla lingua dei messaggi precedenti o da quella del contesto fornito.
    - A meno che nella domanda non venga specificato un anno o una data in particolare, rispondi sempre tenendo presente la data di oggi.
    - Se nomini un evento, adegua i tempi verbali in base alla data attuale.
    - Se non disponi delle informazioni necessarie per rispondere a una domanda, dichiara chiaramente: "Non dispongo delle informazioni necessarie per rispondere a questa domanda."
    - Non inventare mai informazioni, contatti, date o procedure. La tua priorità è l'accuratezza."""
)

if "chat_engine" not in st.session_state:

    shared_memory = ChatMemoryBuffer.from_defaults(token_limit=50000)

    context_prompt = (
        """Date le seguenti informazioni estratte dai documenti ufficiali e la domanda dell'utente, fornisci una risposta chiara ed esaustiva.

        Contesto:
        {context_str}

        Istruzioni per la risposta:
        - Basa la tua risposta esclusivamente sul contesto fornito.
        
        - ISTRUZIONE PER I LINK: Se nel contesto è presente una risorsa rilevante (come un PDF di un bando, una graduatoria o una pagina web) che supporta la tua risposta, devi citarla usando il formato Markdown: [Titolo Significativo](URL).
        - Il "Titolo Significativo" dovrebbe essere il titolo del documento (es. 'Bando Collaborazioni studentesche 2024') che trovi nel contesto.
        - L' "URL" è l'indirizzo web (source_url) associato a quel titolo.
        
        - Esempio di formato CORRETTO:
        Per maggiori dettagli, puoi consultare il [Bando per Collaborazioni Studentesche](https://www.unisa.it/bando-collaborazioni-...).
        
        - Esempio di formato ERRATO (da non usare):
        Per maggiori dettagli, puoi consultare https://www.unisa.it/bando-collaborazioni-...
        
        - Non includere link o titoli che non siano esplicitamente presenti nel contesto.

        Domanda: {query_str}
        Risposta:
        """
    )

    # Definiamo i post-processori che vogliamo filtrare
    filtering_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.15)
    ]

    st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=vector_index.as_retriever(similarity_top_k=15),
        memory=shared_memory,
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
        context_prompt=context_prompt,
        node_postprocessors=[
            CohereRerank(api_key=COHERE_API_KEY, top_n=15), 
            KeepAtLeastOneNodePostprocessor(postprocessors=filtering_postprocessors)
        ],
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

        # Se il messaggio è dell'assistente e contiene fonti, mostrale
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander(ui_texts["sources_expander"]):
                
                all_urls = []
                for node in message["sources"]:
                    source_info = node.metadata.get("source_url") or node.metadata.get("file_name")
                    if source_info:
                        all_urls.append(source_info)
                
                # Ottieni una lista di URL unici mantenendo l'ordine
                unique_urls = list(dict.fromkeys(all_urls))
                
                if not unique_urls:
                    st.info(ui_texts["no_sources_message"])
                else:
                    # Elenca ogni URL unico
                    for url in unique_urls:
                        st.markdown(f"- {url}")

if prompt := st.chat_input(ui_texts["chat_input_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        streaming_response = None

        with st.spinner(ui_texts["thinking_message"]):

            # Formatta la data
            current_date_str = format_datetime(datetime.datetime.now(), format="EEEE, d MMMM yyyy", locale="it_IT")

            chat_engine = st.session_state.chat_engine
            # Aggiorna il system prompt del motore RAG con la data corrente
            chat_engine._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(current_date=current_date_str)
            
            # Avvia lo stream
            streaming_response = chat_engine.stream_chat(prompt)


        # Scrivi lo stream sul frontend e cattura la risposta completa
        final_response_text = st.write_stream(streaming_response.response_gen)

        source_nodes_for_display = streaming_response.source_nodes

        # Mostra le fonti
        with st.expander(ui_texts["sources_expander"]):
            
            is_conversational = False
            if len(source_nodes_for_display) == 1:
                if source_nodes_for_display[0].score < 0.15: 
                     is_conversational = True
            
            if not source_nodes_for_display or is_conversational:
                st.info(ui_texts["no_sources_message"])
            else:
                all_urls = []
                for node in source_nodes_for_display:
                    source_info = node.metadata.get("source_url") or node.metadata.get("file_name")
                    if source_info:
                        all_urls.append(source_info)
                
                unique_urls = list(dict.fromkeys(all_urls))
                
                if not unique_urls:
                    st.info(ui_texts["no_sources_message"])
                else:
                    for url in unique_urls:
                        st.markdown(f"- {url}")
    
    nodes_to_save = []
    if not is_conversational:
        nodes_to_save = source_nodes_for_display
    
    # Aggiungi la risposta e le fonti dell'assistente alla cronologia
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_response_text,
        "sources": nodes_to_save
    })