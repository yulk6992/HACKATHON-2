# HACKATHON 2 - RAG local Streamlit
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

st.set_page_config(page_title="Hackathon2 RAG", layout="wide")


def batch_encode_texts(model, chunks: Sequence[str], batch_size: int = 4) -> np.ndarray:
    if not chunks:
        dim = model.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype="float32")

    embeddings: List[np.ndarray] = []
    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=True if start == 0 else False,
        )
        embeddings.append(emb)
    return np.vstack(embeddings).astype("float32")


def summarize_in_batches(
    summarizer_pipeline,
    texts: Sequence[str],
    batch_size: int = 3,
    instruction: str = "summarize: ",
) -> List[str]:
    summaries: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        if not batch:
            continue
        prompt = instruction + " ".join(batch)
        output = summarizer_pipeline(
            prompt,
            max_new_tokens=180,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"]
        summaries.append(output)
    return summaries


def build_faiss_index(embeddings: np.ndarray, cosine: bool = True) -> faiss.Index:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")
    dim = embeddings.shape[1] if embeddings.size else 384
    if cosine:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    if embeddings.size:
        index.add(embeddings)
    return index


@dataclass
class ChunkMetadata:
    source: str
    chunk_id: int
    text: str


@dataclass
class RollingChunkStore:
    max_chunks: int = 2000
    texts: List[str] = field(default_factory=list)
    metadata: List[ChunkMetadata] = field(default_factory=list)

    def add_document(self, filename: str, chunks: Iterable[str]) -> None:
        for idx, chunk in enumerate(chunks):
            self.texts.append(chunk)
            self.metadata.append(ChunkMetadata(filename, idx, chunk))
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        overflow = len(self.texts) - self.max_chunks
        if overflow > 0:
            del self.texts[:overflow]
            del self.metadata[:overflow]

    def export(self) -> Tuple[List[str], List[ChunkMetadata]]:
        return list(self.texts), list(self.metadata)


def rebuild_index_from_store(
    store: RollingChunkStore,
    encoder_model,
    batch_size: int = 4,
) -> Tuple[faiss.Index, np.ndarray, List[ChunkMetadata]]:
    texts, metadata = store.export()
    embeddings = batch_encode_texts(encoder_model, texts, batch_size=batch_size)
    index = build_faiss_index(embeddings, cosine=True)
    return index, embeddings, metadata


def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    return file.read().decode("utf-8")


def split_text_into_chunks(text, max_chars=900):
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) < max_chars:
            current += p + "\n\n"
        else:
            chunks.append(current.strip())
            current = p + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def clean_chunk(chunk):
    lines = [line.strip() for line in chunk.strip().replace("\t", " ").splitlines() if line.strip()]
    return "\n".join(lines)


@st.cache_resource
def load_models():
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=-1)
    return encoder, summarizer


encoder_model, summarizer_pipe = load_models()

if "store" not in st.session_state:
    st.session_state.store = RollingChunkStore(max_chunks=2000)
    st.session_state.faiss_index = None
    st.session_state.metadata = []

st.title("Hackathon2 – RAG local Streamlit")

# Bouton pour vider le cache et réinitialiser
col1, col2 = st.columns([3, 1])
with col2:
    reset_clicked = st.button("🔄 Réinitialiser", help="Vide le cache et réinitialise la session", key="reset_btn")
    
if reset_clicked:
    # Vider tous les caches de session
    for key in list(st.session_state.keys()):
        if key != 'reset_btn':
            del st.session_state[key]
    # Réinitialiser les valeurs essentielles
    st.session_state.store = RollingChunkStore(max_chunks=2000)
    st.session_state.faiss_index = None
    st.session_state.metadata = []
    st.success("✅ Session réinitialisée ! Veuillez ré-uploader vos documents.")
    st.rerun()

uploaded_files = st.file_uploader(
    "Déposez vos PDF/TXT (traitement immédiat)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Ingestion de {uploaded_file.name}..."):
            raw_text = extract_text(uploaded_file)
            chunks = split_text_into_chunks(raw_text)
            clean_chunks = [clean_chunk(c) for c in chunks]

            st.session_state.store.add_document(uploaded_file.name, clean_chunks)
            index, embeddings, metadata = rebuild_index_from_store(
                st.session_state.store,
                encoder_model,
                batch_size=4,
            )
            st.session_state.faiss_index = index
            st.session_state.metadata = metadata
        st.success(f"{uploaded_file.name} ingéré ({len(clean_chunks)} chunks)")

st.subheader("Recherche & Résumé")
query = st.text_input("Votre requête")

if st.button("Lancer la recherche") and query:
    if st.session_state.faiss_index is None:
        st.warning("Veuillez d'abord ingérer au moins un document.")
    else:
        query_emb = encoder_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_emb)
        distances, indices = st.session_state.faiss_index.search(query_emb, 5)

        results = []
        seen_chunks = set()  # Pour éviter les doublons
        seen_indices = set()  # Pour éviter les doublons d'indices FAISS
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx in seen_indices:
                continue
            seen_indices.add(idx)
            
            if idx >= len(st.session_state.metadata):
                continue
                
            meta = st.session_state.metadata[idx]
            # Créer une clé unique pour ce chunk (source + chunk_id)
            chunk_key = (meta.source, meta.chunk_id)
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                results.append({"score": float(dist), "source": meta.source, "chunk": meta.chunk_id, "text": meta.text})

        if not results:
            st.info("Aucun passage pertinent trouvé.")
        else:
            for i, res in enumerate(results, 1):
                st.markdown(f"**Résultat {i} — {res['source']} (chunk {res['chunk']}) | score={res['score']:.3f}**")
                st.write(res["text"])

            # Générer un résumé seulement si le texte total dépasse 200 caractères
            total_text_length = sum(len(r["text"]) for r in results)
            if total_text_length > 200:
                mini_summaries = summarize_in_batches(
                    summarizer_pipe,
                    [r["text"] for r in results],
                    batch_size=3,
                    instruction="summarize: ",
                )
                final_summary = " ".join(mini_summaries)
                st.success("Résumé :")
                st.write(final_summary)
            else:
                st.info("Texte trop court pour générer un résumé significatif. Les résultats ci-dessus constituent déjà un résumé.")

