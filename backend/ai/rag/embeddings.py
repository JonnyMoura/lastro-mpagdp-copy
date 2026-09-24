'''
/ai/rag/embeddings.py
-> calls Ollama's embedding endpoint and persists/loads the chunk vector index
'''
import os
import json
import time

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_EMBED_URL = os.getenv('OLLAMA_EMBED_URL')
EMBED_MODEL = 'bge-m3'

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'database', 'embeddings')
VECTORS_PATH = os.path.join(OUT_DIR, 'chunk_vectors.npy')
META_PATH = os.path.join(OUT_DIR, 'chunk_meta.json')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest.json')


def embed_texts(texts, timeout=60):
    """Embeds a batch of strings in one Ollama call. Returns a list of
    embedding vectors (list[float]), one per input, in the same order."""
    if not OLLAMA_EMBED_URL:
        raise RuntimeError("OLLAMA_EMBED_URL is not set in the environment.")

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={'model': EMBED_MODEL, 'input': texts, 'keep_alive': -1},
        timeout=timeout,
    )
    response.raise_for_status()
    embeddings = response.json()['embeddings']
    if len(embeddings) != len(texts):
        raise RuntimeError(f"got {len(embeddings)} embeddings for {len(texts)} inputs")
    return embeddings


def embed_query(question, timeout=15):
    """Embeds a single question at request time. Short timeout on purpose --
    this is one short string, not a multi-thousand-token generation, so it
    must fail fast rather than hang the request the way ragQuery.py's
    generation call intentionally allows up to 480s."""
    return embed_texts([question], timeout=timeout)[0]


def save_chunk_index(vectors, meta):
    """Persists an L2-normalized chunk vector matrix + parallel metadata to
    disk, so query time only needs one dot product with no re-normalization."""
    os.makedirs(OUT_DIR, exist_ok=True)

    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    np.save(VECTORS_PATH, vectors)
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'model': EMBED_MODEL,
            'dim': int(vectors.shape[1]),
            'n_chunks': int(vectors.shape[0]),
            'built_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }, f, indent=2)


def load_chunk_index():
    """Loads the cached chunk vector index. Returns (None, []) on any
    problem (missing cache, shape mismatch, corrupt file) so semantic
    search degrades to a no-op instead of crashing initRag/rebuildRag."""
    try:
        with open(MANIFEST_PATH, encoding='utf-8') as f:
            manifest = json.load(f)
        vectors = np.load(VECTORS_PATH)
        with open(META_PATH, encoding='utf-8') as f:
            meta = json.load(f)

        if vectors.shape[0] != len(meta) or vectors.shape[1] != manifest['dim']:
            raise ValueError('shape mismatch between vectors/meta/manifest')

        return vectors, meta
    except FileNotFoundError:
        print('[RAG] no chunk embeddings cache -- run database/buildChunkEmbeddings.py')
        return None, []
    except Exception as e:
        print(f'[RAG] chunk embeddings cache load failed, semantic search disabled: {e}')
        return None, []
