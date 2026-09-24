'''
/ai/rag/setup.py
-> builds and caches the artist knowledge graph state used by the exploration RAG pipeline
'''

import threading
from dataclasses import dataclass

from ai.rag.knowledgeGraph import (
    create_knowledge_graph_from_data,
    build_relationships_dict,
    build_artist_projection,
    build_hierarchical_communities,
)
from ai.rag.retrieval import build_text_index
from ai.rag.embeddings import load_chunk_index

# ==================================================
# global vars
# ==================================================

_lock = threading.Lock()
_state = None


@dataclass
class RagState:
    artist_relationships: dict
    name_to_artists: dict
    artist_proj: object
    community_hierarchy: dict
    text_token_index: dict
    artist_texts: dict
    chunk_vectors: object    # numpy matrix, or None if no embeddings cache exists yet
    chunk_meta: list


# ==================================================
# methods
# ==================================================

def loadProjectRows():
    """
    Loads all projects from the SQLAlchemy Project model and maps them to the
    field names expected by ai.rag.knowledgeGraph.create_knowledge_graph_from_data.
    Must be called inside an app context.
    """
    from database.models import Project

    return [
        {
            'id':            project.id,
            'link':          project.link,
            'Nome':          project.author,
            'Tema':          project.title,
            'Instrumentos':  project.instruments,
            'Categoria':     project.category,
            'Local':         project.location,
            'Concelho':      project.municipality,
            'Distrito/Ilha': project.district,
            'Região':        project.region,
            'Data':          project.date.isoformat() if project.date else None,
            'keywords':      project.keywords,
            'history':       project.history,
            'other_info':    project.other_info,
            'biographies':   project.biographies,
            'audio_transcription': project.audio_transcription or '',
            'visual_description':  project.visual_description or '',
        }
        for project in Project.query.all()
    ]


def _drop_stale_chunks(chunk_vectors, chunk_meta, artist_relationships):
    """
    The chunk embeddings cache is built by a separate, manually-triggered
    script (database/buildChunkEmbeddings.py) and can lag behind the live
    projects table. Drop any chunk whose artist no longer exists in the
    freshly-rebuilt graph so a renamed/removed artist can't surface a
    dangling reference in the prompt.
    """
    keep = [i for i, m in enumerate(chunk_meta) if m['artist_name'] in artist_relationships]
    if len(keep) == len(chunk_meta):
        return chunk_vectors, chunk_meta
    return chunk_vectors[keep], [chunk_meta[i] for i in keep]


def _build_state():
    data_rows = loadProjectRows()

    graph = create_knowledge_graph_from_data(data_rows)
    artist_relationships, name_to_artists = build_relationships_dict(graph)
    artist_proj = build_artist_projection(graph)
    community_hierarchy = build_hierarchical_communities(artist_proj, graph)
    text_token_index, artist_texts = build_text_index(artist_relationships)

    chunk_vectors, chunk_meta = load_chunk_index()
    if chunk_vectors is not None:
        chunk_vectors, chunk_meta = _drop_stale_chunks(chunk_vectors, chunk_meta, artist_relationships)

    total_communities = sum(len(c) for c in community_hierarchy.values())
    print(
        f"[RAG] graph: {graph.number_of_nodes()} nodes, "
        f"{artist_proj.number_of_edges()} artist-artist edges, "
        f"{len(artist_relationships)} artists, "
        f"{total_communities} communities across {len(community_hierarchy)} levels, "
        f"{len(chunk_meta)} semantic chunks."
    )

    return RagState(
        artist_relationships=artist_relationships,
        name_to_artists=name_to_artists,
        artist_proj=artist_proj,
        community_hierarchy=community_hierarchy,
        text_token_index=text_token_index,
        artist_texts=artist_texts,
        chunk_vectors=chunk_vectors,
        chunk_meta=chunk_meta,
    )


def initRag(app):
    """Builds the RAG graph state once and caches it. Call at app startup."""
    global _state
    with app.app_context():
        new_state = _build_state()
    with _lock:
        _state = new_state


def rebuildRag(app):
    """Rebuilds the RAG graph state from the current projects table. Safe to
    call from a scheduled job after the projects table has been refreshed."""
    global _state
    with app.app_context():
        new_state = _build_state()
    with _lock:
        _state = new_state


def getRagState():
    if _state is None:
        raise RuntimeError("RAG state not initialized. Call initRag(app) at startup first.")
    return _state
