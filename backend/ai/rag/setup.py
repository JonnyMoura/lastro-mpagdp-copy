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
            'keywords':      project.keywords,
            'history':       project.history,
            'other_info':    project.other_info,
            'biographies':   project.biographies,
        }
        for project in Project.query.all()
    ]


def _build_state():
    data_rows = loadProjectRows()

    graph = create_knowledge_graph_from_data(data_rows)
    artist_relationships, name_to_artists = build_relationships_dict(graph)
    artist_proj = build_artist_projection(graph)
    community_hierarchy = build_hierarchical_communities(artist_proj, graph)
    text_token_index, artist_texts = build_text_index(artist_relationships)

    total_communities = sum(len(c) for c in community_hierarchy.values())
    print(
        f"[RAG] graph: {graph.number_of_nodes()} nodes, "
        f"{artist_proj.number_of_edges()} artist-artist edges, "
        f"{len(artist_relationships)} artists, "
        f"{total_communities} communities across {len(community_hierarchy)} levels."
    )

    return RagState(
        artist_relationships=artist_relationships,
        name_to_artists=name_to_artists,
        artist_proj=artist_proj,
        community_hierarchy=community_hierarchy,
        text_token_index=text_token_index,
        artist_texts=artist_texts,
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
