'''
/ai/rag/runLoggedTests.py
-> runs a representative set of test questions through the full retrieval +
   LLM pipeline and logs question/diagnostics/answer to a markdown file.
   Standalone, throwaway QA script -- not imported by the app or devTests.py.
   Run with: python -m ai.rag.runLoggedTests
'''
import sys
import time

from ai.rag.knowledgeGraph import (
    create_knowledge_graph_from_data,
    build_relationships_dict,
    build_artist_projection,
    build_hierarchical_communities,
)
from ai.rag.retrieval import build_text_index, _semantic_search, _tokenize, SEMANTIC_PROMOTE_THRESHOLD
from ai.rag.ragQuery import answer_question_with_exploration
from ai.rag.setup import loadProjectRows
from ai.rag.embeddings import load_chunk_index

LOG_PATH = r"c:\Users\joanm\AppData\Local\Temp\claude\c--Users-joanm-Documents-LASTRO-lastro-mpagdp-copy\782d0aff-5419-494e-a517-aeae4577c8e9\scratchpad\rag_test_log_claude_v9_rechunk_embeddings.md"

QUERIES = [
    ("Direct artist", "Quem e Dazkarieh?"),
    ("Region", "Que artistas existem no Alentejo?"),
    ("Instrument", "Quem toca gaita de foles?"),
    ("Category/genre", "Mostra-me exemplos de musica narrativa."),
    ("Abstract/disruptive", 'Musica que referencie pastel de nata.'),
    ("Abstract/disruptive", "Ha artistas cuja historia mencione emigracao?"),
    ("Broad thematic", "Que papel tem a voz na musica tradicional portuguesa?"),
    ("Edge case", "Existem registos de fado neste arquivo?"),
    ("Multi-artist comparison", "Qual e a diferenca entre Celio Pires e Antonio Andre?"),
    ("Paraphrase", "Que musica fala sobre o cheiro do pao a cozer no forno?"),
    ("Paraphrase", "Ha relatos de saudade de quem foi trabalhar para o estrangeiro?"),
    ("Paraphrase", "Que videos mencionam bruxas ou feiticaria?"),
    ("Paraphrase", "Ha video que mostrem criancas a brincar na rua?"),
    ("Paraphrase", "Que musica transmite um sentimento de perda irreparavel?"),
    ("Paraphrase", "Que artistas descrevem rituais ou bencaos praticadas pelos mais velhos?"),
    ("Paraphrase", "Existem historias sobre pratos tipicos passados de geracao em geracao?"),
]


def _log(f, text=""):
    f.write(text + "\n")
    f.flush()


def run():
    all_rows = loadProjectRows()
    graph = create_knowledge_graph_from_data(all_rows)
    artist_relationships, name_to_artists = build_relationships_dict(graph)
    artist_proj = build_artist_projection(graph)
    community_hierarchy = build_hierarchical_communities(artist_proj, graph)
    text_token_index, artist_texts = build_text_index(artist_relationships)
    chunk_vectors, chunk_meta = load_chunk_index()

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        _log(f, f"# RAG test log -- {len(QUERIES)} queries")
        _log(f, f"\nGraph: {graph.number_of_nodes()} nodes, {len(artist_relationships)} artists, "
                f"{len(chunk_meta)} semantic chunks loaded.\n")

        for i, (category, question) in enumerate(QUERIES, 1):
            print(f"[{i}/{len(QUERIES)}] ({category}) {question}")
            _log(f, f"\n## {i}. [{category}] {question}\n")

            question_tokens = _tokenize(question)
            semantic_hits = _semantic_search(question, chunk_vectors, chunk_meta, top_k=3)
            if semantic_hits:
                top_score, top_meta = semantic_hits[0]
                promote = " (would PROMOTE)" if top_score >= SEMANTIC_PROMOTE_THRESHOLD else ""
                _log(f, f"*Top semantic hit: score={top_score:.3f}{promote} "
                        f"[{top_meta['artist_name']}] {top_meta['field']}*\n")
            else:
                _log(f, "*No semantic hits above discovery threshold*\n")

            t0 = time.time()
            try:
                answer = answer_question_with_exploration(
                    question, artist_relationships, name_to_artists,
                    community_hierarchy, text_token_index, artist_texts,
                    chunk_vectors=chunk_vectors, chunk_meta=chunk_meta,
                )
            except Exception as e:
                answer = f"ERROR: {e}"
            elapsed = time.time() - t0

            _log(f, f"*Generation time: {elapsed:.1f}s*\n")
            _log(f, "**Answer:**\n")
            _log(f, answer)
            _log(f, "\n---")

        _log(f, "\n# Log complete")

    print(f"\nDone. Log written to {LOG_PATH}")


if __name__ == '__main__':
    from flask import Flask
    from database.setup import initDatabase

    app = Flask(__name__)
    initDatabase(app)

    with app.app_context():
        run()
