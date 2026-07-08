'''
/ai/rag/devTests.py
-> manual test harness for the exploration RAG pipeline (not imported by the Flask app).
   Run with: python -m ai.rag.devTests [--no-llm]
'''

import sys
import networkx as nx
import matplotlib.pyplot as plt

from ai.rag.knowledgeGraph import (
    create_knowledge_graph_from_data,
    build_relationships_dict,
    build_artist_projection,
    build_hierarchical_communities,
)
from ai.rag.retrieval import (
    build_text_index,
    find_related_artists,
    _search_text_fields,
    _score_communities,
    _tokenize,
    _is_specific_entity,
)
from ai.rag.ragQuery import answer_question_with_exploration
from ai.rag.setup import loadProjectRows


def visualize_graph(graph, title="Knowledge Graph"):
    """
    Creates and displays a visualization of the graph with tuple-based nodes.
    """
    if not graph or graph.number_of_nodes() == 0:
        print("Graph is empty or invalid. Cannot visualize.")
        return

    plt.figure(figsize=(40, 40))
    pos = nx.kamada_kawai_layout(graph)

    labels = {node: str(node) for node in graph.nodes()}
    node_colors = []
    color_map = {
        'Artist': 'lightblue', 'Theme': 'lightgreen', 'Instrument': 'lightcoral',
        'Category': 'gold', 'Municipality': 'plum', 'District': 'slateblue',
        'Region': 'darkcyan', 'Location': 'lightgrey'
    }
    for _, data in graph.nodes(data=True):
        node_colors.append(color_map.get(data.get('type'), 'grey'))

    nx.draw(graph, pos, labels=labels, with_labels=True, node_color=node_colors,
            node_size=3000, font_size=10, width=1.5, edge_color='gray')
    edge_labels = nx.get_edge_attributes(graph, 'relationship')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    plt.title(title, fontsize=30)
    plt.show()


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

TEST_QUERIES_PT = [
    # --- Direct artist hit ---
    "Quem e o Jorge Cruz e que tipo de musica faz?",
    "Fala-me sobre a Velha Gaiteira.",
    "O que e A Musica Portuguesa a Gostar Dela Propria?",
    "Quem e Dazkarieh?",
    "Conta-me sobre Celina da Piedade.",

    # --- Region/geography ---
    "Que artistas existem no Alentejo?",
    "Quem sao os artistas dos Acores?",
    "Que musica tradicional existe no Minho?",
    "Ha artistas de Tras-os-Montes e Alto Douro?",
    "Que projetos existem na Beira Alta?",

    # --- Instrument queries ---
    "Quem toca gaita de foles?",
    "Que artistas usam a viola amarantina?",
    "Onde se toca adufe em Portugal?",
    "Quem toca guitarra portuguesa?",
    "Que artistas tocam concertina?",

    # --- Category/genre queries ---
    "Que musica instrumental existe no arquivo?",
    "Mostra-me exemplos de musica narrativa.",
    "Existem registos de musica de trabalho?",
    "Ha exemplos de musica infantil tradicional?",

    # --- Abstract / disruptive concept queries ---
    "Musica que referencie pastel de nata.",
    "Que artistas falam sobre gastronomia ou comida nas suas biografias?",
    "Que ligacoes existem entre musica e religiao neste arquivo?",
    "Ha musica relacionada com festas e romarias?",
    "Que artistas mencionam tradicoes rurais nas suas historias?",
    "Existe musica ligada ao mar e aos pescadores?",
    "Que referencias a santos e procissoes existem?",
    "Ha artistas cuja historia mencione emigracao?",

    # --- Broad thematic queries ---
    "Qual e a relacao entre a gaita de foles e as tradicoes beiras?",
    "Que instrumentos sao tipicos do folclore do norte de Portugal?",
    "Que papel tem a voz na musica tradicional portuguesa?",

    # --- Edge cases ---
    "Que artistas tocam em Lisboa?",
    "Existem registos de fado neste arquivo?",
    "Qual e a historia do cante alentejano?",
]

TEST_QUERIES_EN = [
    "Who is O Povo Que Ainda Canta and what is it related to?",
    "Which instruments are commonly played in the Alentejo region?",
    "Tell me about artists from the Acores.",
    "What is the role of the adufe in Portuguese folk music?",
    "Are there any films or documentaries in the archive?",
]


def _diagnose_exploration(question, artist_relationships, name_to_artists,
                          artist_proj, community_hierarchy,
                          text_token_index, artist_texts):
    """Runs the exploration pipeline and prints what each stage found."""
    question_lower = question.lower()
    question_tokens = _tokenize(question)
    artist_names_lower = {n.lower(): n for n in artist_relationships}

    # Stage 1: artist name
    artist_hit = None
    for name_lower in sorted(artist_names_lower.keys(), key=len, reverse=True):
        if name_lower in question_lower:
            artist_hit = artist_names_lower[name_lower]
            break

    if artist_hit:
        print(f"  [S1 Artist] '{artist_hit}'")
        related = find_related_artists(artist_proj, artist_hit, top_k=3)
        if related:
            print(f"  [Related] {', '.join(related)}")
    else:
        print(f"  [S1 Artist] (no match)")

    # Stage 2: entity name
    if not artist_hit:
        entity_hits = []
        for name_lower in sorted(name_to_artists.keys(), key=len, reverse=True):
            if not _is_specific_entity(name_lower):
                continue
            if name_lower in question_lower and name_lower not in artist_names_lower:
                artists = sorted(name_to_artists[name_lower])[:5]
                entity_hits.append((name_lower, artists))
                break
        if entity_hits:
            for name, artists in entity_hits:
                print(f"  [S2 Entity] '{name}' -> {', '.join(artists)}")
        else:
            print(f"  [S2 Entity] (no match)")

    # Stage 3: text search
    text_results = _search_text_fields(
        question_tokens, text_token_index, artist_texts, max_results=5
    )
    if text_results:
        print(f"  [S3 Text] {len(text_results)} artists found in text fields:")
        for name, field_snippets, score in text_results[:3]:
            fields = ', '.join(field_snippets.keys())
            preview = ''
            for snippets in field_snippets.values():
                if snippets:
                    preview = snippets[0][:80]
                    break
            print(f"    {name} (score={score}, fields={fields}) \"{preview}...\"")
    else:
        print(f"  [S3 Text] (no matches)")

    # Stage 4: community
    if not artist_hit and not text_results and community_hierarchy:
        for res in sorted(community_hierarchy.keys(), reverse=True):
            hits = _score_communities(question_tokens, community_hierarchy[res])
            if hits:
                _, top = hits[0]
                print(f"  [S4 Community] res={res}, {len(top['artists'])} artists: "
                      f"{', '.join(top['artists'][:5])}")
                print(f"  [Token Overlap] {question_tokens & top['tokens']}")
                break
        else:
            print(f"  [S4 Community] (no match)")

    print(f"  [Tokens] {question_tokens}")


def run_tests(artist_relationships, name_to_artists, artist_proj,
              community_hierarchy, text_token_index, artist_texts,
              call_llm=True):
    all_queries = (
        [("PT", q) for q in TEST_QUERIES_PT] +
        [("EN", q) for q in TEST_QUERIES_EN]
    )

    print(f"\n{'='*70}")
    print(f"  EXPLORATION TEST SUITE -- {len(all_queries)} queries")
    print(f"  Mode: {'retrieval + LLM' if call_llm else 'retrieval diagnostics only'}")
    print(f"{'='*70}")

    for i, (lang, question) in enumerate(all_queries, 1):
        print(f"\n[{i:02d}/{len(all_queries):02d}] [{lang}] {question}")
        _diagnose_exploration(
            question, artist_relationships, name_to_artists,
            artist_proj, community_hierarchy,
            text_token_index, artist_texts,
        )

        if call_llm:
            answer = answer_question_with_exploration(
                question, artist_relationships, name_to_artists,
                artist_proj, community_hierarchy,
                text_token_index, artist_texts,
            )
            print(f"\n  --- LLM Answer ---\n{answer}\n  --- End ---")

    print(f"\n{'='*70}")
    print(f"  TEST SUITE COMPLETE")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Standalone run needs a Flask app context to query the Project model.
    from flask import Flask
    from database.setup import initDatabase

    app = Flask(__name__)
    initDatabase(app)

    with app.app_context():
        all_rows = loadProjectRows()

        if not all_rows:
            print("No data found in the database. Please ensure the database is populated.")
        else:
            graph = create_knowledge_graph_from_data(all_rows)
            artist_relationships, name_to_artists = build_relationships_dict(graph)

            artist_proj = build_artist_projection(graph)
            community_hierarchy = build_hierarchical_communities(artist_proj, graph)

            text_token_index, artist_texts = build_text_index(artist_relationships)

            total_communities = sum(len(c) for c in community_hierarchy.values())
            print(f"Graph: {graph.number_of_nodes()} nodes | "
                  f"{artist_proj.number_of_edges()} artist-artist edges | "
                  f"{len(artist_relationships)} artists indexed | "
                  f"{total_communities} communities across {len(community_hierarchy)} levels | "
                  f"{len(artist_texts)} artists with text | "
                  f"{len(text_token_index)} text tokens indexed.")

            for res, comms in sorted(community_hierarchy.items()):
                print(f"  Resolution {res}: {len(comms)} communities")

            call_llm = '--no-llm' not in sys.argv
            run_tests(artist_relationships, name_to_artists, artist_proj,
                      community_hierarchy, text_token_index, artist_texts,
                      call_llm=call_llm)
