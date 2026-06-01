import os
import re
import requests
from dotenv import load_dotenv
from knowledge_graph import (
    create_knowledge_graph_from_data,
    build_relationships_dict,
    build_leiden_communities,
    fetch_data_from_csv,
    fetch_data_from_db,
)

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'which',
    'who', 'where', 'how', 'tell', 'me', 'about', 'of', 'in', 'on',
    'at', 'to', 'for', 'and', 'or', 'with', 'from', 'that', 'this',
    'do', 'does', 'did', 'have', 'has', 'by', 'be', 'its', 'their',
    'there', 'commonly', 'found', 'related', 'main', 'played',
}


def _tokenize(text):
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return {t for t in cleaned.split() if t not in STOP_WORDS and len(t) > 2}


def _format_artist_context(artist_name, entry):
    """Renders one artist's relationship dict as a readable string for the LLM prompt."""
    lines = [f"Artist: {artist_name}"]
    label_map = {
        'themes':         'Project titles (names of recorded pieces or videos)',
        'instruments':    'Instruments played',
        'categories':     'Music categories',
        'locations':      'Specific locations',
        'municipalities': 'Municipalities',
        'districts':      'Districts / Islands',
        'regions':        'Regions',
    }
    for key, label in label_map.items():
        values = entry.get(key, [])
        if values:
            lines.append(f"  {label}: {', '.join(values)}")
    return '\n'.join(lines)


def retrieve_relevant_context(
    question,
    artist_relationships,
    name_to_artists,
    leiden_communities=None,  # optional — falls back gracefully if not supplied
    max_artists=10,
):
    """
    Returns a list of formatted context strings relevant to the question.

    Stage 1 – direct name match:
        Scan every key in name_to_artists (artist names + all connected entity
        names) against the question text.  Collect the matching artists.

    Stage 2 – Leiden community match (only if leiden_communities is provided):
        Score each community by how many question tokens overlap with its
        pooled entity token bag.  Communities scoring >= 2 are selected and
        all their artists are returned, preceded by the community summary so
        the LLM has cluster-level context.

    Stage 3 – keyword fallback:
        Tokenise the question and each artist entry; keep entries with at least
        2 overlapping meaningful tokens.
    """
    MIN_COMMUNITY_OVERLAP = 2

    question_lower  = question.lower()
    question_tokens = _tokenize(question)
    matched_artists = set()

    # --- Stage 1: direct substring match on any known name ---
    for name_lower, artists in name_to_artists.items():
        if name_lower in question_lower:
            matched_artists.update(artists)

    if matched_artists:
        return [
            _format_artist_context(a, artist_relationships[a])
            for a in sorted(matched_artists)[:max_artists]
            if a in artist_relationships
        ]

    # --- Stage 2: Leiden community match ---
    if leiden_communities:
        community_hits = []
        for community in leiden_communities:
            overlap = len(question_tokens & community['tokens'])
            if overlap >= MIN_COMMUNITY_OVERLAP:
                community_hits.append((overlap, community))

        if community_hits:
            community_hits.sort(key=lambda x: x[0], reverse=True)

            seen_artists = set()
            context_blocks = []

            for _, community in community_hits:
                # Prepend the community summary for cluster-level context
                context_blocks.append(community['summary'])

                for artist_name in community['artists']:
                    if artist_name not in seen_artists and artist_name in artist_relationships:
                        context_blocks.append(
                            _format_artist_context(artist_name, artist_relationships[artist_name])
                        )
                        seen_artists.add(artist_name)
                        if len(seen_artists) >= max_artists:
                            break

                if len(seen_artists) >= max_artists:
                    break

            if context_blocks:
                return context_blocks

    # --- Stage 3: keyword overlap fallback ---
    scored = []
    for artist_name, entry in artist_relationships.items():
        # Build a bag-of-words from all values in the entry
        all_text = artist_name + ' ' + ' '.join(
            v for values in entry.values() for v in values
        )
        overlap = question_tokens & _tokenize(all_text)
        if len(overlap) >= 2:
            scored.append((len(overlap), artist_name))

    scored.sort(reverse=True)
    return [
        _format_artist_context(a, artist_relationships[a])
        for _, a in scored[:max_artists]
    ]


# ---------------------------------------------------------------------------
# RAG answer
# ---------------------------------------------------------------------------

def answer_question_with_rag(question, artist_relationships, name_to_artists,
                              leiden_communities=None):
    context_blocks = retrieve_relevant_context(
        question, artist_relationships, name_to_artists, leiden_communities
    )

    if not context_blocks:
        return "I couldn't find any relevant information in the graph to answer that question."

    context = '\n\n'.join(context_blocks)

    prompt = f"""You are an assistant with knowledge about Portuguese folk music artists.
Use only the information provided below to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    if not OLLAMA_URL:
        return "Error: OLLAMA_URL is not set in the environment."

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': 'llama3.1:8b',
                'prompt': prompt,
                'stream': False,
                'keep_alive': -1,
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get('response', 'No response content found.')
        return f"Error from Ollama API: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def print_artist_entry(artist_name, artist_relationships):
    """Pretty-prints the relationship dict entry for one artist."""
    entry = artist_relationships.get(artist_name)
    if not entry:
        print(f"Artist '{artist_name}' not found in relationships dict.")
        return
    print(_format_artist_context(artist_name, entry))


def run_tests(artist_relationships, name_to_artists, leiden_communities=None):
    test_questions = [
        "What are the main themes related to the artist Tocandar?",
        "Which instruments are commonly played in the Alentejo region?",
        "Tell me about the categories of music found in Lisboa.",
        "Who are the artists from the Açores?",
    ]

    print("\n--- Running GraphRAG Tests ---")
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        answer = answer_question_with_rag(
            question, artist_relationships, name_to_artists, leiden_communities
        )
        print(f"Answer: {answer}")
    print("\n--- Tests Complete ---")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    all_rows = fetch_data_from_csv()

    if not all_rows:
        print("No data found.")
    else:
        graph = create_knowledge_graph_from_data(all_rows)
        artist_relationships, name_to_artists = build_relationships_dict(graph)
        leiden_communities = build_leiden_communities(graph, resolution=1.0)

        print(f"Graph: {graph.number_of_nodes()} nodes | "
              f"{len(artist_relationships)} artists indexed | "
              f"{len(leiden_communities)} Leiden communities.")

        # Inspect a single artist before running tests
        print_artist_entry("Tocandar", artist_relationships)

        run_tests(artist_relationships, name_to_artists, leiden_communities)