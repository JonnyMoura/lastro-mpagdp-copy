'''
/ai/rag/retrieval.py
-> multi-stage retrieval pipeline over the artist knowledge graph (tokenization,
   text index, community/entity/text matching) used to build LLM context
'''

import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'which',
    'who', 'where', 'how', 'tell', 'me', 'about', 'of', 'in', 'on',
    'at', 'to', 'for', 'and', 'or', 'with', 'from', 'that', 'this',
    'do', 'does', 'did', 'have', 'has', 'by', 'be', 'its', 'their',
    'there', 'commonly', 'found', 'related', 'main', 'played',
}

STOP_WORDS_PT = {
    'que', 'qual', 'quais', 'quem', 'como', 'onde', 'quando', 'porque',
    'sobre', 'uma', 'uns', 'umas', 'para', 'por', 'com', 'sem', 'entre',
    'nos', 'nas', 'dos', 'das', 'aos', 'pelo', 'pela', 'mais', 'muito',
    'sua', 'seu', 'seus', 'suas', 'num', 'numa', 'nao', 'sim', 'era',
    'essa', 'esse', 'esses', 'essas', 'esta', 'este', 'estes', 'estas',
    'isso', 'isto', 'aquilo', 'tem', 'ser', 'sao', 'foi', 'eram',
    'pode', 'podem', 'deve', 'fazer', 'faz', 'outro', 'outros', 'outra',
    'outras', 'ainda', 'tambem', 'mesmo', 'tipo', 'existem', 'existe',
    'mostra', 'fala', 'conta', 'diz', 'diga',
}

ALL_STOP_WORDS = STOP_WORDS | STOP_WORDS_PT


def _tokenize(text):
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return {t for t in cleaned.split() if t not in ALL_STOP_WORDS and len(t) > 2}


def _format_artist_context(artist_name, entry):
    """Renders one artist's relationship dict as a readable string for the LLM prompt."""
    lines = [f"Artist: {artist_name}"]
    label_map = {
        'themes':         'Project titles (names of recorded pieces or videos)',
        'instruments':    'Instruments played',
        'categories':     'Music categories',
        'keywords':       'Keywords',
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


# ---------------------------------------------------------------------------
# Text field index for abstract concept search
# ---------------------------------------------------------------------------

TEXT_FIELDS = ('history', 'other_info', 'biographies', 'keywords')


def build_text_index(artist_relationships):
    """
    Builds an inverted index: token -> {artist_names} across the text fields
    (history, other_info, biographies, keywords).
    Also returns artist_texts: artist_name -> {field: full_text}.
    """
    token_to_artists = defaultdict(set)
    artist_texts = {}

    for artist_name, entry in artist_relationships.items():
        texts = {}
        for field in TEXT_FIELDS:
            raw = entry.get(field, '')
            if isinstance(raw, list):
                raw = ', '.join(raw)
            if raw:
                texts[field] = raw
                for tok in _tokenize(raw):
                    token_to_artists[tok].add(artist_name)
        if texts:
            artist_texts[artist_name] = texts

    return dict(token_to_artists), artist_texts


def _extract_snippets(text, question_tokens, max_sentences=5):
    """Extracts sentences from text that contain at least one query token."""
    sentences = re.split(r'[.!?\n]+', text)
    matching = []
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 10:
            continue
        s_lower = s.lower()
        if any(tok in s_lower for tok in question_tokens):
            matching.append(s)
            if len(matching) >= max_sentences:
                break
    return matching


def _search_text_fields(question_tokens, text_token_index, artist_texts,
                        max_results=10):
    """
    Searches artist text fields for abstract concept matches.
    Returns: list of (artist_name, {field: [snippets]}, score)
    """
    artist_scores = Counter()
    for token in question_tokens:
        for artist_name in text_token_index.get(token, set()):
            artist_scores[artist_name] += 1

    results = []
    for artist_name, score in artist_scores.most_common(max_results * 2):
        if score < 1:
            break
        field_snippets = {}
        for field, text in artist_texts.get(artist_name, {}).items():
            snippets = _extract_snippets(text, question_tokens)
            if snippets:
                field_snippets[field] = snippets
        if field_snippets:
            results.append((artist_name, field_snippets, score))
            if len(results) >= max_results:
                break

    return results


# ---------------------------------------------------------------------------
# Related artist discovery via the projection graph
# ---------------------------------------------------------------------------

def find_related_artists(artist_proj, artist_name, top_k=5):
    """
    Finds the most similar artists by edge weight in the artist projection
    graph. Higher weight = more shared rare entities.
    """
    artist_node = (artist_name, 'Artist')
    if artist_node not in artist_proj:
        return []

    neighbors = [
        (artist_proj.nodes[nb].get('name', ''), artist_proj[artist_node][nb].get('weight', 0))
        for nb in artist_proj.neighbors(artist_node)
    ]
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in neighbors[:top_k]]


# ---------------------------------------------------------------------------
# Geographic neighbor discovery (graph-driven, no hardcoded keywords)
# ---------------------------------------------------------------------------

FIELD_LABELS = {
    'history': 'Historia', 'other_info': 'Notas',
    'biographies': 'Biografia', 'keywords': 'Palavras-chave',
}


def _find_geographic_neighbors(primary_artists, artist_relationships,
                               artist_texts, max_results=8):
    """
    Uses the knowledge graph's geographic structure to find entries from the
    same locations/regions as the primary artists. Returns their raw text
    content so the LLM can discover cross-domain connections.
    """
    primary_set = set(primary_artists)

    geo_values = set()
    for a in primary_artists:
        entry = artist_relationships.get(a, {})
        for field in ('locations', 'municipalities', 'districts', 'regions'):
            geo_values.update(entry.get(field, []))

    if not geo_values:
        return []

    geo_tokens = set()
    for v in geo_values:
        geo_tokens.update(_tokenize(v))

    candidates = []
    for artist_name, texts in artist_texts.items():
        if artist_name in primary_set:
            continue

        entry = artist_relationships.get(artist_name, {})
        artist_geo = set()
        artist_geo_names = set()
        for field in ('locations', 'municipalities', 'districts', 'regions'):
            for name in entry.get(field, []):
                artist_geo.update(_tokenize(name))
                artist_geo_names.add(name)

        overlap = geo_tokens & artist_geo
        if not overlap:
            continue

        shared_geo = ''
        for field in ('locations', 'municipalities', 'districts', 'regions'):
            for name in entry.get(field, []):
                if name in geo_values:
                    shared_geo = name
                    break
            if shared_geo:
                break

        candidates.append((artist_name, texts, shared_geo, len(overlap)))

    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates[:max_results]


def _find_concept_neighbors(primary_artists, artist_relationships,
                            artist_texts, max_results=8):
    """
    Walks the knowledge graph's entity edges (keywords, categories,
    instruments, themes) to find entries that share concepts with the
    primary artists but come from DIFFERENT geographic areas.

    The shared entities tell the LLM *why* the connection exists.
    No hardcoded keyword lists — the graph structure defines relevance.
    """
    primary_set = set(primary_artists)

    # Collect entity values from primary artists
    concept_fields = ('keywords', 'categories', 'instruments', 'themes')
    primary_concepts = defaultdict(set)
    primary_geo = set()
    for a in primary_artists:
        entry = artist_relationships.get(a, {})
        for field in concept_fields:
            for val in entry.get(field, []):
                primary_concepts[val.lower()].add(field)
        for field in ('regions', 'districts'):
            primary_geo.update(entry.get(field, []))

    if not primary_concepts:
        return []

    candidates = []
    for artist_name, texts in artist_texts.items():
        if artist_name in primary_set:
            continue

        entry = artist_relationships.get(artist_name, {})

        # Prefer entries from different regions for cross-domain surprise
        artist_geo = set()
        for field in ('regions', 'districts'):
            artist_geo.update(entry.get(field, []))
        same_region = bool(primary_geo & artist_geo)

        # Count shared concepts and track which ones match
        shared = []
        for field in concept_fields:
            for val in entry.get(field, []):
                if val.lower() in primary_concepts:
                    shared.append(val)

        if not shared:
            continue

        # Score: number of shared concepts, with a bonus for cross-region
        score = len(shared)
        if not same_region and artist_geo:
            score += 2

        candidates.append((artist_name, texts, shared, score))

    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates[:max_results]


def _find_cultural_neighbors(primary_artists, artist_relationships,
                             artist_texts, question_tokens, max_results=8):
    """
    Searches the unstructured text fields (history, other_info, biographies)
    for entries that share cultural/thematic concepts beyond music — e.g.
    gastronomy, religion, festivals, crafts, rural traditions, emigration.

    Unlike _find_concept_neighbors (which only matches structured entity
    fields), this function does full-text token matching so cross-domain
    cultural references surface even when they aren't in keyword/category
    fields.
    """
    primary_set = set(primary_artists)

    primary_text_tokens = set()
    for a in primary_artists:
        texts = artist_texts.get(a, {})
        for field in ('history', 'other_info', 'biographies'):
            text = texts.get(field, '')
            if text:
                primary_text_tokens.update(_tokenize(text))

    search_tokens = (primary_text_tokens | question_tokens) - ALL_STOP_WORDS
    if not search_tokens:
        return []

    candidates = []
    for artist_name, texts in artist_texts.items():
        if artist_name in primary_set:
            continue

        combined_text = ' '.join(
            texts.get(f, '') for f in ('history', 'other_info', 'biographies')
            if texts.get(f, '')
        )
        if not combined_text:
            continue

        artist_tokens = _tokenize(combined_text)
        overlap = search_tokens & artist_tokens
        if len(overlap) < 2:
            continue

        entry = artist_relationships.get(artist_name, {})
        geo = ''
        for f in ('regions', 'districts', 'municipalities'):
            vals = entry.get(f, [])
            if vals:
                geo = vals[0]
                break

        candidates.append((artist_name, texts, sorted(overlap)[:8], len(overlap), geo))

    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates[:max_results]


# ---------------------------------------------------------------------------
# Exploration retrieval pipeline
# ---------------------------------------------------------------------------

ENTITY_BLOCKLIST = {
    'tar', 'ver', 'dar', 'ter', 'voz',
}


def _is_specific_entity(name_lower):
    """Filter out generic single-word entity matches."""
    if name_lower in ENTITY_BLOCKLIST:
        return False
    words = name_lower.split()
    if len(words) >= 2:
        return True
    return len(name_lower) >= 5


def retrieve_exploration_context(
    question,
    artist_relationships,
    name_to_artists,
    artist_proj,
    community_hierarchy,
    text_token_index,
    artist_texts,
    max_direct=5,
    max_discoveries=8,
    max_related=5,
):
    """
    Multi-stage exploration retrieval:

    Stage 1 -- Artist name match (precise)
    Stage 2 -- Entity name match (instruments, regions, categories)
    Stage 3 -- Text field search (biographies, history, other_info)
    Stage 4 -- Community match
    Stage 5 -- Keyword fallback

    Discovery context is built from:
    A) Primary artists' full text (bio, history, notes)
    B) Geographic neighbors' text — other entries from the same
       region/district/location, retrieved via the graph, with raw
       content passed to the LLM to find cross-domain connections
    C) Text search results as fallback

    Returns dict with keys: direct, discoveries, related, community_summary
    """
    question_lower = question.lower()
    question_tokens = _tokenize(question)
    artist_names_lower = {n.lower(): n for n in artist_relationships}

    direct_artists = []
    entity_artists = []

    # --- Stage 1: Artist name match ---
    sorted_artist_names = sorted(artist_names_lower.keys(), key=len, reverse=True)
    for name_lower in sorted_artist_names:
        if name_lower in question_lower:
            direct_artists.append(artist_names_lower[name_lower])
            break

    # --- Stage 2: Entity name match (excluding artist names) ---
    if not direct_artists:
        sorted_entity_names = sorted(name_to_artists.keys(), key=len, reverse=True)
        seen = set()
        for name_lower in sorted_entity_names:
            if not _is_specific_entity(name_lower):
                continue
            if name_lower in question_lower and name_lower not in artist_names_lower:
                for a in sorted(name_to_artists[name_lower]):
                    if a not in seen and a in artist_relationships:
                        entity_artists.append(a)
                        seen.add(a)
                        if len(entity_artists) >= max_direct:
                            break
                if entity_artists:
                    break

    primary_artists = direct_artists or entity_artists

    # --- Stage 3: Text field search ---
    text_results = _search_text_fields(
        question_tokens, text_token_index, artist_texts, max_results=max_discoveries
    )
    primary_set = set(primary_artists)
    text_discoveries = [
        (name, snippets, score) for name, snippets, score in text_results
        if name not in primary_set
    ]

    if not primary_artists and text_discoveries:
        promoted = text_discoveries[:max_direct]
        primary_artists = [name for name, _, _ in promoted]
        primary_set = set(primary_artists)
        text_discoveries = text_discoveries[max_direct:]

    # --- Stage 4: Community match ---
    community_summary = None
    if not primary_artists and community_hierarchy:
        for resolution in sorted(community_hierarchy.keys(), reverse=True):
            communities = community_hierarchy[resolution]
            hits = _score_communities(question_tokens, communities)
            if hits:
                _, top_community = hits[0]
                community_summary = top_community['summary']
                primary_artists = [
                    a for a in top_community['artists'][:max_direct]
                    if a in artist_relationships
                ]
                break

    # --- Stage 5: Keyword fallback ---
    if not primary_artists:
        scored = []
        for artist_name, entry in artist_relationships.items():
            bag_parts = [artist_name]
            for v in entry.values():
                if isinstance(v, list):
                    bag_parts.extend(v)
                elif isinstance(v, str):
                    bag_parts.append(v)
            overlap = question_tokens & _tokenize(' '.join(bag_parts))
            if len(overlap) >= 2:
                scored.append((len(overlap), artist_name))
        scored.sort(reverse=True)
        primary_artists = [name for _, name in scored[:max_direct]]

    # --- Build direct context ---
    direct_blocks = [
        _format_artist_context(a, artist_relationships[a])
        for a in primary_artists
        if a in artist_relationships
    ]

    # --- Build discovery context ---
    discovery_blocks = []
    covered = set(primary_artists)

    # A) Primary artists' full text — send everything, let the LLM reason
    for a in primary_artists:
        if a not in artist_texts:
            continue
        lines = [f"  [Sobre {a}]"]
        for field in ('biographies', 'history', 'other_info'):
            text = artist_texts[a].get(field, '')
            if text:
                label = FIELD_LABELS.get(field, field)
                lines.append(f"    {label}: {text[:1500]}")
        if len(lines) > 1:
            discovery_blocks.append('\n'.join(lines))

    # B) Geographic neighbors — entries from the same region with their own
    #    text content, passed raw so the LLM finds cultural bridges
    geo_neighbors = _find_geographic_neighbors(
        primary_artists, artist_relationships, artist_texts,
        max_results=max_discoveries,
    )
    for artist_name, texts, shared_geo, _ in geo_neighbors:
        if artist_name in covered:
            continue
        covered.add(artist_name)
        entry = artist_relationships.get(artist_name, {})
        geo_info = shared_geo or ''
        lines = [f"  [{artist_name}] (mesma regiao: {geo_info})"]
        for f, label in [('categories', 'Categorias'), ('instruments', 'Instrumentos')]:
            vals = entry.get(f, [])
            if vals:
                lines.append(f"    {label}: {', '.join(vals[:5])}")
        for field in ('biographies', 'history', 'other_info'):
            text = texts.get(field, '')
            if text:
                label = FIELD_LABELS.get(field, field)
                lines.append(f"    {label}: {text[:1000]}")
        if len(lines) > 1:
            discovery_blocks.append('\n'.join(lines))

    # C) Concept neighbors — entries sharing keywords/categories/instruments
    #    but from different regions, with the shared entities labeled so the
    #    LLM understands the conceptual bridge
    concept_neighbors = _find_concept_neighbors(
        primary_artists, artist_relationships, artist_texts,
        max_results=max_discoveries,
    )
    for artist_name, texts, shared_concepts, _ in concept_neighbors:
        if artist_name in covered:
            continue
        covered.add(artist_name)
        entry = artist_relationships.get(artist_name, {})
        bridge = ', '.join(shared_concepts[:5])
        geo = ''
        for f in ('regions', 'districts', 'municipalities'):
            vals = entry.get(f, [])
            if vals:
                geo = vals[0]
                break
        header = f"  [{artist_name}]"
        if geo:
            header += f" ({geo})"
        lines = [header]
        lines.append(f"    Conceitos partilhados: {bridge}")
        for f, label in [('categories', 'Categorias'), ('instruments', 'Instrumentos')]:
            vals = entry.get(f, [])
            if vals:
                lines.append(f"    {label}: {', '.join(vals[:5])}")
        for field in ('biographies', 'history', 'other_info'):
            text = texts.get(field, '')
            if text:
                label = FIELD_LABELS.get(field, field)
                lines.append(f"    {label}: {text[:1000]}")
        if len(lines) > 1:
            discovery_blocks.append('\n'.join(lines))

    # C2) Cultural neighbors — entries sharing thematic concepts in text
    #     fields (history, gastronomy, religion, festivals, crafts, etc.)
    cultural_neighbors = _find_cultural_neighbors(
        primary_artists, artist_relationships, artist_texts,
        question_tokens, max_results=max_discoveries,
    )
    for artist_name, texts, shared_tokens, _, geo in cultural_neighbors:
        if artist_name in covered:
            continue
        covered.add(artist_name)
        entry = artist_relationships.get(artist_name, {})
        bridge = ', '.join(shared_tokens[:6])
        header = f"  [{artist_name}]"
        if geo:
            header += f" ({geo})"
        lines = [header]
        lines.append(f"    Temas culturais partilhados: {bridge}")
        for f, label in [('categories', 'Categorias'), ('keywords', 'Palavras-chave')]:
            vals = entry.get(f, [])
            if vals:
                lines.append(f"    {label}: {', '.join(vals[:6])}")
        for field in ('biographies', 'history', 'other_info'):
            text = texts.get(field, '')
            if text:
                label = FIELD_LABELS.get(field, field)
                lines.append(f"    {label}: {text[:1000]}")
        if len(lines) > 1:
            discovery_blocks.append('\n'.join(lines))

    # D) Text search results as fallback
    for name, field_snippets, _score in text_discoveries:
        if name in covered:
            continue
        covered.add(name)
        lines = [f"  [{name}]"]
        for field, snippets in field_snippets.items():
            label = FIELD_LABELS.get(field, field)
            for snippet in snippets[:3]:
                lines.append(f"    {label}: {snippet.strip()[:600]}")
        discovery_blocks.append('\n'.join(lines))

    # --- Build related context ---
    related_names = []
    if primary_artists:
        related_names = find_related_artists(
            artist_proj, primary_artists[0], top_k=max_related
        )
    related_blocks = [
        _format_artist_context(a, artist_relationships[a])
        for a in related_names
        if a in artist_relationships and a not in set(primary_artists)
    ]

    return {
        'direct': direct_blocks,
        'discoveries': discovery_blocks,
        'related': related_blocks,
        'community_summary': community_summary,
    }


def retrieve_enhanced_context(
    question,
    artist_relationships,
    name_to_artists,
    artist_proj,
    community_hierarchy=None,
    max_related=5,
):
    """
    Stage 1 -- find the directly named artist.
    Stage 2 -- find related artists via projection graph neighbors.

    Returns: (direct_context_blocks, related_context_blocks)
    """
    question_lower = question.lower()

    # --- Stage 1: Find the "Direct Hit" ---
    direct_hit_artist = None
    sorted_names = sorted(name_to_artists.keys(), key=len, reverse=True)

    for name_lower in sorted_names:
        if name_lower in question_lower:
            artists = name_to_artists[name_lower]
            if artists:
                direct_hit_artist = sorted(list(artists))[0]
                break

    if not direct_hit_artist:
        fallback_context = retrieve_relevant_context(
            question, artist_relationships, name_to_artists, community_hierarchy
        )
        return (fallback_context, [])

    direct_context = [
        _format_artist_context(direct_hit_artist, artist_relationships[direct_hit_artist])
    ]

    # --- Stage 2: Related discoveries via projection graph ---
    related_names = find_related_artists(artist_proj, direct_hit_artist, top_k=max_related)

    related_context = [
        _format_artist_context(a, artist_relationships[a])
        for a in related_names
        if a in artist_relationships
    ]

    return direct_context, related_context


def _score_communities(question_tokens, communities):
    """Score communities by token overlap + entity profile depth."""
    hits = []
    for community in communities:
        token_overlap = len(question_tokens & community['tokens'])
        if token_overlap < 2:
            continue

        entity_score = 0
        for counter in community.get('entity_profile', {}).values():
            for name in counter:
                entity_score += len(question_tokens & _tokenize(name))

        total = token_overlap + entity_score * 0.5
        hits.append((total, community))

    hits.sort(key=lambda x: x[0], reverse=True)
    return hits


def retrieve_relevant_context(
    question,
    artist_relationships,
    name_to_artists,
    community_hierarchy=None,
    max_artists=10,
):
    """
    Stage 1 -- direct substring match on any known name.
    Stage 2 -- hierarchical Leiden community match.
    Stage 3 -- keyword overlap fallback.
    """
    question_lower = question.lower()
    question_tokens = _tokenize(question)
    matched_artists = set()

    for name_lower, artists in name_to_artists.items():
        if name_lower in question_lower:
            matched_artists.update(artists)

    if matched_artists:
        return [
            _format_artist_context(a, artist_relationships[a])
            for a in sorted(matched_artists)[:max_artists]
            if a in artist_relationships
        ]

    if community_hierarchy:
        for resolution in sorted(community_hierarchy.keys(), reverse=True):
            communities = community_hierarchy[resolution]
            hits = _score_communities(question_tokens, communities)
            if not hits:
                continue
            seen_artists = set()
            context_blocks = []
            for _, community in hits:
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

    scored = []
    for artist_name, entry in artist_relationships.items():
        bag_parts = [artist_name]
        for v in entry.values():
            if isinstance(v, list):
                bag_parts.extend(v)
            elif isinstance(v, str):
                bag_parts.append(v)
        all_text = ' '.join(bag_parts)
        overlap = question_tokens & _tokenize(all_text)
        if len(overlap) >= 2:
            scored.append((len(overlap), artist_name))

    scored.sort(reverse=True)
    return [
        _format_artist_context(a, artist_relationships[a])
        for _, a in scored[:max_artists]
    ]


def print_artist_entry(artist_name, artist_relationships):
    entry = artist_relationships.get(artist_name)
    if not entry:
        print(f"Artist '{artist_name}' not found in relationships dict.")
        return
    print(_format_artist_context(artist_name, entry))
