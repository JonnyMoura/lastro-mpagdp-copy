'''
/ai/rag/retrieval.py
-> multi-stage retrieval pipeline over the artist knowledge graph (tokenization,
   text index, community/entity/text matching) used to build LLM context
'''

import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _strip_accents(text):
    """
    Removes diacritics (á->a, ç->c, ...) so matching is accent-insensitive.
    44% of artist names in this corpus contain accented characters, and the
    same free-text entry that produced case-duplicated entities (see
    knowledgeGraph._canonical_entity_key) also produced accent-duplicated
    ones -- 'Acustico' vs 'Acústico' at 6 vs 6,598 occurrences being the
    starkest example. A user typing without accents (common: fast typing,
    non-PT keyboards, mobile autocomplete) would otherwise silently miss an
    exact Stage 1/2 name match and fall through to weaker fallback stages
    with no indication anything went wrong.
    """
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

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
    # Corpus-wide boilerplate: near-universal in this dataset (every entry
    # belongs to "o arquivo" of "musica portuguesa"), so left unfiltered
    # they carry ~zero discriminative signal and can dominate the Stage 3
    # lexical fallback via sheer frequency, making unrelated questions that
    # share only these words collapse onto the same "primary artists" (see
    # rag_diag_v1.md queries 22 vs 24, which returned identical top-5s).
    'arquivo', 'arquivos', 'musica', 'musicas', 'portuguesa', 'portuguesas',
    'portugues', 'portugueses', 'documentado', 'documentados', 'documentada',
    'documentadas', 'gravacao', 'gravacoes', 'video', 'videos',
}

ALL_STOP_WORDS = STOP_WORDS | STOP_WORDS_PT


def _tokenize(text):
    cleaned = re.sub(r'[^\w\s]', '', _strip_accents(text.lower()))
    return {t for t in cleaned.split() if t not in ALL_STOP_WORDS and len(t) > 2}


def _find_name_match(text_lower, name_lower):
    """
    Whole-word/phrase match: a plain substring check would let a short name. Word boundaries prevent that while still matching multi-word
    names normally. An empty name_lower must never match -- \\b\\b is a
    vacuous pattern that matches almost any text (e.g. artists with a blank
    'Nome'/author field), which would otherwise hijack every generic query.
    Both sides are accent-stripped so e.g. a question typed without accents
    still matches an accented artist/entity name (see _strip_accents).
    Returns the re.Match (so callers can inspect .span()) or None.
    """
    if not name_lower:
        return None
    text_norm = _strip_accents(text_lower)
    name_norm = _strip_accents(name_lower)
    return re.search(r'\b' + re.escape(name_norm) + r'\b', text_norm)


def _span_covered(span, claimed_spans):
    """True if span (start, end) falls entirely inside an already-claimed
    span -- used so a shorter name that's merely a substring of an
    already-matched longer name (e.g. a hypothetical artist 'Pires' inside
    an already-matched 'Célio Pires') doesn't fire as a second, spurious
    match on the same words."""
    start, end = span
    return any(start >= cs and end <= ce for cs, ce in claimed_spans)


MAX_VIDEOS_LISTED = 10


def _format_video_links(entry, limit=MAX_VIDEOS_LISTED):
    """
    Renders each video's title + link so the model can point to a specific,
    findable video instead of only naming an artist in the abstract. Links
    otherwise never reach the model at all: artist_relationships aggregates
    every other field across an artist's videos, but 'link' is per-video and
    lives only in entry['videos'] -- nothing upstream of this ever reads it.
    """
    videos = entry.get('videos', [])
    if not videos:
        return []
    lines = ["  Videos (titulo -> link):"]
    for v in videos[:limit]:
        link = v.get('link')
        if not link:
            continue
        title = v.get('theme') or '(sem titulo)'
        lines.append(f"    - {title} -> {link}")
    return lines if len(lines) > 1 else []


def _build_video_link_index(artist_relationships):
    """
    video_id -> (theme, link), built once per request from the videos already
    embedded in artist_relationships. Lets semantic chunk hits (which only
    carry chunk_meta's 'video_id', set at embedding-build time before the
    chunk index tracked links) resolve back to a citable link without
    needing to touch/rebuild the embeddings cache.
    """
    index = {}
    for entry in artist_relationships.values():
        for v in entry.get('videos', []):
            vid = v.get('index')
            link = v.get('link')
            if vid is not None and link:
                index[vid] = (v.get('theme') or '(sem titulo)', link)
    return index


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
        'dates':          'Recording dates',
    }
    for key, label in label_map.items():
        values = entry.get(key, [])
        if values:
            lines.append(f"  {label}: {', '.join(values)}")
    lines.extend(_format_video_links(entry))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Text field index for abstract concept search
# ---------------------------------------------------------------------------

LONG_TEXT_FIELDS = ('biographies', 'history', 'other_info', 'audio_transcription', 'visual_description')
TEXT_FIELDS = LONG_TEXT_FIELDS + ('keywords',)


def build_text_index(artist_relationships):
    """
    Builds an inverted index: token -> {artist_names} across the text fields
    (history, other_info, biographies, keywords, audio_transcription,
    visual_description).
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


def _extract_snippets(text, question_tokens, max_sentences=8):
    """
    Extracts sentences from text that contain at least one query token.
    question_tokens come from _tokenize, which is accent-stripped -- the
    sentence text must be stripped the same way or an accented word in the
    source text (e.g. 'acústico') would never match an unaccented token
    ('acustico'), silently defeating the accent-insensitivity fix.
    """
    sentences = re.split(r'[.!?\n]+', text)
    matching = []
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 10:
            continue
        s_lower = _strip_accents(s.lower())
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
    'audio_transcription': 'Transcricao do video', 'visual_description': 'Descricao visual',
}

# Long, per-video text (can be several joined videos deep) where the search
# step indexes the full text but display would otherwise blind-truncate to
# the first N characters -- reduce these to the sentences that actually
# match the question instead of losing anything past the cutoff.
SNIPPET_FIELDS = {'audio_transcription', 'visual_description'}


def _render_field_text(field, text, question_tokens, limit, semantic_fallback=None):
    """
    Renders a text field for LLM context. SNIPPET_FIELDS (long, per-video
    text that can span multiple joined videos) are reduced to the sentences
    that actually match the question, via _extract_snippets, so relevant
    content isn't lost past a fixed character cut the way a blind prefix
    slice would lose it. Curated per-artist fields (bio/history/notes) keep
    the existing simple prefix slice -- they're short enough that truncation
    rarely cuts anything meaningful.

    semantic_fallback: the highest-scoring embedded chunk for this exact
    (artist, field) pair, if this artist was surfaced via semantic search
    (Stage 3.5). Literal-token snippet extraction is blind to paraphrases by
    construction -- a question with zero literal vocabulary overlap with the
    text (the whole point of semantic search) will find no snippets here
    either, and a blind text[:limit] prefix slice has no reason to contain
    the passage that actually caused the semantic match. Preferring the
    known-relevant chunk avoids silently dropping the evidence that made
    this artist relevant in the first place.
    """
    if field in SNIPPET_FIELDS and len(text) > limit:
        snippets = _extract_snippets(text, question_tokens, max_sentences=8)
        if snippets:
            return ' [...] '.join(snippets)
        if semantic_fallback:
            return semantic_fallback
    return text[:limit]


# ---------------------------------------------------------------------------
# Semantic (embedding) similarity search -- catches paraphrases/synonyms/
# purely thematic links that share no literal vocabulary with the question,
# which the lexical stages above can never find by design.
# ---------------------------------------------------------------------------

SEMANTIC_TOP_K = 30
# Calibrated against the full 24,386-chunk corpus (bge-m3): out-of-domain
# probes (quantum computing, diesel engines, American football, Linux
# install steps -- topics genuinely absent from this archive) consistently
# scored 0.42-0.46 at top rank across 6 queries, establishing the noise
# floor. Genuinely hard paraphrases with zero literal vocabulary overlap
# (e.g. "melancolia de quem deixou a patria" correctly matching a lyric
# about wanting to return home, sharing not one literal word with the
# question) scored 0.49-0.52. DISCOVERY sits just above the noise floor so
# exploratory content stays cheap to surface (low stakes: it only ever
# supplements the "Explorar e Descobrir" section, alongside the existing
# geographic/concept/cultural neighbor blocks). PROMOTE sits well above
# typical paraphrase-level matches, reserving "this IS the direct answer"
# for confident semantic hits -- deliberately conservative, because even
# when nothing promotes, discovery block C3 still fires independently (it's
# computed from the same semantic_hits regardless of promotion), so a
# borderline paraphrase match isn't lost, just not framed as the definitive
# direct answer.
SEMANTIC_PROMOTE_THRESHOLD = 0.55
SEMANTIC_DISCOVERY_THRESHOLD = 0.48


def _semantic_search(question, chunk_vectors, chunk_meta, top_k=SEMANTIC_TOP_K,
                     min_score=SEMANTIC_DISCOVERY_THRESHOLD):
    """
    Embeds the question and returns the top_k most similar chunks (score,
    chunk_meta_dict) scoring at least min_score. chunk_vectors must already
    be L2-normalized (done once at build time in embeddings.save_chunk_index)
    so similarity is a single dot product. Returns [] on any problem --
    missing cache, embed-call failure/timeout -- so a broken/unavailable
    embedding layer degrades the pipeline to its pre-existing lexical-only
    behavior instead of breaking the request.
    """
    if chunk_vectors is None or not chunk_meta:
        return []
    try:
        from ai.rag.embeddings import embed_query
        query_vector = np.asarray(embed_query(question), dtype=np.float32)
    except Exception as e:
        print(f'[RAG] semantic search unavailable (embed call failed): {e}')
        return []

    norm = np.linalg.norm(query_vector)
    if norm == 0:
        return []
    query_vector = query_vector / norm

    scores = chunk_vectors @ query_vector
    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(float(scores[i]), chunk_meta[i]) for i in top_idx if scores[i] >= min_score]


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
    concept_fields = ('keywords', 'categories', 'instruments', 'themes', 'dates')
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
        for field in LONG_TEXT_FIELDS:
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
            texts.get(f, '') for f in LONG_TEXT_FIELDS
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


def _is_non_music_category(category_name):
    return not _strip_accents(category_name.strip().lower()).startswith('musica')


def _find_disruptive_neighbors(primary_artists, artist_relationships, artist_texts,
                               question_tokens, max_results=6):
    """
    Deliberately seeks OTHER-DOMAIN content (gastronomy, crafts, religion,
    poetry, oral history, soundscapes...) connected to the primary artists.
    The corpus's own category distribution is heavily music-dominated (three
    "Musica*" categories alone cover roughly two-thirds of all ~8,400
    projects; Gastronomia/Artesanato/Religiao are each under 2% of the
    total), so an unweighted search keeps resurfacing more music by sheer
    weight of numbers -- the non-music category filter exists to
    counteract that and guarantee this material is considered on every
    answer, not just when the question already touches one.

    Deliberately NOT geography-only: a connection here can be geographic
    (same region), textual/lyrical (shared vocabulary in transcripts,
    history, biographies -- a phrase, an image, a reference that recurs),
    or conceptual (shared keyword/instrument/theme/recording date). Any one
    of these alone qualifies a candidate; they combine into the ranking
    score. Geography-only would miss exactly the kind of connection this
    archive is richest in -- two artists on opposite ends of the country
    whose transcripts share nothing but a phrase, a saint's name, or a
    recipe.
    """
    primary_set = set(primary_artists)

    geo_values = set()
    primary_text_tokens = set()
    primary_concepts = defaultdict(set)
    concept_fields = ('keywords', 'categories', 'instruments', 'themes', 'dates')
    for a in primary_artists:
        entry = artist_relationships.get(a, {})
        for field in ('locations', 'municipalities', 'districts', 'regions'):
            geo_values.update(entry.get(field, []))
        for field in concept_fields:
            for val in entry.get(field, []):
                primary_concepts[val.lower()].add(field)
        texts = artist_texts.get(a, {})
        for field in LONG_TEXT_FIELDS:
            text = texts.get(field, '')
            if text:
                primary_text_tokens.update(_tokenize(text))

    geo_tokens = set()
    for v in geo_values:
        geo_tokens.update(_tokenize(v))

    search_tokens = (primary_text_tokens | question_tokens) - ALL_STOP_WORDS

    candidates = []
    for artist_name, texts in artist_texts.items():
        if artist_name in primary_set:
            continue

        entry = artist_relationships.get(artist_name, {})
        non_music_cats = [c for c in entry.get('categories', []) if _is_non_music_category(c)]
        if not non_music_cats:
            continue

        artist_geo = set()
        for field in ('locations', 'municipalities', 'districts', 'regions'):
            for name in entry.get(field, []):
                artist_geo.update(_tokenize(name))
        geo_overlap = len(geo_tokens & artist_geo)

        combined_text = ' '.join(texts.get(f, '') for f in LONG_TEXT_FIELDS if texts.get(f, ''))
        text_overlap = len(search_tokens & _tokenize(combined_text)) if combined_text else 0

        concept_overlap = sum(
            1 for field in concept_fields
            for val in entry.get(field, [])
            if val.lower() in primary_concepts
        )

        if geo_overlap == 0 and text_overlap < 2 and concept_overlap == 0:
            continue

        score = geo_overlap + text_overlap + concept_overlap * 1.5
        candidates.append((artist_name, texts, non_music_cats, score))

    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates[:max_results]


# ---------------------------------------------------------------------------
# Exploration retrieval pipeline
# ---------------------------------------------------------------------------

ENTITY_BLOCKLIST = {
    'tar', 'ver', 'dar', 'ter', 'voz',
    # Tagged on 41 artists (likely a Theme/Keyword value, not Category/
    # Instrument) yet also a corpus-wide generic word -- every entry
    # belongs to "o arquivo", so any question mentioning it (almost all of
    # them do) matches all 41. Worse than a silent false positive: Stage 2
    # ranks matched entities most-shared-first, so 'arquivo' (41 artists)
    # outranks and completely crowds out a genuinely specific same-question
    # match with only 1 artist (e.g. 'ranchos folcloricos', 'opera') --
    # see rag_diag_v2.md queries 22 vs 24, which returned identical
    # top-5s despite sharing no real topic.
    'arquivo',
}

# Keyword is large (13,776 distinct free-text values) and noisy -- among
# just its 212 four-character entries are 'Casa', 'Amor', 'Flor', 'Real',
# 'Agua', 'Rosa', 'Vale', 'Alto', 'Auto', 'Ouro' and a pile of first names,
# any of which would hijack unrelated questions if the length>=5 floor were
# simply lowered. But real, specific, low-collision genre/form terms are
# mixed into that same short tail -- hand-picked here rather than admitted
# by a blanket rule, the same way ENTITY_BLOCKLIST hand-picks exclusions
# rather than applying a blanket one.
SHORT_KEYWORD_ALLOWLIST = {
    'fado', 'jazz', 'funk', 'punk', 'rock', 'folk', 'soul', 'reel', 'scat', 'hino',
}


def _is_specific_entity(name_lower, curated_names=None):
    """
    Filter out generic single-word entity matches. The length>=5 floor
    exists because Keyword is large and noisy -- short entries there are
    disproportionately generic words that would hijack unrelated questions
    (see SHORT_KEYWORD_ALLOWLIST for concrete examples). Category and
    Instrument are small, curated vocabularies (79 and 554 canonical values
    after entity normalization) where that risk barely exists, so a name
    found in curated_names bypasses the length floor entirely. A handful of
    short Keyword values that are genuinely specific genre/form terms (e.g.
    'fado') are allowed through the same way via SHORT_KEYWORD_ALLOWLIST.
    ENTITY_BLOCKLIST is still checked first and wins regardless: 'voz' is
    both a real instrument name and an extremely common ordinary word, so
    it stays excluded even though it would otherwise qualify.
    """
    if name_lower in ENTITY_BLOCKLIST:
        return False
    if name_lower in SHORT_KEYWORD_ALLOWLIST:
        return True
    if curated_names and name_lower in curated_names:
        return True
    words = name_lower.split()
    if len(words) >= 2:
        return True
    return len(name_lower) >= 5


# Middle of the three hierarchy levels (0.3 coarse / 0.8 mid / 1.5 fine) --
# broad enough that a community groups artists connected across several
# entity types at once (not just one shared dimension the way geo/concept/
# cultural neighbors are), tight enough not to degenerate into "everyone".
COMMUNITY_DISCOVERY_RESOLUTION = 0.8


def _artist_community(artist_name, communities):
    """Finds the community dict containing artist_name, or None."""
    for community in communities:
        if artist_name in community['artists']:
            return community
    return None


def retrieve_exploration_context(
    question,
    artist_relationships,
    name_to_artists,
    community_hierarchy,
    text_token_index,
    artist_texts,
    chunk_vectors=None,
    chunk_meta=None,
    max_direct=5,
    max_discoveries=8,
):
    """
    Multi-stage exploration retrieval:

    Stage 1   -- Artist name match (precise)
    Stage 2   -- Entity name match (instruments, regions, categories)
    Stage 3   -- Text field search (biographies, history, other_info)
    Stage 3.5 -- Semantic similarity search (embeddings) -- only when 1-3
                 found nothing; catches paraphrases/synonyms with zero
                 literal token overlap, which no lexical stage can find
    Stage 4   -- Community match
    Stage 5   -- Keyword fallback

    Discovery context is built from:
    A) Primary artists' full text (bio, history, notes)
    B) Geographic neighbors' text — other entries from the same
       region/district/location, retrieved via the graph, with raw
       content passed to the LLM to find cross-domain connections
    C) Text search results as fallback

    Returns dict with keys: direct, discoveries, community_summary
    """
    question_lower = question.lower()
    question_tokens = _tokenize(question)
    artist_names_lower = {n.lower(): n for n in artist_relationships}
    video_link_index = _build_video_link_index(artist_relationships)

    direct_artists = []
    entity_artists = []

    # --- Stage 1: Artist name match ---
    # Collects every artist name mentioned, not just the first: a "compare
    # X and Y" question names two artists, and stopping at the first match
    # silently dropped the second. Longest names are checked first (already
    # sorted that way) so a longer match claims its span in the question
    # before a shorter, coincidentally-contained name can also fire on the
    # same words (e.g. a hypothetical standalone artist "Pires" firing a
    # second time on the "Pires" inside an already-matched "Célio Pires" --
    # see _span_covered).
    sorted_artist_names = sorted(artist_names_lower.keys(), key=len, reverse=True)
    claimed_spans = []
    for name_lower in sorted_artist_names:
        match = _find_name_match(question_lower, name_lower)
        if not match or _span_covered(match.span(), claimed_spans):
            continue
        claimed_spans.append(match.span())
        direct_artists.append(artist_names_lower[name_lower])
        if len(direct_artists) >= max_direct:
            break

    # --- Stage 2: Entity name match (excluding artist names) ---
    # Two passes, not one. Pass 1 collects EVERY non-overlapping entity
    # match in the question (not just the first found) -- a question can
    # name two instruments/categories/etc (e.g. "gaita de foles e
    # sanfona"), and stopping at the first match would silently drop the
    # second the same way it did for artists before the Stage 1 fix.
    # Pass 2 then allocates the max_direct budget across whatever matched,
    # most-shared entity first: the more artists reference a term, the more
    # likely it's an established, deliberately-meant one rather than a
    # coincidental collision with ordinary descriptive language in the
    # question. Without this second pass, whichever entity happened to be
    # found first in the length-sorted scan could fill the whole budget
    # regardless of relevance -- e.g. "registos" (Portuguese for "records",
    # coincidentally also a rare Keyword tag on just 12 artists) matching
    # before "fado" (388 artists, clearly the actual point) in "Existem
    # registos de fado neste arquivo?" and starving fado out entirely.
    if not direct_artists:
        curated_names = set()
        for entry in artist_relationships.values():
            for c in entry.get('categories', []):
                curated_names.add(c.lower())
            for i in entry.get('instruments', []):
                curated_names.add(i.lower())

        sorted_entity_names = sorted(name_to_artists.keys(), key=len, reverse=True)
        claimed_spans = []
        matched_entities = []  # [(name_lower, {artist, ...}), ...]
        for name_lower in sorted_entity_names:
            if not _is_specific_entity(name_lower, curated_names) or name_lower in artist_names_lower:
                continue
            match = _find_name_match(question_lower, name_lower)
            if not match or _span_covered(match.span(), claimed_spans):
                continue
            claimed_spans.append(match.span())
            candidates = {a for a in name_to_artists[name_lower] if a in artist_relationships}
            if candidates:
                matched_entities.append((name_lower, candidates))

        matched_entities.sort(key=lambda pair: -len(pair[1]))

        seen = set()
        for name_lower, candidates in matched_entities:
            for a in sorted(candidates):
                if a not in seen:
                    entity_artists.append(a)
                    seen.add(a)
                    if len(entity_artists) >= max_direct:
                        break
            if len(entity_artists) >= max_direct:
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

    # --- Stage 3.5: Semantic similarity search ---
    # Computed once regardless of whether a primary artist was already
    # found -- the result feeds both the entry-point fallback below (only
    # used if still empty) and the semantic discovery block further down
    # (used unconditionally, so lexically-precise questions also benefit
    # from semantic serendipity, not just questions with no literal match).
    semantic_hits = _semantic_search(question, chunk_vectors, chunk_meta)

    # Highest-scoring chunk per (artist, field) -- semantic_hits is already
    # sorted by score descending, so the first occurrence per key wins. Feeds
    # _render_field_text's semantic_fallback so an artist surfaced by
    # semantic search doesn't lose the exact passage that justified it to a
    # blind prefix truncation further down.
    semantic_best_by_artist_field = {}
    for _, m in semantic_hits:
        key = (m['artist_name'], m['field'])
        if key not in semantic_best_by_artist_field:
            semantic_best_by_artist_field[key] = m['text']

    if not primary_artists:
        strong_hits = [(s, m) for s, m in semantic_hits if s >= SEMANTIC_PROMOTE_THRESHOLD]
        seen_semantic_artists = set()
        promoted_semantic = []
        for _, m in strong_hits:
            name = m['artist_name']
            if name in seen_semantic_artists or name not in artist_relationships:
                continue
            seen_semantic_artists.add(name)
            promoted_semantic.append(name)
            if len(promoted_semantic) >= max_direct:
                break
        if promoted_semantic:
            primary_artists = promoted_semantic
            primary_set = set(primary_artists)

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
        lines.extend(_format_video_links(artist_relationships.get(a, {}), limit=3))
        for field in LONG_TEXT_FIELDS:
            text = artist_texts[a].get(field, '')
            if text:
                label = FIELD_LABELS.get(field, field)
                fallback = semantic_best_by_artist_field.get((a, field))
                lines.append(f"    {label}: {_render_field_text(field, text, question_tokens, 2200, semantic_fallback=fallback)}")
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
        lines.extend(_format_video_links(entry, limit=2))
        for field in LONG_TEXT_FIELDS:
            text = texts.get(field, '')
            if text:
                label = FIELD_LABELS.get(field, field)
                fallback = semantic_best_by_artist_field.get((artist_name, field))
                lines.append(f"    {label}: {_render_field_text(field, text, question_tokens, 1600, semantic_fallback=fallback)}")
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
        lines.extend(_format_video_links(entry, limit=2))
        for field in LONG_TEXT_FIELDS:
            text = texts.get(field, '')
            if text:
                label = FIELD_LABELS.get(field, field)
                fallback = semantic_best_by_artist_field.get((artist_name, field))
                lines.append(f"    {label}: {_render_field_text(field, text, question_tokens, 1600, semantic_fallback=fallback)}")
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
        lines.extend(_format_video_links(entry, limit=2))
        for field in LONG_TEXT_FIELDS:
            text = texts.get(field, '')
            if text:
                label = FIELD_LABELS.get(field, field)
                fallback = semantic_best_by_artist_field.get((artist_name, field))
                lines.append(f"    {label}: {_render_field_text(field, text, question_tokens, 1600, semantic_fallback=fallback)}")
        if len(lines) > 1:
            discovery_blocks.append('\n'.join(lines))

    # C3) Semantic neighbors — chunks found by embedding similarity rather
    #     than shared tokens/entities, so genuinely non-literal/paraphrased
    #     connections can surface. Chunk text is already a properly-bounded,
    #     sentence-complete excerpt from chunking.chunk_text -- render it
    #     directly rather than through _render_field_text/_extract_snippets,
    #     which select on literal question tokens a semantic-only hit is,
    #     by definition, likely not to contain.
    seen_semantic = set()
    for _, m in semantic_hits:
        artist_name = m['artist_name']
        if artist_name in covered or artist_name in seen_semantic:
            continue
        seen_semantic.add(artist_name)
        covered.add(artist_name)
        label = FIELD_LABELS.get(m['field'], m['field'])
        theme = f" -- {m['theme']}" if m.get('theme') else ''
        video_hit = video_link_index.get(m.get('video_id'))
        link_line = f"\n    Video: {video_hit[0]} -> {video_hit[1]}" if video_hit else ''
        discovery_blocks.append(
            f"  [{artist_name}] (correspondencia semantica)\n"
            f"    {label}{theme}: {m['text'][:1600]}{link_line}"
        )
        if len(seen_semantic) >= max_discoveries:
            break

    # F) Disruptive/cross-domain neighbors — deliberately surfaces non-music
    #    categories (gastronomy, crafts, religion, poetry, oral history,
    #    soundscapes...) from the same geography as the primary artists,
    #    regardless of the question's own vocabulary. See
    #    _find_disruptive_neighbors for why this needs to be explicit rather
    #    than left to the other (vocabulary-driven) neighbor searches.
    #    Runs BEFORE the community block (E) on purpose: this block (and B/C/
    #    C2/C3 above it) always attaches real text + video links, while E
    #    falls back to a bare name list for whatever's left over. Running E
    #    first would let it "claim" artists into `covered` before the richer
    #    blocks get a chance, silently downgrading what could've been a
    #    citable quote+link into just a name in a list.
    disruptive_neighbors = _find_disruptive_neighbors(
        primary_artists, artist_relationships, artist_texts,
        question_tokens, max_results=max_discoveries,
    )
    for artist_name, texts, non_music_cats, _ in disruptive_neighbors:
        if artist_name in covered:
            continue
        covered.add(artist_name)
        entry = artist_relationships.get(artist_name, {})
        lines = [f"  [{artist_name}] (categoria fora da musica: {', '.join(non_music_cats[:3])})"]
        lines.extend(_format_video_links(entry, limit=2))
        for field in LONG_TEXT_FIELDS:
            text = texts.get(field, '')
            if text:
                label = FIELD_LABELS.get(field, field)
                fallback = semantic_best_by_artist_field.get((artist_name, field))
                lines.append(f"    {label}: {_render_field_text(field, text, question_tokens, 1600, semantic_fallback=fallback)}")
        if len(lines) > 1:
            discovery_blocks.append('\n'.join(lines))

    # E) Community neighbors — artists Leiden-clustered with a primary
    #    artist across many shared entity types at once (category,
    #    instrument, theme, keyword, location, date -- not just one
    #    dimension the way geo/concept/cultural neighbors above are).
    #    Surfaced for every primary artist, not gated behind Stage 4 --
    #    otherwise the pre-computed community hierarchy is only ever
    #    consulted on the rare question that stages 1-3.5 all miss, which
    #    almost never happens once any text field shares a single token.
    if community_hierarchy:
        communities_at_res = community_hierarchy.get(COMMUNITY_DISCOVERY_RESOLUTION, [])
        seen_community_ids = set()
        for a in primary_artists:
            community = _artist_community(a, communities_at_res)
            if community is None or community['id'] in seen_community_ids:
                continue
            seen_community_ids.add(community['id'])

            members = [m for m in community['artists'] if m not in covered][:max_discoveries]
            if not members:
                continue
            covered.update(members)

            # Spotlight up to 2 members with real, citable content (text +
            # video links). A bare name has nothing the LLM can quote or
            # link to, which is why community mentions in the answer tend
            # to stay generic ("also X, Y, Z") instead of specific --
            # give it something concrete to grab for at least a couple.
            spotlight_count = 0
            for member in members:
                if spotlight_count >= 2:
                    break
                texts = artist_texts.get(member)
                if not texts:
                    continue
                entry = artist_relationships.get(member, {})
                lines = [f"  [{member}] (mesma comunidade de {a})"]
                for f, label in [('categories', 'Categorias'), ('instruments', 'Instrumentos')]:
                    vals = entry.get(f, [])
                    if vals:
                        lines.append(f"    {label}: {', '.join(vals[:5])}")
                lines.extend(_format_video_links(entry, limit=2))
                for field in LONG_TEXT_FIELDS:
                    text = texts.get(field, '')
                    if text:
                        label = FIELD_LABELS.get(field, field)
                        fallback = semantic_best_by_artist_field.get((member, field))
                        lines.append(f"    {label}: {_render_field_text(field, text, question_tokens, 1200, semantic_fallback=fallback)}")
                if len(lines) > 1:
                    discovery_blocks.append('\n'.join(lines))
                    spotlight_count += 1

            traits = []
            for t, label in [('Category', 'Categorias'), ('Instrument', 'Instrumentos'), ('Region', 'Regioes')]:
                counter = community.get('entity_profile', {}).get(t)
                if counter:
                    traits.append(f"{label}: {', '.join(n for n, _ in counter.most_common(5))}")

            lines = [f"  Comunidade de {a} ({len(community['artists'])} artistas; {'; '.join(traits)})"]
            lines.append(f"    Outros artistas desta comunidade: {', '.join(members)}")
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

    return {
        'direct': direct_blocks,
        'discoveries': discovery_blocks,
        'community_summary': community_summary,
    }


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


def print_artist_entry(artist_name, artist_relationships):
    entry = artist_relationships.get(artist_name)
    if not entry:
        print(f"Artist '{artist_name}' not found in relationships dict.")
        return
    print(_format_artist_context(artist_name, entry))
