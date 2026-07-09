'''
/ai/rag/knowledgeGraph.py
-> builds the artist knowledge graph and Leiden community hierarchy from project rows
'''

import re
import sys
import networkx as nx
from collections import Counter, defaultdict
import math
import igraph as ig
import leidenalg


def create_knowledge_graph_from_data(data_rows):
    """
    Creates a weighted knowledge graph using unambiguous (name, type) tuples as node identifiers.
    This prevents conflicts where the same name exists for different types (e.g., an artist and a theme).

    Each row is also its own Video node, keyed by a unique index (the row's DB
    id when available, otherwise its position in data_rows). This keeps two
    rows distinct even when Nome and Tema are identical, or Tema is missing,
    so per-row info never overwrites another row's.

    A theme is effectively unique per artist, so when a row has one, it takes
    the artist's usual structural slot and the video hangs off of it
    (Artist -> Theme -> Video). When a row has no theme, its video node fills
    that slot itself and edges directly to the artist (Artist -> Video).

    Everything else collected per row (instruments, categories, keywords,
    location, history, other_info, biographies) can vary between videos from
    the same artist, so it's attached to that row's video node rather than
    the shared artist node.
    """
    G = nx.Graph()

    # Count node frequencies for weighting
    node_counts = Counter()
    for row in data_rows:
        if row.get('Tema'): node_counts[row.get('Tema')] += 1
        if row.get('Instrumentos'):
            for inst in row['Instrumentos'].split(','):
                node_counts[inst.strip()] += 1
        if row.get('Categoria'):
            for cat in row['Categoria'].split(','):
                node_counts[cat.strip()] += 1
        if row.get('Local'): node_counts[row.get('Local')] += 1
        if row.get('keywords'):
            for keyword in row['keywords'].split(','):
                node_counts[keyword.strip()] += 1

    total_rows = len(data_rows)

    def get_weight(name):
        return math.log(total_rows / (node_counts.get(name, 0) + 1)) + 1

    for row_index, row in enumerate(data_rows):
        artist_name = row.get('Nome')

        artist_node = (artist_name, 'Artist')
        if not G.has_node(artist_node):
            G.add_node(artist_node, type='Artist', name=artist_name)

        video_index = row['id'] if row.get('id') is not None else row_index
        video_node = (video_index, 'Video')
        G.add_node(
            video_node,
            type='Video',
            index=video_index,
            name=row.get('Tema') or artist_name,
            theme=row.get('Tema'),
            link=row.get('link'),
            history=row.get('history', ''),
            other_info=row.get('other_info', ''),
            biographies=row.get('biographies', ''),
        )

        # A theme is effectively unique per artist, so when present it takes
        # the artist's usual structural slot and the video hangs off of it
        # (Artist -> Theme -> Video); multiple videos can share a Theme this
        # way. When a row has no theme, the video's own id fills that slot
        # instead and edges directly to the artist (Artist -> Video). Either
        # way, the rest of the row's info always edges to the video itself.
        theme_name = row.get('Tema')
        if theme_name:
            theme_node = (theme_name, 'Theme')
            if not G.has_node(theme_node): G.add_node(theme_node, type='Theme', name=theme_name)
            G.add_edge(artist_node, theme_node, relationship='has_theme', weight=get_weight(theme_name))
            G.add_edge(theme_node, video_node, relationship='documents', weight=1.0)
        else:
            G.add_edge(artist_node, video_node, relationship='has_video', weight=1.0)

        if row.get('Instrumentos'):
            for name in [i.strip() for i in row['Instrumentos'].split(',') if i.strip()]:
                node = (name, 'Instrument')
                if not G.has_node(node): G.add_node(node, type='Instrument', name=name)
                G.add_edge(video_node, node, relationship='plays', weight=get_weight(name))

        if row.get('Categoria'):
            for name in [c.strip() for c in row['Categoria'].split(',') if c.strip()]:
                node = (name, 'Category')
                if not G.has_node(node): G.add_node(node, type='Category', name=name)
                G.add_edge(video_node, node, relationship='in_category', weight=get_weight(name))

        if row.get('keywords'):
            for name in [k.strip() for k in row['keywords'].split(',') if k.strip()]:
                node = (name, 'Keyword')
                if not G.has_node(node): G.add_node(node, type='Keyword', name=name)
                G.add_edge(video_node, node, relationship='has_keyword', weight=get_weight(name))

        local_name    = row.get('Local')
        concelho_name = row.get('Concelho')
        distrito_name = row.get('Distrito/Ilha')
        regiao_name   = row.get('Região')

        regiao_node = concelho_node = distrito_node = local_node = None

        if regiao_name:
            regiao_node = (regiao_name, 'Region')
            if not G.has_node(regiao_node):
                G.add_node(regiao_node, type='Region', name=regiao_name)

        if distrito_name:
            distrito_node = (distrito_name, regiao_name, 'District')
            if not G.has_node(distrito_node):
                G.add_node(distrito_node, type='District', name=distrito_name)

        if concelho_name:
            concelho_node = (concelho_name, distrito_name, 'Municipality')
            if not G.has_node(concelho_node):
                G.add_node(concelho_node, type='Municipality', name=concelho_name)

        if local_name:
            local_node = (local_name, concelho_name, 'Location')
            if not G.has_node(local_node):
                G.add_node(local_node, type='Location', name=local_name)
            G.add_edge(video_node, local_node, relationship='located_in', weight=get_weight(local_name))

        if local_node and concelho_node:
            G.add_edge(local_node, concelho_node, relationship='part_of', weight=1.0)
        if concelho_node and distrito_node:
            G.add_edge(concelho_node, distrito_node, relationship='belongs_to', weight=1.0)
        if distrito_node and regiao_node:
            G.add_edge(distrito_node, regiao_node, relationship='belongs_to', weight=1.0)

    return G


# ---------------------------------------------------------------------------
# Shared artist -> video -> entity traversal
#
# Instruments/categories/locations/keywords now hang off each row's Video
# node instead of the Artist node, and a themed video is only reachable via
# its Theme node (Artist -> Theme -> Video), while a themeless video edges
# directly to the Artist (Artist -> Video). These two helpers centralize that
# walk so every consumer (relationships dict, projection, communities) finds
# the same videos/entities for a given artist.
# ---------------------------------------------------------------------------

def _iter_artist_videos(graph, artist_node):
    """Returns every Video node belonging to an artist, themed or not."""
    videos = []
    for neighbor in graph.neighbors(artist_node):
        ntype = graph.nodes[neighbor].get('type', '')
        if ntype == 'Video':
            videos.append(neighbor)
        elif ntype == 'Theme':
            for hop in graph.neighbors(neighbor):
                if graph.nodes[hop].get('type') == 'Video':
                    videos.append(hop)
    return videos


def _iter_video_entities(graph, video_node):
    """
    Yields (entity_type, entity_node) for every entity attached to a video:
    Instrument, Category, Keyword, Location, and the geographic hierarchy
    (Municipality/District/Region) reached by walking up from Location.
    """
    for neighbor in graph.neighbors(video_node):
        ntype = graph.nodes[neighbor].get('type', '')
        if ntype in ('Theme', 'Artist', 'Video') or not ntype:
            continue

        yield ntype, neighbor

        if ntype == 'Location':
            current = neighbor
            for expected_type in ('Municipality', 'District', 'Region'):
                for hop in graph.neighbors(current):
                    hop_type = graph.nodes[hop].get('type', '')
                    if hop_type == expected_type:
                        yield hop_type, hop
                        current = hop
                        break


def build_artist_subgraph(graph, artist_name):
    """
    Extracts one artist's slice of the graph: their node, every Theme/Video
    they're connected to, and every entity (instrument/category/keyword/
    location + geographic hierarchy) attached to those videos. Meant for
    visually inspecting how one artist's rows are wired into the graph.
    """
    artist_node = (artist_name, 'Artist')
    if artist_node not in graph:
        return nx.Graph()

    nodes = {artist_node}
    for neighbor in graph.neighbors(artist_node):
        if graph.nodes[neighbor].get('type') in ('Theme', 'Video'):
            nodes.add(neighbor)

    for video_node in _iter_artist_videos(graph, artist_node):
        nodes.add(video_node)
        for _, entity_node in _iter_video_entities(graph, video_node):
            nodes.add(entity_node)

    return graph.subgraph(nodes).copy()


def build_relationships_dict(graph):
    """
    Converts the graph into a flat dictionary keyed by artist name.
    Each entry aggregates entities across all of the artist's videos, and
    also lists each video individually so info can be added or updated for
    one specific video without touching the others.

    Structure:
    {
        "ArtistName": {
            "themes":       ["T1", "T2"],
            "instruments":  ["I1"],
            "categories":   ["C1", "C2"],
            "locations":    ["L1"],
            "municipalities": ["M1"],
            "districts":    ["D1"],
            "regions":      ["R1"],
            "keywords":     ["K1"],
            "history":      "...",       # joined across videos
            "other_info":   "...",
            "biographies":  "...",
            "videos": [
                {
                    "index": 42, "theme": "...", "link": "...",
                    "instruments": [...], "categories": [...], "keywords": [...],
                    "locations": [...], "municipalities": [...], "districts": [...], "regions": [...],
                    "history": "...", "other_info": "...", "biographies": "...",
                },
                ...
            ],
        },
        ...
    }

    A reverse index is also returned so the retrieval layer can quickly find
    all artists connected to any given name:
    {
        "some_name": ["ArtistA", "ArtistB"],
        ...
    }
    """
    # entity type → plural key used in the artist/video dicts
    TYPE_KEY = {
        'Instrument':   'instruments',
        'Category':     'categories',
        'Location':     'locations',
        'Municipality': 'municipalities',
        'District':     'districts',
        'Region':       'regions',
        'Keyword':      'keywords',
    }

    artist_relationships = {}   # artist_name  -> {type_key: [names]}
    name_to_artists     = defaultdict(set)  # any_name -> {artist_names}

    for node, data in graph.nodes(data=True):
        if data.get('type') != 'Artist':
            continue

        artist_name = data.get('name', '')
        entry = {key: [] for key in TYPE_KEY.values()}
        entry['themes'] = []
        entry['videos'] = []

        for neighbor in graph.neighbors(node):
            if graph.nodes[neighbor].get('type') == 'Theme':
                theme_name = graph.nodes[neighbor].get('name', '')
                if theme_name:
                    entry['themes'].append(theme_name)
                    name_to_artists[theme_name.lower()].add(artist_name)

        history_parts, other_info_parts, biographies_parts = [], [], []

        for video_node in _iter_artist_videos(graph, node):
            vdata = graph.nodes[video_node]
            video_entry = {key: [] for key in TYPE_KEY.values()}
            video_entry['index']       = vdata.get('index')
            video_entry['theme']       = vdata.get('theme')
            video_entry['link']        = vdata.get('link')
            video_entry['history']     = vdata.get('history', '')
            video_entry['other_info']  = vdata.get('other_info', '')
            video_entry['biographies'] = vdata.get('biographies', '')

            if vdata.get('history'):     history_parts.append(vdata['history'])
            if vdata.get('other_info'):  other_info_parts.append(vdata['other_info'])
            if vdata.get('biographies'): biographies_parts.append(vdata['biographies'])

            for entity_type, entity_node in _iter_video_entities(graph, video_node):
                key = TYPE_KEY.get(entity_type)
                entity_name = graph.nodes[entity_node].get('name', '')
                if key and entity_name:
                    entry[key].append(entity_name)
                    video_entry[key].append(entity_name)
                    name_to_artists[entity_name.lower()].add(artist_name)

            entry['videos'].append(video_entry)

        # Deduplicate list-based fields while preserving order
        for key in list(TYPE_KEY.values()) + ['themes']:
            entry[key] = list(dict.fromkeys(entry[key]))

        if history_parts:     entry['history']     = '\n\n'.join(dict.fromkeys(history_parts))
        if other_info_parts:  entry['other_info']  = '\n\n'.join(dict.fromkeys(other_info_parts))
        if biographies_parts: entry['biographies'] = '\n\n'.join(dict.fromkeys(biographies_parts))

        entry['videos'].sort(key=lambda v: (v['index'] is None, v['index']))

        artist_relationships[artist_name] = entry
        name_to_artists[artist_name.lower()].add(artist_name)

    return artist_relationships, dict(name_to_artists)


def build_artist_projection(graph, entity_types=None, max_entity_share=0.10):
    """
    Projects the heterogeneous knowledge graph onto an artist-artist similarity
    graph. Two artists share an edge if they have common entities, weighted by
    the inverse of how many artists share each entity (rarity bonus).


    """
    if entity_types is None:
        entity_types = {'Theme', 'Instrument', 'Category', 'Keyword', 'Location'}

    entity_to_artists = defaultdict(set)
    artist_nodes = set()
    for node, data in graph.nodes(data=True):
        if data.get('type') != 'Artist':
            continue
        artist_nodes.add(node)

        if 'Theme' in entity_types:
            for neighbor in graph.neighbors(node):
                if graph.nodes[neighbor].get('type') == 'Theme':
                    entity_to_artists[neighbor].add(node)

        for video_node in _iter_artist_videos(graph, node):
            for entity_type, entity_node in _iter_video_entities(graph, video_node):
                if entity_type in entity_types:
                    entity_to_artists[entity_node].add(node)

    n_artists = len(artist_nodes)
    threshold = int(n_artists * max_entity_share)

    proj = nx.Graph()
    for node in artist_nodes:
        proj.add_node(node, **graph.nodes[node])

    for _, artists in entity_to_artists.items():
        if len(artists) > threshold:
            continue
        rarity = 1.0 / len(artists)
        artists_list = list(artists)
        for i in range(len(artists_list)):
            for j in range(i + 1, len(artists_list)):
                a, b = artists_list[i], artists_list[j]
                if proj.has_edge(a, b):
                    proj[a][b]['weight'] += rarity
                else:
                    proj.add_edge(a, b, weight=rarity)

    return proj


def _run_leiden(graph, resolution=1.0):
    """Runs Leiden on a NetworkX graph and returns the igraph partition + node list."""
    nx_nodes = list(graph.nodes())
    node_index = {n: i for i, n in enumerate(nx_nodes)}

    ig_edges = [(node_index[u], node_index[v]) for u, v in graph.edges()]
    ig_weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]

    ig_graph = ig.Graph(n=len(nx_nodes), edges=ig_edges)
    ig_graph.es['weight'] = ig_weights

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=42,
    )
    return partition, nx_nodes


def _build_community_dicts(partition, nx_nodes, artist_proj, full_graph):
    """
    Converts a Leiden partition into community dicts enriched with entity
    information from the full heterogeneous graph.
    """
    communities = []
    for cid, member_ids in enumerate(partition):
        artists = []
        for mid in member_ids:
            node = nx_nodes[mid]
            data = artist_proj.nodes[node]
            name = data.get('name', '')
            if name:
                artists.append(name)

        if not artists:
            continue

        # Gather entities from the full graph for this community's artists
        entity_buckets = defaultdict(Counter)
        for artist_name in artists:
            artist_node = (artist_name, 'Artist')
            if artist_node not in full_graph:
                continue

            for neighbor in full_graph.neighbors(artist_node):
                if full_graph.nodes[neighbor].get('type') == 'Theme':
                    theme_name = full_graph.nodes[neighbor].get('name', '')
                    if theme_name:
                        entity_buckets['Theme'][theme_name] += 1

            for video_node in _iter_artist_videos(full_graph, artist_node):
                for entity_type, entity_node in _iter_video_entities(full_graph, video_node):
                    entity_name = full_graph.nodes[entity_node].get('name', '')
                    if entity_name:
                        entity_buckets[entity_type][entity_name] += 1

        # Build token bag for retrieval matching
        tokens = set()
        for artist_name in artists:
            for tok in re.sub(r'[^\w\s]', '', artist_name.lower()).split():
                if len(tok) > 2:
                    tokens.add(tok)
        for counter in entity_buckets.values():
            for name in counter:
                for tok in re.sub(r'[^\w\s]', '', name.lower()).split():
                    if len(tok) > 2:
                        tokens.add(tok)

        # Build a summary showing the defining traits of this community
        summary_lines = [f"Community {cid} ({len(artists)} artists):"]
        summary_lines.append(f"  Artists: {', '.join(sorted(artists))}")

        type_order = ['Category', 'Instrument', 'Theme', 'Keyword',
                      'Location', 'Municipality', 'District', 'Region']
        for t in type_order:
            if t in entity_buckets:
                top_names = [n for n, _ in entity_buckets[t].most_common(8)]
                if top_names:
                    summary_lines.append(f"  {t}(s): {', '.join(top_names)}")

        communities.append({
            'id':             cid,
            'artists':        artists,
            'tokens':         tokens,
            'entity_profile': dict(entity_buckets),
            'summary':        '\n'.join(summary_lines),
        })

    return communities


def build_leiden_communities(artist_proj, full_graph, resolution=1.0):
    """
    Runs Leiden on the artist projection graph and returns community dicts
    enriched with entity information from the full knowledge graph.

    Each community dict contains:
        id, artists, tokens, entity_profile, summary
    """
    if artist_proj.number_of_edges() == 0:
        return []

    partition, nx_nodes = _run_leiden(artist_proj, resolution)
    return _build_community_dicts(partition, nx_nodes, artist_proj, full_graph)


def build_hierarchical_communities(artist_proj, full_graph, resolutions=None):
    """
    Runs Leiden at multiple resolutions to produce a hierarchy of communities.
    Returns: dict[float, list[dict]]  — resolution → community list

    Coarse (low res)  → broad clusters (e.g. regional traditions)
    Fine   (high res) → tight clusters (e.g. specific instrument groups)
    """
    if resolutions is None:
        resolutions = [0.3, 0.8, 1.5]

    if artist_proj.number_of_edges() == 0:
        return {r: [] for r in resolutions}

    hierarchy = {}
    for res in resolutions:
        partition, nx_nodes = _run_leiden(artist_proj, res)
        hierarchy[res] = _build_community_dicts(
            partition, nx_nodes, artist_proj, full_graph
        )
    return hierarchy


def visualize_artist_subgraph(graph, artist_name):
    """Draws one artist's slice of the graph (see build_artist_subgraph)."""
    import matplotlib.pyplot as plt

    sub = build_artist_subgraph(graph, artist_name)
    if sub.number_of_nodes() == 0:
        print(f"No artist named '{artist_name}' found in the graph.")
        return

    color_map = {
        'Artist': 'lightblue', 'Theme': 'lightgreen', 'Video': 'salmon',
        'Instrument': 'lightcoral', 'Category': 'gold', 'Keyword': 'khaki',
        'Location': 'lightgrey', 'Municipality': 'plum',
        'District': 'slateblue', 'Region': 'darkcyan',
    }
    node_colors = [color_map.get(d.get('type'), 'grey') for _, d in sub.nodes(data=True)]

    plt.figure(figsize=(200, 200))
    pos = nx.kamada_kawai_layout(sub)
    nx.draw(
        sub, pos, labels={n: str(n) for n in sub.nodes()}, with_labels=True,
        node_color=node_colors, node_size=1000, font_size=8,
        width=1.0, edge_color='gray',
    )
    edge_labels = nx.get_edge_attributes(sub, 'relationship')
    nx.draw_networkx_edge_labels(sub, pos, edge_labels=edge_labels, font_color='red', font_size=7)
    plt.title(f"Subgraph: {artist_name}", fontsize=20)
    plt.show()


if __name__ == '__main__':
    # Standalone run needs a Flask app context to query the Project model.
    # Usage: python -m ai.rag.knowledgeGraph "Artist Name"
    from flask import Flask
    from database.setup import initDatabase
    from ai.rag.setup import loadProjectRows

    if len(sys.argv) < 2:
        print('Usage: python -m ai.rag.knowledgeGraph "Artist Name"')
        sys.exit(1)

    artist_name = sys.argv[1]

    app = Flask(__name__)
    initDatabase(app)

    with app.app_context():
        all_rows = loadProjectRows()
        if not all_rows:
            print("No data found in the database. Please ensure the database is populated.")
        else:
            graph = create_knowledge_graph_from_data(all_rows)
            visualize_artist_subgraph(graph, artist_name)
