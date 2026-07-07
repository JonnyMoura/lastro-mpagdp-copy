import csv
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import math
import sqlite3
import igraph as ig
import leidenalg
import sys


def create_knowledge_graph_from_data(data_rows):
    """
    Creates a weighted knowledge graph using unambiguous (name, type) tuples as node identifiers.
    This prevents conflicts where the same name exists for different types (e.g., an artist and a theme).
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

    for row in data_rows:
        artist_name = row.get('Nome')

        artist_node = (artist_name, 'Artist')
        if not G.has_node(artist_node):
            G.add_node(artist_node, type='Artist', name=artist_name)

        if row.get('Tema'):
            theme_name = row['Tema']
            theme_node = (theme_name, 'Theme')
            if not G.has_node(theme_node): G.add_node(theme_node, type='Theme', name=theme_name)
            G.add_edge(artist_node, theme_node, relationship='has_theme', weight=get_weight(theme_name))

        if row.get('Instrumentos'):
            for name in [i.strip() for i in row['Instrumentos'].split(',') if i.strip()]:
                node = (name, 'Instrument')
                if not G.has_node(node): G.add_node(node, type='Instrument', name=name)
                G.add_edge(artist_node, node, relationship='plays', weight=get_weight(name))

        if row.get('Categoria'):
            for name in [c.strip() for c in row['Categoria'].split(',') if c.strip()]:
                node = (name, 'Category')
                if not G.has_node(node): G.add_node(node, type='Category', name=name)
                G.add_edge(artist_node, node, relationship='in_category', weight=get_weight(name))

        if row.get('keywords'):
            for name in [k.strip() for k in row['keywords'].split(',') if k.strip()]:
                node = (name, 'Keyword')
                if not G.has_node(node): G.add_node(node, type='Keyword', name=name)
                G.add_edge(artist_node, node, relationship='has_keyword', weight=get_weight(name))

        # Add history, other_info, and biographies as attributes to the artist node
        if row.get('history'):
            G.nodes[artist_node]['history'] = row['history']
        if row.get('other_info'):
            G.nodes[artist_node]['other_info'] = row['other_info']
        if row.get('biographies'):
            G.nodes[artist_node]['biographies'] = row['biographies']

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
            G.add_edge(artist_node, local_node, relationship='located_in', weight=get_weight(local_name))

        if local_node and concelho_node:
            G.add_edge(local_node, concelho_node, relationship='part_of', weight=1.0)
        if concelho_node and distrito_node:
            G.add_edge(concelho_node, distrito_node, relationship='belongs_to', weight=1.0)
        if distrito_node and regiao_node:
            G.add_edge(distrito_node, regiao_node, relationship='belongs_to', weight=1.0)

    return G


def build_relationships_dict(graph):
    """
    Converts the graph into a flat dictionary keyed by artist name.
    Each entry lists all directly connected entities, grouped by type.

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
    # type → plural key used in the artist dict
    TYPE_KEY = {
        'Theme':        'themes',
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
        
        # Add node attributes to the entry
        if 'history' in data:
            entry['history'] = data['history']
        if 'other_info' in data:
            entry['other_info'] = data['other_info']
        if 'biographies' in data:
            entry['biographies'] = data['biographies']

        for neighbor in graph.neighbors(node):
            neighbor_data = graph.nodes[neighbor]
            neighbor_type = neighbor_data.get('type', '')
            neighbor_name = neighbor_data.get('name', '')

            key = TYPE_KEY.get(neighbor_type)
            if key and neighbor_name:
                entry[key].append(neighbor_name)
                name_to_artists[neighbor_name.lower()].add(artist_name)

            # Location is the only direct geographic link to the artist.
            # Walk up the hierarchy to collect Municipality → District → Region.
            if neighbor_type == 'Location':
                current = neighbor
                for expected_type in ('Municipality', 'District', 'Region'):
                    for hop in graph.neighbors(current):
                        hop_data  = graph.nodes[hop]
                        hop_type  = hop_data.get('type', '')
                        hop_name  = hop_data.get('name', '')
                        if hop_type == expected_type and hop_name:
                            geo_key = TYPE_KEY[expected_type]
                            entry[geo_key].append(hop_name)
                            name_to_artists[hop_name.lower()].add(artist_name)
                            current = hop  # advance up the chain
                            break

        # Deduplicate list-based fields while preserving order
        for key in TYPE_KEY.values():
            if key in entry and isinstance(entry[key], list):
                entry[key] = list(dict.fromkeys(entry[key]))

        artist_relationships[artist_name] = entry
        name_to_artists[artist_name.lower()].add(artist_name)

    return artist_relationships, dict(name_to_artists)


def build_artist_projection(graph, entity_types=None, max_entity_share=0.10):
    """
    Projects the heterogeneous knowledge graph onto an artist-artist similarity
    graph. Two artists share an edge if they have common entities, weighted by
    the inverse of how many artists share each entity (rarity bonus).

    Excludes geographic nodes by default so communities reflect musical/thematic
    similarity rather than physical proximity.

    Entities shared by more than max_entity_share fraction of all artists are
    dropped (e.g. 'voz') — they add noise without signal.
    """
    if entity_types is None:
        entity_types = {'Theme', 'Instrument', 'Category', 'Keyword', 'Location'}

    entity_to_artists = defaultdict(set)
    artist_nodes = set()
    for node, data in graph.nodes(data=True):
        if data.get('type') != 'Artist':
            continue
        artist_nodes.add(node)
        for neighbor in graph.neighbors(node):
            ndata = graph.nodes[neighbor]
            if ndata.get('type') in entity_types:
                entity_to_artists[neighbor].add(node)

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
                ndata = full_graph.nodes[neighbor]
                ntype = ndata.get('type', '')
                nname = ndata.get('name', '')
                if ntype and nname and ntype != 'Artist':
                    entity_buckets[ntype][nname] += 1

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


def build_hierarchical_communities(artist_proj, full_graph,
                                   resolutions=None):
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


def fetch_data_from_db(db_path='lastro.db'):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM projects")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            'Nome':          dict(r).get('author'),
            'Tema':          dict(r).get('title'),
            'Instrumentos':  dict(r).get('instruments'),
            'Categoria':     dict(r).get('category'),
            'Local':         dict(r).get('location'),
            'Concelho':      dict(r).get('municipality'),
            'Distrito/Ilha': dict(r).get('district'),
            'Região':        dict(r).get('region'),
            'keywords':      dict(r).get('keywords'),
            'history':       dict(r).get('history'),
            'other_info':    dict(r).get('other_info'),
            'biographies':   dict(r).get('biographies'),
        }
        for r in rows
    ]


def fetch_data_from_csv(csv_path=r'C://Users//joanm//Downloads//Base dados - VIMEO.csv'):
    data_rows = []
    encodings = ['utf-8', 'latin-1']
    for enc in encodings:
        try:
            with open(csv_path, 'r', encoding=enc) as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    data_rows.append({
                        'Nome':          row.get('Nome', '').strip(),
                        'Tema':          row.get('Tema', '').strip(),
                        'Instrumentos':  row.get('Instrumentos', '').strip(),
                        'Categoria':     row.get('Categoria', '').strip(),
                        'Local':         row.get('Local', '').strip(),
                        'Concelho':      row.get('Concelho', '').strip(),
                        'Distrito/Ilha': row.get('Distrito/Ilha', '').strip(),
                        'Região':        row.get('Região', '').strip(),
                    })
            print(f"Loaded {len(data_rows)} rows from CSV ({enc}).")
            return data_rows
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            return []
        except UnicodeDecodeError:
            continue
    return data_rows


if __name__ == '__main__':
    data_rows = fetch_data_from_db()
    if not data_rows:
        print("No data found in database. Exiting.")
        sys.exit(1)

    full_graph = create_knowledge_graph_from_data(data_rows)
    artist_relationships, name_to_artists = build_relationships_dict(full_graph)

    artist_proj = build_artist_projection(full_graph)
    hierarchy = build_hierarchical_communities(artist_proj, full_graph)

    print(f"Full graph: {full_graph.number_of_nodes()} nodes, {full_graph.number_of_edges()} edges")
    print(f"Artist projection: {artist_proj.number_of_nodes()} nodes, {artist_proj.number_of_edges()} edges")
    for res, comms in sorted(hierarchy.items()):
        print(f"  Resolution {res}: {len(comms)} communities")

    if len(sys.argv) > 1:
        search_term = sys.argv[1]
        print(f"\nSearching for artists related to '{search_term}'...")

        matched_artists = {
            artist for name, artists in name_to_artists.items()
            if search_term.lower() in name.lower()
            for artist in artists
        }

        if matched_artists:
            for artist_name in sorted(matched_artists):
                print(f"\n--- Entry for '{artist_name}' ---")
                entry = artist_relationships.get(artist_name)
                if entry:
                    for k, v in entry.items():
                        if v:
                            print(f"  {k}: {v}")
                else:
                    print(f"  No detailed entry found for '{artist_name}'.")
        else:
            print(f"No artists found matching '{search_term}'.")

    else:
        print("\nNo search term provided. Showing a sample artist entry.")
        if artist_relationships:
            sample_artist = next(iter(artist_relationships))
            print(f"\n--- Sample entry — '{sample_artist}' ---")
            for k, v in artist_relationships[sample_artist].items():
                if v:
                    print(f"  {k}: {v}")
        else:
            print("No artist relationships were built.")