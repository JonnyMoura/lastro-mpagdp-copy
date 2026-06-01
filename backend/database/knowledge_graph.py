import csv
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import math
import sqlite3
import igraph as ig
import leidenalg


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
    }

    artist_relationships = {}   # artist_name  -> {type_key: [names]}
    name_to_artists     = defaultdict(set)  # any_name -> {artist_names}

    for node, data in graph.nodes(data=True):
        if data.get('type') != 'Artist':
            continue

        artist_name = data.get('name', '')
        entry = {key: [] for key in TYPE_KEY.values()}

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

        # Deduplicate while preserving order
        for key in entry:
            entry[key] = list(dict.fromkeys(entry[key]))

        artist_relationships[artist_name] = entry
        name_to_artists[artist_name.lower()].add(artist_name)

    return artist_relationships, dict(name_to_artists)


def build_leiden_communities(graph, resolution=1.0):
    """
    Runs the Leiden algorithm on the NetworkX graph and returns a list of
    community objects, each being a dict with:

        {
            "id":        int,                  # community index
            "artists":   [str, ...],           # artist names in this community
            "tokens":    set[str],             # pooled lowercase token bag for retrieval
            "summary":   str,                  # human-readable label for the LLM prompt
        }

    Why Leiden over Louvain?
    - Leiden guarantees well-connected partitions (no disconnected communities).
    - It is faster and more stable across runs at the same resolution.
    - With edge weights from the TF-IDF-like scoring in the graph, it naturally
      clusters artists that share rare instruments, themes, or locations.

    The `resolution` parameter controls granularity:
    - Higher  → more, smaller communities  (fine-grained clusters)
    - Lower   → fewer, larger communities  (broad regional/stylistic groups)
    A value around 0.8–1.2 works well for typical dataset sizes here.

    Returns: list[dict]  (empty list if the graph has no edges)
    """
    if graph.number_of_edges() == 0:
        return []

    # --- 1. Convert NetworkX → igraph, preserving edge weights ---
    # igraph needs integer node ids; we map tuples to ints via a list.
    nx_nodes  = list(graph.nodes())
    node_index = {n: i for i, n in enumerate(nx_nodes)}

    ig_edges  = [(node_index[u], node_index[v]) for u, v in graph.edges()]
    ig_weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]

    ig_graph = ig.Graph(n=len(nx_nodes), edges=ig_edges)
    ig_graph.es['weight'] = ig_weights

    # --- 2. Run Leiden with RBConfigurationVertexPartition (supports weights) ---
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        n_iterations=-1,   # run until convergence
        seed=42,           # reproducible results
    )

    # --- 3. Build community dicts ---
    # Pre-build a lookup: igraph node id → NetworkX node data
    nx_data = [graph.nodes[nx_nodes[i]] for i in range(len(nx_nodes))]

    communities = []
    for cid, member_ids in enumerate(partition):
        artists, tokens = [], set()

        for mid in member_ids:
            node_data = nx_data[mid]
            name      = node_data.get('name', '')
            ntype     = node_data.get('type', '')

            # Collect artist names for cross-referencing with artist_relationships
            if ntype == 'Artist' and name:
                artists.append(name)

            # Pool all entity names as tokens for retrieval matching
            if name:
                # Tokenise: lowercase, split on spaces/punctuation
                for tok in re.sub(r'[^\w\s]', '', name.lower()).split():
                    if len(tok) > 2:
                        tokens.add(tok)

        if not artists:
            continue  # skip communities with no artists (pure geo/instrument hubs)

        # Build a concise readable summary for the LLM prompt
        type_buckets: dict[str, list[str]] = defaultdict(list)
        for mid in member_ids:
            nd = nx_data[mid]
            if nd.get('name'):
                type_buckets[nd.get('type', 'Unknown')].append(nd['name'])

        summary_lines = [f"Community {cid}:"]
        type_order = ['Artist', 'Theme', 'Category', 'Instrument',
                      'Location', 'Municipality', 'District', 'Region']
        for t in type_order:
            names = sorted(set(type_buckets.get(t, [])))
            if names:
                summary_lines.append(f"  {t}(s): {', '.join(names)}")

        communities.append({
            'id':      cid,
            'artists': artists,
            'tokens':  tokens,
            'summary': '\n'.join(summary_lines),
        })

    return communities


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
    data_rows = fetch_data_from_csv()
    full_graph = create_knowledge_graph_from_data(data_rows)

    artist_relationships, name_to_artists = build_relationships_dict(full_graph)

    # Quick inspection
    sample_artist = next(iter(artist_relationships))
    print(f"\nSample entry — '{sample_artist}':")
    for k, v in artist_relationships[sample_artist].items():
        if v:
            print(f"  {k}: {v}")