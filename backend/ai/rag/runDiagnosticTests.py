'''
/ai/rag/runDiagnosticTests.py
-> larger test battery than runLoggedTests.py, with rich per-query diagnostics
   (retrieval stage breakdown, discovery block composition, semantic hit
   stats, context sizes, timing) captured to both a markdown log (full
   answers) and a CSV (structured, for later analysis).
   Standalone, throwaway QA script -- not imported by the app.
   Run with: python -m ai.rag.runDiagnosticTests
'''
import csv
import sys
import time

from ai.rag.knowledgeGraph import (
    create_knowledge_graph_from_data,
    build_relationships_dict,
    build_artist_projection,
    build_hierarchical_communities,
)
from ai.rag.retrieval import (
    build_text_index, _semantic_search, _tokenize,
    retrieve_exploration_context, SEMANTIC_PROMOTE_THRESHOLD, SEMANTIC_TOP_K,
)
from ai.rag.ragQuery import answer_question_with_exploration
from ai.rag.setup import loadProjectRows
from ai.rag.embeddings import load_chunk_index

SCRATCH = r"c:\Users\joanm\AppData\Local\Temp\claude\c--Users-joanm-Documents-LASTRO-lastro-mpagdp-copy\782d0aff-5419-494e-a517-aeae4577c8e9\scratchpad"
LOG_PATH = SCRATCH + r"\rag_diag_v1.md"
CSV_PATH = SCRATCH + r"\rag_diag_v1.csv"

QUERIES = [
    ("Direct artist", "Quem e Dazkarieh?"),
    ("Direct artist", "Quem e Duarte Silveira?"),
    ("Region", "Que artistas existem no Alentejo?"),
    ("Region", "Que artistas existem nos Acores?"),
    ("Region", "Ha artistas do Minho no arquivo?"),
    ("Instrument", "Quem toca gaita de foles?"),
    ("Instrument", "Quem toca viola campanica?"),
    ("Instrument", "Quem toca adufe?"),
    ("Category/genre", "Mostra-me exemplos de musica narrativa."),
    ("Category/genre", "Existem exemplos de paisagem sonora no arquivo?"),
    ("Abstract/disruptive", "Musica que referencie pastel de nata."),
    ("Abstract/disruptive", "Ha artistas cuja historia mencione emigracao?"),
    ("Abstract/disruptive", "Que ligacoes existem entre musica e artesanato no arquivo?"),
    ("Abstract/disruptive", "Como e que a religiao aparece documentada no arquivo?"),
    ("Broad thematic", "Que papel tem a voz na musica tradicional portuguesa?"),
    ("Broad thematic", "Como e que a musica se relaciona com o trabalho agricola?"),
    ("Edge case", "Existem registos de fado neste arquivo?"),
    ("Edge case", "Ha musica rock no arquivo?"),
    ("Multi-artist comparison", "Qual e a diferenca entre Celio Pires e Antonio Andre?"),
    ("Multi-entity", "Quem toca sanfona e cavaquinho?"),
    ("Multi-entity", "Ha artistas do Alentejo e do Algarve que cantam sobre o mar?"),
    ("Community/group", "Que grupos ou ranchos folcloricos estao documentados no arquivo?"),
    ("Date-based", "Que projetos foram gravados em 2023?"),
    ("Negative/no-result", "Existem gravacoes de opera italiana no arquivo?"),
    ("Paraphrase", "Que musica fala sobre o cheiro do pao a cozer no forno?"),
    ("Paraphrase", "Ha relatos de saudade de quem foi trabalhar para o estrangeiro?"),
    ("Paraphrase", "Que videos mencionam bruxas ou feiticaria?"),
    ("Paraphrase", "Ha video que mostrem criancas a brincar na rua?"),
    ("Paraphrase", "Que musica transmite um sentimento de perda irreparavel?"),
    ("Paraphrase", "Que artistas descrevem rituais ou bencaos praticadas pelos mais velhos?"),
    ("Paraphrase", "Existem historias sobre pratos tipicos passados de geracao em geracao?"),
    ("Paraphrase", "Ha testemunhos sobre a dificuldade da vida no campo?"),
    ("Paraphrase", "Que artistas falam sobre a construcao artesanal de instrumentos?"),
    ("Compound/complex", "Que artistas do Alentejo tocam viola campanica e tambem falam de tradicoes religiosas?"),
]


def _classify_block(block_text):
    """Maps a discovery_blocks[i] string back to which mechanism produced
    it, using each block's distinguishing marker (see retrieval.py's
    comments for A/B/C/C2/C3/E/F/D). Checked in priority order since some
    markers are substrings of what another block type could coincidentally
    contain in free-text content."""
    if '(mesma comunidade de' in block_text:
        return 'community_spotlight'
    if 'Outros artistas desta comunidade:' in block_text:
        return 'community_namelist'
    if '(categoria fora da musica:' in block_text:
        return 'disruptive'
    if '(correspondencia semantica)' in block_text:
        return 'semantic'
    if '(mesma regiao:' in block_text:
        return 'geographic'
    if 'Conceitos partilhados:' in block_text:
        return 'concept'
    if 'Temas culturais partilhados:' in block_text:
        return 'cultural'
    if block_text.startswith('  [Sobre '):
        return 'primary_text'
    return 'text_fallback'


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

    csv_rows = []
    csv_fields = [
        'index', 'category', 'question', 'error',
        'primary_artist_count', 'primary_artists',
        'direct_blocks', 'discovery_blocks_total',
        'blk_primary_text', 'blk_geographic', 'blk_concept', 'blk_cultural',
        'blk_semantic', 'blk_community_spotlight', 'blk_community_namelist',
        'blk_disruptive', 'blk_text_fallback',
        'community_summary_used',
        'semantic_hit_count', 'semantic_top_score', 'semantic_promoted',
        'direct_context_chars', 'discovery_context_chars', 'community_context_chars',
        'total_context_chars',
        'generation_time_s', 'answer_words', 'answer_links',
        'non_music_mentions',
    ]

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        _log(f, f"# RAG diagnostic test log -- {len(QUERIES)} queries")
        _log(f, f"\nGraph: {graph.number_of_nodes()} nodes, {len(artist_relationships)} artists, "
                f"{artist_proj.number_of_edges()} artist-artist edges, "
                f"{sum(len(c) for c in community_hierarchy.values())} communities, "
                f"{len(chunk_meta)} semantic chunks loaded.\n")

        for i, (category, question) in enumerate(QUERIES, 1):
            print(f"[{i}/{len(QUERIES)}] ({category}) {question}")
            _log(f, f"\n## {i}. [{category}] {question}\n")

            row = {'index': i, 'category': category, 'question': question, 'error': ''}

            try:
                ctx = retrieve_exploration_context(
                    question, artist_relationships, name_to_artists,
                    community_hierarchy, text_token_index, artist_texts,
                    chunk_vectors=chunk_vectors, chunk_meta=chunk_meta,
                )
            except Exception as e:
                _log(f, f"*RETRIEVAL ERROR: {e}*")
                row['error'] = f'retrieval: {e}'
                csv_rows.append(row)
                continue

            primary_names = []
            for block in ctx['direct']:
                first_line = block.splitlines()[0]
                primary_names.append(first_line.replace('Artist: ', ''))

            block_counts = {k: 0 for k in [
                'primary_text', 'geographic', 'concept', 'cultural', 'semantic',
                'community_spotlight', 'community_namelist', 'disruptive', 'text_fallback',
            ]}
            for block in ctx['discoveries']:
                block_counts[_classify_block(block)] += 1

            semantic_hits = _semantic_search(question, chunk_vectors, chunk_meta, top_k=SEMANTIC_TOP_K)
            top_score = semantic_hits[0][0] if semantic_hits else 0.0

            direct_context = '\n\n'.join(ctx['direct']) if ctx['direct'] else ''
            discovery_context = '\n\n'.join(ctx['discoveries']) if ctx['discoveries'] else ''
            community_context = ctx.get('community_summary', '') or ''

            row.update({
                'primary_artist_count': len(primary_names),
                'primary_artists': '; '.join(primary_names),
                'direct_blocks': len(ctx['direct']),
                'discovery_blocks_total': len(ctx['discoveries']),
                'blk_primary_text': block_counts['primary_text'],
                'blk_geographic': block_counts['geographic'],
                'blk_concept': block_counts['concept'],
                'blk_cultural': block_counts['cultural'],
                'blk_semantic': block_counts['semantic'],
                'blk_community_spotlight': block_counts['community_spotlight'],
                'blk_community_namelist': block_counts['community_namelist'],
                'blk_disruptive': block_counts['disruptive'],
                'blk_text_fallback': block_counts['text_fallback'],
                'community_summary_used': bool(ctx.get('community_summary')),
                'semantic_hit_count': len(semantic_hits),
                'semantic_top_score': round(top_score, 4),
                'semantic_promoted': top_score >= SEMANTIC_PROMOTE_THRESHOLD,
                'direct_context_chars': len(direct_context),
                'discovery_context_chars': len(discovery_context),
                'community_context_chars': len(community_context),
                'total_context_chars': len(direct_context) + len(discovery_context) + len(community_context),
            })

            _log(f, f"*Primary artists ({row['primary_artist_count']}): {row['primary_artists']}*")
            _log(f, f"*Discovery blocks ({row['discovery_blocks_total']}): "
                    f"geo={block_counts['geographic']} concept={block_counts['concept']} "
                    f"cultural={block_counts['cultural']} semantic={block_counts['semantic']} "
                    f"community_spotlight={block_counts['community_spotlight']} "
                    f"community_namelist={block_counts['community_namelist']} "
                    f"disruptive={block_counts['disruptive']} fallback={block_counts['text_fallback']}*")
            _log(f, f"*Semantic: {row['semantic_hit_count']} hits above threshold, "
                    f"top score={row['semantic_top_score']:.3f}"
                    f"{' (PROMOTED)' if row['semantic_promoted'] else ''}*")
            _log(f, f"*Context size: direct={row['direct_context_chars']}, "
                    f"discovery={row['discovery_context_chars']}, "
                    f"community={row['community_context_chars']}, "
                    f"total={row['total_context_chars']} chars*\n")

            t0 = time.time()
            try:
                answer = answer_question_with_exploration(
                    question, artist_relationships, name_to_artists,
                    community_hierarchy, text_token_index, artist_texts,
                    chunk_vectors=chunk_vectors, chunk_meta=chunk_meta,
                )
            except Exception as e:
                answer = f"ERROR: {e}"
                row['error'] = f'generation: {e}'
            elapsed = time.time() - t0

            row['generation_time_s'] = round(elapsed, 1)
            row['answer_words'] = len(answer.split())
            row['answer_links'] = answer.count('http')
            row['non_music_mentions'] = answer.lower().count('categoria fora da m')

            _log(f, f"*Generation time: {elapsed:.1f}s, {row['answer_words']} words, "
                    f"{row['answer_links']} links, {row['non_music_mentions']} non-music-category mentions*\n")
            _log(f, "**Answer:**\n")
            _log(f, answer)
            _log(f, "\n---")

            csv_rows.append(row)

        _log(f, "\n# Log complete")

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"\nDone. Markdown log: {LOG_PATH}")
    print(f"CSV diagnostics: {CSV_PATH}")


if __name__ == '__main__':
    from flask import Flask
    from database.setup import initDatabase

    app = Flask(__name__)
    initDatabase(app)

    with app.app_context():
        run()
