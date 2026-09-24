'''
/database/buildChunkEmbeddings.py
-> chunks every Project's long text fields, embeds each unique chunk via
   Ollama (bge-m3), and persists the vector index for semantic search.
   Standalone script -- not wired into app startup or the nightly scheduler,
   same rationale as ingestTranscripts.py: this is slow and should only run
   when explicitly triggered.
'''
import sys
import time

from database.models import Project
from ai.rag.chunking import chunk_text
from ai.rag.embeddings import embed_texts, save_chunk_index

LONG_TEXT_FIELDS = ('biographies', 'history', 'other_info', 'audio_transcription', 'visual_description')
BATCH_SIZE = 24
CHECKPOINT_EVERY_BATCHES = 20


def build_chunk_records(project):
    """
    Yields one dict per chunk for a single Project row, keyed exactly like
    loadProjectRows() maps ai.rag.setup so chunk_meta's artist_name matches
    artist_relationships keys downstream.
    """
    artist_name = project.author or ''
    theme = project.title or ''
    for field in LONG_TEXT_FIELDS:
        raw = getattr(project, field) or ''
        for i, chunk in enumerate(chunk_text(raw)):
            yield {
                'video_id': project.id,
                'artist_name': artist_name,
                'field': field,
                'theme': theme,
                'chunk_index': i,
                'text': chunk,
            }


def buildChunkEmbeddings(limit=None):
    projects = Project.query.limit(limit).all() if limit else Project.query.all()

    records = []
    for project in projects:
        records.extend(build_chunk_records(project))

    print(f"[buildChunkEmbeddings] {len(projects)} projects -> {len(records)} chunks")

    # Dedupe by exact chunk text before calling Ollama -- ~18% of
    # audio_transcription values are the literal "Instrumental - no lyrics
    # spoken or sung." sentinel; embed each unique string once and assign
    # the same vector to every record sharing that text.
    unique_texts = list(dict.fromkeys(r['text'] for r in records if r['text'].strip()))
    print(f"[buildChunkEmbeddings] {len(unique_texts)} unique chunk texts to embed")

    text_to_vector = {}
    start = time.time()
    for batch_start in range(0, len(unique_texts), BATCH_SIZE):
        batch = unique_texts[batch_start:batch_start + BATCH_SIZE]
        vectors = embed_texts(batch)
        for text, vector in zip(batch, vectors):
            text_to_vector[text] = vector

        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(unique_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        elapsed = time.time() - start
        print(f"[buildChunkEmbeddings] batch {batch_num}/{total_batches} "
              f"({len(text_to_vector)}/{len(unique_texts)} texts embedded, {elapsed:.0f}s elapsed)")

        if batch_num % CHECKPOINT_EVERY_BATCHES == 0:
            _save_partial(records, text_to_vector)

    all_vectors, all_meta = _assemble(records, text_to_vector)
    save_chunk_index(all_vectors, all_meta)
    print(f"[buildChunkEmbeddings] done: {len(all_meta)} chunks, {len(unique_texts)} unique embeddings, "
          f"{time.time() - start:.0f}s total")
    return {"chunks": len(all_meta), "unique_texts": len(unique_texts)}


def _assemble(records, text_to_vector):
    vectors, meta = [], []
    for r in records:
        vector = text_to_vector.get(r['text'])
        if vector is None:
            continue  # empty/whitespace-only chunk text, skipped above
        vectors.append(vector)
        meta.append(r)
    return vectors, meta


def _save_partial(records, text_to_vector):
    covered = [r for r in records if r['text'] in text_to_vector]
    if not covered:
        return
    vectors, meta = _assemble(covered, text_to_vector)
    save_chunk_index(vectors, meta)
    print(f"[buildChunkEmbeddings] checkpoint saved: {len(meta)} chunks so far")


if __name__ == '__main__':
    from flask import Flask
    from database.setup import initDatabase

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    app = Flask(__name__)
    initDatabase(app)

    with app.app_context():
        buildChunkEmbeddings(limit=limit)
