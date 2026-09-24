'''
/database/ingestTranscripts.py
-> reads video-to-text/results.csv and merges audio_transcription/visual_description
   into matching Project rows, keyed by the numeric Vimeo id parsed from Link
'''
import csv
import os
from database.models import db, Project

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video-to-text', 'results.csv')
MAX_FIELD_LENGTH = 20_000

# A few transcripts run into the hundreds of thousands of characters
# (repetition-loop artifacts); the stdlib csv module's default field size
# limit (131072) is too small for those rows.
csv.field_size_limit(10_000_000)


def _extract_vimeo_id(link):
    """Mirrors fetchData.py's extraction so both scripts treat the same links identically
    (a naive vimeo\\.com/(\\d+) regex silently drops rows with malformed double-slash links,
    e.g. vimeo.com//828650429, that split-based extraction handles fine). Also mirrors
    fetchData.py's single-trailing-char strip for links like vimeo.com/671882581/ or
    vimeo.com/54685223# that don't end in a digit."""
    if not isinstance(link, str):
        return None
    cleaned = link.replace(' ', '').replace('\n', '')
    if 'vimeo.com/' not in cleaned:
        return None
    if cleaned and not cleaned[-1].isdigit():
        cleaned = cleaned[:-1]
    tail = cleaned.split('/')[-1]
    return int(tail) if tail.isdigit() else None


def _read_rows(csv_path):
    """
    Parses results.csv (no header row; append-only log resumed many times
    over its history) and yields (link, audio_transcription, visual_description)
    for every structurally-recoverable row. Returns (total, dropped) counts
    via the mutable `stats` dict passed in, since this is a generator.

    Two row shapes occur in the file:
    - 7 fields: Index, Sheet, Artist, Title, Link, Audio transcription,
      Visual description -- the normal shape written by video-to-text/main.py.
      Only kept when Sheet is 'ESPANHA' or 'VIMEO'; a couple of rows have an
      unescaped quote that spills content into the Sheet column instead.
    - 6 fields: Index, Artist, Title, Link, Audio transcription,
      Visual description -- the Sheet column is missing entirely (46 rows in
      the file as of this writing, apparently appended by a differently-shaped
      run of the source tool). Still fully usable: Sheet isn't read for
      anything beyond validating the row shape.
    Any other field count is dropped as unrecoverable.
    """
    with open(csv_path, encoding='utf-8', newline='') as f:
        for row in csv.reader(f):
            if len(row) == 7 and row[1] in ('ESPANHA', 'VIMEO'):
                yield row[4], row[5], row[6]
            elif len(row) == 6:
                yield row[3], row[4], row[5]
            else:
                yield None


def ingestTranscripts(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        print(f"[ingestTranscripts] {csv_path} not found, skipping.")
        return {"matched": 0, "unmatched": 0, "total": 0}

    # link -> (audio, visual), keyed by vimeo_id so a re-processed video's
    # later row (further down the file) overwrites its earlier one
    by_vimeo_id = {}
    total, dropped_malformed = 0, 0
    for parsed in _read_rows(csv_path):
        total += 1
        if parsed is None:
            dropped_malformed += 1
            continue
        link, audio, visual = parsed
        vimeo_id = _extract_vimeo_id(link)
        if vimeo_id is None:
            dropped_malformed += 1
            continue
        by_vimeo_id[vimeo_id] = (audio, visual)

    matched, unmatched, capped = 0, 0, 0
    for i, (vimeo_id, (audio, visual)) in enumerate(by_vimeo_id.items(), 1):
        project = Project.query.filter_by(id=vimeo_id).first()   # vimeo_id is a plain int -- Project.query.filter_by(id=np.int64(...)) silently matches nothing
        if not project:
            unmatched += 1
            continue

        if len(audio) > MAX_FIELD_LENGTH or len(visual) > MAX_FIELD_LENGTH:
            capped += 1
        project.audio_transcription = audio[:MAX_FIELD_LENGTH]
        project.visual_description = visual[:MAX_FIELD_LENGTH]
        matched += 1

        if i % 500 == 0:
            db.session.commit()

    db.session.commit()
    print(f"[ingestTranscripts] total_rows={total} dropped_malformed={dropped_malformed} "
          f"unique_ids={len(by_vimeo_id)} matched={matched} unmatched={unmatched} length_capped={capped}")
    return {"matched": matched, "unmatched": unmatched, "total": total}


if __name__ == '__main__':
    from flask import Flask
    from database.setup import initDatabase
    from ai.rag.setup import rebuildRag

    app = Flask(__name__)
    initDatabase(app)

    with app.app_context():
        ingestTranscripts()

    # Rebuilds this throwaway process's own graph state as a self-contained
    # smoke test (prints updated node/artist/text-token counts). Has NO effect
    # on a separately-running app.py/gunicorn process -- that process picks up
    # the new columns next time its own nightly job runs fetchCSVAndRebuildRag,
    # or on its next restart.
    rebuildRag(app)
