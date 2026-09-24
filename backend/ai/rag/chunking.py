'''
/ai/rag/chunking.py
-> splits long text fields into sentence-aware, embedding-sized chunks
'''
import re

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
# bge-m3's architecture supports an 8192-token context, but this Ollama
# deployment silently truncates embed input well below that regardless of
# any num_ctx override (confirmed empirically: cosine similarity between a
# 5000-char and a 20000-char version of the same text is 0.9997, and by
# 5500 chars it's an exact 1.0 match to the full text -- content beyond
# that point never reaches the model. A custom Modelfile with
# `PARAMETER num_ctx 8192` was tested and made no difference, so this is
# not a request-time or Modelfile-level setting; it's baked into how this
# Ollama build serves bge-m3's embedding/pooling path). TARGET_CHUNK_CHARS
# is set with margin under that confirmed ceiling. At this size, 97.7% of
# real (non-instrumental-sentinel) audio_transcription values fit in a
# single, unfragmented chunk, up from 86.5% at the old 1200/1800 setting --
# and since visual_description/history/other_info/biographies are all
# shorter than audio_transcription at every percentile, this comfortably
# covers them too.
TARGET_CHUNK_CHARS = 4800
MAX_CHUNK_CHARS = 5200
OVERLAP_CHARS = 600
MIN_TRAILING_CHUNK_CHARS = 800


def _split_sentences(text):
    return [p.strip() for p in SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]


def _hard_split(sentence, max_chars=MAX_CHUNK_CHARS):
    """
    Word-boundary fallback for a single "sentence" longer than max_chars --
    ASR repetition-loop artifacts (e.g. transcripts capped at 20,000 chars
    with little to no punctuation) would otherwise become one giant
    unsplittable chunk that destroys retrieval precision for that video.
    """
    words = sentence.split(' ')
    out, current = [], ''
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            out.append(current)
            current = word
        else:
            current = f'{current} {word}'.strip()
    if current:
        out.append(current)
    return out


def chunk_text(text, target_chars=TARGET_CHUNK_CHARS, max_chars=MAX_CHUNK_CHARS,
               overlap_chars=OVERLAP_CHARS):
    """
    Splits text into sentence-aware chunks around target_chars, carrying the
    trailing ~overlap_chars of one chunk into the start of the next so
    meaning isn't lost at a boundary. A too-short trailing chunk is merged
    into its predecessor instead of being embedded as a near-empty vector.
    """
    if not text or not text.strip():
        return []

    sentences = []
    for sentence in _split_sentences(text):
        if len(sentence) > max_chars:
            sentences.extend(_hard_split(sentence, max_chars))
        else:
            sentences.append(sentence)

    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        if current and current_len + 1 + len(sentence) > target_chars:
            chunks.append(' '.join(current))
            tail, tail_len = [], 0
            for s in reversed(current):
                if tail_len >= overlap_chars:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            current, current_len = tail, tail_len
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(' '.join(current))

    if len(chunks) >= 2 and len(chunks[-1]) < MIN_TRAILING_CHUNK_CHARS:
        chunks[-2] = chunks[-2] + ' ' + chunks[-1]
        chunks.pop()

    return chunks
