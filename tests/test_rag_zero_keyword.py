"""
Additional tests for zero-keyword semantic search (Issue #47).

These tests are APPENDED to the existing tests/test_rag.py — they cover:
  * Query expansion (`_expand_query`)
  * Command text preprocessing (`_preprocess_command_text`)
  * Zero-keyword hybrid search (conceptual query -> semantic match)
  * Semantic-only mode (alpha=1.0 bypasses FTS)
  * Preservation of the existing FTS-miss fallback path
"""

import math
import pytest
from termstory.database import Database
from termstory.models import Project, Session, Command


# --- Reuse the mocks from the existing test module (re-declared here so the
# --- new test file is self-contained). --------------------------------------

class MockLinalg:
    @staticmethod
    def norm(v):
        return math.sqrt(sum(x * x for x in v))


class MockNp:
    linalg = MockLinalg
    ndarray = list

    @staticmethod
    def dot(v1, v2):
        return sum(x * y for x, y in zip(v1, v2))


class MockSentenceTransformer:
    """Embedding mock that maps by keyword presence.

    The mock mimics the *spirit* of all-MiniLM-L6-v2: it produces the same
    embedding for any text containing a given keyword. This lets us assert
    that the query-expansion pipeline (`_expand_query`) actually causes the
    expanded query to land in the same semantic neighborhood as the target
    document.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        embeddings = []
        for text in texts:
            text_lower = text.lower()
            if "docker" in text_lower:
                embeddings.append([1.0, 0.0, 0.0])
            elif "pytest" in text_lower or "test" in text_lower:
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])
        return embeddings


# ---------------------------------------------------------------------------
# Unit tests for the new preprocessing helpers
# ---------------------------------------------------------------------------

def test_expand_query_adds_anchors_for_known_topics():
    from termstory.rag import _expand_query

    # 'containerization' is a key in _TOPIC_EXPANSIONS — should pull in
    # 'docker', 'kubernetes', 'k8s', etc.
    expanded = _expand_query("containerization")
    assert "containerization" in expanded
    assert "docker" in expanded
    assert "kubernetes" in expanded

    # Original query always preserved verbatim at the front.
    assert expanded.startswith("containerization")

    # Multi-word phrase anchor.
    expanded_phrase = _expand_query("how does version control work here")
    assert "git" in expanded_phrase
    assert "commit" in expanded_phrase

    # Unknown topic: query returned unchanged (no expansion).
    assert _expand_query("flibbertigibbet") == "flibbertigibbet"

    # Empty query short-circuits.
    assert _expand_query("") == ""


def test_expand_query_is_capped_to_eight_anchors():
    from termstory.rag import _expand_query, _TOPIC_EXPANSIONS

    # Find a topic with > 8 anchors to verify the cap.
    big_topic = None
    for k, v in _TOPIC_EXPANSIONS.items():
        if len(v) > 8 and ' ' not in k:
            big_topic = k
            break
    if big_topic is None:
        pytest.skip("No topic with >8 single-token anchors in the map")

    expanded = _expand_query(big_topic)
    # The "(related: ...)" suffix should contain at most 8 comma-separated
    # anchor terms.
    suffix = expanded.split("(related:", 1)[1]
    anchors = [a.strip().rstrip(")") for a in suffix.split(",")]
    assert len(anchors) <= 8


def test_preprocess_command_text_expands_shorthands():
    from termstory.rag import _preprocess_command_text

    # Known shorthand gets expanded inline.
    out = _preprocess_command_text("ps -a")
    assert "ps" in out  # original token preserved
    assert "process status" in out  # expansion appended in parens

    # Multiple known shorthands in one command.
    out2 = _preprocess_command_text("sudo rm -rf /tmp/foo")
    assert "superuser" in out2
    assert "remove delete" in out2

    # Unknown command passes through unchanged.
    out3 = _preprocess_command_text("foobar --flag value")
    assert out3 == "foobar --flag value"

    # Empty / None-safe.
    assert _preprocess_command_text("") == ""


def test_build_session_document_uses_preprocessed_commands():
    from termstory.rag import _build_session_document

    session = {
        "project_name": "Demo",
        "ai_summary": "Running process checks",
        "all_commands": ["ps -a", "docker ps"],
        "all_commits": [{"message": "feat: add stuff"}],
    }
    doc = _build_session_document(session)
    # Preprocessing expanded the shorthands...
    assert "process status" in doc
    assert "container runtime" in doc
    # ...and the raw tokens are still individually present (just with
    # parenthetical expansions inserted after them, so the bare command
    # form is NOT contiguous — that's intentional).
    assert "ps" in doc
    assert "docker" in doc
    # Commit message included verbatim.
    assert "feat: add stuff" in doc


# ---------------------------------------------------------------------------
# Integration tests for the new zero-keyword behavior in hybrid_search()
# ---------------------------------------------------------------------------

def _seed_two_session_db(db, now):
    """Seed a DB with one Docker session and one unrelated session."""
    p1 = Project(id=1, name="Docker Registry", path="~/projects/docker",
                 first_seen=now, last_seen=now,
                 session_count=1, total_time=100)
    p2 = Project(id=2, name="Unrelated Work", path="~/projects/other",
                 first_seen=now, last_seen=now,
                 session_count=1, total_time=100)

    cmd1 = Command(timestamp=now, command="docker ps -a", session_id=1, project_id=1)
    s1 = Session(id=1, start_time=now, end_time=now + 100,
                 duration_seconds=100, project_id=1, commands=[cmd1])
    s1.ai_summary = "Running docker daemon container checks"

    cmd2 = Command(timestamp=now + 5000, command="echo hello world",
                   session_id=2, project_id=2)
    s2 = Session(id=2, start_time=now + 5000, end_time=now + 5100,
                 duration_seconds=100, project_id=2, commands=[cmd2])
    s2.ai_summary = "Unrelated shell work"

    db.save_data([p1, p2], [s1, s2], [cmd1, cmd2])
    return p1, p2, s1, s2


def test_hybrid_search_zero_keyword_query(tmp_path, monkeypatch):
    """
    Zero-keyword scenario: query 'containerization' has NO lexical overlap
    with any command/commit in the DB. The system should still surface the
    Docker session as the top result via:
      (a) FTS miss -> fallback to a bounded non-FTS candidate pool, AND
      (b) `_expand_query('containerization')` adds 'docker' to the
          embedding, so the mock SentenceTransformer produces the same
          embedding for both the query and the Docker document.
    """
    monkeypatch.setattr("termstory.rag.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr("termstory.rag.SentenceTransformer", MockSentenceTransformer, raising=False)
    monkeypatch.setattr("termstory.rag.np", MockNp)

    db_file = tmp_path / "test_rag_zero_keyword.db"
    db = Database(str(db_file))
    db.init_db()

    now = 1780000000
    _seed_two_session_db(db, now)

    from termstory.rag import hybrid_search

    results = hybrid_search(db, "containerization", alpha=0.8)

    # FTS will miss -> fallback pulls both sessions.
    assert len(results) >= 1
    # Docker session should be ranked FIRST via semantic similarity.
    assert results[0]["session_id"] == 1
    assert results[0]["project_name"] == "Docker Registry"
    # The zero_keyword flag is surfaced so the UI can indicate semantic-only
    # matches to the user.
    assert results[0]["zero_keyword"] is True
    # Sub-scores are exposed for downstream consumers.
    assert "semantic_score" in results[0]
    assert "bm25_score" in results[0]
    # BM25 score is 0 for every candidate (zero lexical overlap).
    assert results[0]["bm25_score"] == 0.0
    # No fake lexical matches against the expanded query.
    assert results[0]["matching_commands"] == []


def test_hybrid_search_zero_keyword_ranks_relevant_above_irrelevant(tmp_path, monkeypatch):
    """
    When two candidates are returned by the fallback pool, the semantically
    relevant one must rank strictly above the irrelevant one.
    """
    monkeypatch.setattr("termstory.rag.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr("termstory.rag.SentenceTransformer", MockSentenceTransformer, raising=False)
    monkeypatch.setattr("termstory.rag.np", MockNp)

    db_file = tmp_path / "test_rag_zero_keyword_rank.db"
    db = Database(str(db_file))
    db.init_db()

    now = 1780000000
    _seed_two_session_db(db, now)

    from termstory.rag import hybrid_search

    results = hybrid_search(db, "containerization", alpha=0.8)
    assert len(results) == 2
    # Docker session > Unrelated session.
    assert results[0]["session_id"] == 1
    assert results[1]["session_id"] == 2
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]


def test_hybrid_search_semantic_only_mode_bypasses_fts(tmp_path, monkeypatch):
    """
    With alpha >= 0.999 the FTS step is skipped entirely; the candidate
    pool comes from the bounded non-FTS fallback and ranking is purely
    semantic.
    """
    monkeypatch.setattr("termstory.rag.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr("termstory.rag.SentenceTransformer", MockSentenceTransformer, raising=False)
    monkeypatch.setattr("termstory.rag.np", MockNp)

    db_file = tmp_path / "test_rag_semantic_only.db"
    db = Database(str(db_file))
    db.init_db()

    now = 1780000000
    _seed_two_session_db(db, now)

    # Grab a reference to the real advanced_search BEFORE monkeypatching
    # so the spy can delegate without infinite recursion.
    from termstory.search import advanced_search as _real_advanced_search

    calls = []

    def spy_advanced_search(db_obj, query=None, **kwargs):
        calls.append(query)
        return _real_advanced_search(db_obj, query=query, **kwargs)

    monkeypatch.setattr("termstory.search.advanced_search", spy_advanced_search)

    from termstory.rag import hybrid_search

    results = hybrid_search(db, "docker", alpha=1.0)
    # Only the fallback (query=None) call should have happened.
    assert all(q is None for q in calls), calls
    # Docker session still ranks first (semantic match).
    assert results[0]["session_id"] == 1


def test_hybrid_search_preserves_existing_fts_fallback_behavior(tmp_path, monkeypatch):
    """
    Regression test: the existing FTS-miss fallback path (used by the
    pre-issue-#47 test `test_hybrid_search_falls_back_when_fts_has_no_matches`)
    must still trigger exactly twice — once with the query, once with
    `query=None` — when FTS misses.
    """
    monkeypatch.setattr("termstory.rag.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr("termstory.rag.SentenceTransformer", MockSentenceTransformer, raising=False)
    monkeypatch.setattr("termstory.rag.np", MockNp)

    db_file = tmp_path / "test_rag_fallback_regression.db"
    db = Database(str(db_file))
    db.init_db()

    now = 1780000000
    p1 = Project(id=1, name="Docker Registry", path="~/projects/docker",
                 first_seen=now, last_seen=now,
                 session_count=1, total_time=100)
    cmd1 = Command(timestamp=now, command="docker ps -a", session_id=1, project_id=1)
    s1 = Session(id=1, start_time=now, end_time=now + 100,
                 duration_seconds=100, project_id=1, commands=[cmd1])
    s1.ai_summary = "Running docker daemon container checks"
    db.save_data([p1], [s1], [cmd1])

    calls = []

    def fake_advanced_search(db_obj, query=None, **kwargs):
        calls.append(query)
        if query == "containerization":
            return []  # FTS miss
        return [{
            "session_id": 1,
            "start_time": now,
            "end_time": now + 100,
            "duration_seconds": 100,
            "project_id": 1,
            "project_name": "Docker Registry",
            "project_path": "~/projects/docker",
            "ai_summary": "Running docker daemon container checks",
            "all_commands": ["docker ps -a"],
            "matching_commands": [],
            "all_commits": [],
            "matching_commits": [],
        }]

    monkeypatch.setattr("termstory.search.advanced_search", fake_advanced_search)

    from termstory.rag import hybrid_search
    results = hybrid_search(db, "containerization", alpha=0.5)

    assert len(results) == 1
    assert results[0]["session_id"] == 1
    assert calls[0] == "containerization"
    assert calls[1] is None


def test_hybrid_search_empty_query_returns_empty(tmp_path, monkeypatch):
    """
    An empty query should short-circuit to [] WITHOUT invoking FTS or the
    embedding pipeline. This prevents accidentally pulling the entire
    corpus through semantic scoring.
    """
    monkeypatch.setattr("termstory.rag.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr("termstory.rag.SentenceTransformer", MockSentenceTransformer, raising=False)
    monkeypatch.setattr("termstory.rag.np", MockNp)

    db_file = tmp_path / "test_rag_empty_query.db"
    db = Database(str(db_file))
    db.init_db()

    # Spy on advanced_search to make sure it is NOT called.
    calls = []

    def spy_advanced_search(db_obj, query=None, **kwargs):
        calls.append(query)
        return []

    monkeypatch.setattr("termstory.search.advanced_search", spy_advanced_search)

    from termstory.rag import hybrid_search
    results = hybrid_search(db, "", alpha=0.5)

    assert results == []
    assert calls == []


def test_hybrid_search_normal_keyword_query_still_uses_fts_first(tmp_path, monkeypatch):
    """
    Regression test: a normal keyword query (e.g. 'docker') should still
    hit the FTS path FIRST and not unnecessarily fall through to the
    zero-keyword fallback pool.
    """
    monkeypatch.setattr("termstory.rag.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr("termstory.rag.SentenceTransformer", MockSentenceTransformer, raising=False)
    monkeypatch.setattr("termstory.rag.np", MockNp)

    db_file = tmp_path / "test_rag_normal_keyword.db"
    db = Database(str(db_file))
    db.init_db()

    now = 1780000000
    _seed_two_session_db(db, now)

    # Grab a reference to the real advanced_search BEFORE monkeypatching
    # so the spy can delegate without infinite recursion.
    from termstory.search import advanced_search as _real_advanced_search

    calls = []

    def spy_advanced_search(db_obj, query=None, **kwargs):
        calls.append(query)
        return _real_advanced_search(db_obj, query=query, **kwargs)

    monkeypatch.setattr("termstory.search.advanced_search", spy_advanced_search)

    from termstory.rag import hybrid_search
    results = hybrid_search(db, "docker", alpha=0.5)

    # FTS path was used (query == "docker").
    assert "docker" in calls
    # No fallback to query=None should have been necessary.
    assert None not in calls
    # zero_keyword flag is False because 'docker' has lexical overlap.
    assert all(r["zero_keyword"] is False for r in results)
    # Docker session ranked first.
    assert results[0]["session_id"] == 1
    # matching_commands populated as before.
    assert "docker ps -a" in results[0]["matching_commands"]
