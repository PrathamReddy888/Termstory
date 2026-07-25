import math
import re
from typing import List, Dict, Optional, Tuple, Any

import threading

# Try to import sentence_transformers and numpy as optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    np = None

# Cache for loaded SentenceTransformer models to avoid expensive reloading
_model_cache = {}
_model_cache_lock = threading.Lock()


def clear_model_cache():
    """
    Clears the cached sentence-transformer models.
    """
    with _model_cache_lock:
        _model_cache.clear()


# ---------------------------------------------------------------------------
# Domain-specific preprocessing maps for command/commit embeddings
# ---------------------------------------------------------------------------
#
# all-MiniLM-L6-v2 is a general-purpose sentence embedding model trained on
# natural-language text. Raw shell commands and short commit subjects are
# *not* natural language, so the model produces poor embeddings for them.
# The two maps below are lightweight, rule-based preprocessing aids that
# bridge this vocabulary gap without requiring a custom-trained model:
#
#   * `_COMMAND_VERB_MAP` expands common shell shorthands into natural-
#     language phrases (e.g. "ps" -> "process status") so the embedding has
#     something meaningful to encode.
#
#   * `_TOPIC_EXPANSIONS` maps abstract conceptual query terms to the
#     concrete vocabulary that typically appears in commits/commands. This
#     is what powers zero-keyword queries: a user searching "containerization"
#     gets the query enriched with "docker, kubernetes, k8s, compose, pod"
#     before embedding, so the cosine similarity against a `docker ps -a`
#     session is meaningfully non-zero.
#
# Both maps are intentionally small and conservative. They are NOT used for
# FTS/BM25 retrieval (which still operates on the raw query) — only for the
# semantic embedding step. This keeps lexical scoring unaffected while
# improving recall for conceptual queries.

_COMMAND_VERB_MAP: Dict[str, str] = {
    "ps": "process status",
    "ls": "list files",
    "ll": "list files long",
    "rm": "remove delete",
    "rmdir": "remove directory",
    "cp": "copy",
    "mv": "move rename",
    "cd": "change directory",
    "cat": "concatenate print file",
    "less": "view file",
    "head": "show file head",
    "tail": "show file tail",
    "grep": "search text",
    "find": "find files",
    "ssh": "secure shell remote login",
    "scp": "secure copy remote",
    "rsync": "sync files remote",
    "curl": "http request",
    "wget": "download",
    "tar": "archive extract",
    "zip": "compress archive",
    "unzip": "extract archive",
    "gzip": "compress",
    "gunzip": "decompress",
    "chmod": "change permissions",
    "chown": "change owner",
    "chgrp": "change group",
    "sudo": "superuser privileged",
    "kill": "terminate process",
    "killall": "terminate processes by name",
    "top": "process monitor",
    "htop": "process monitor interactive",
    "df": "disk free space",
    "du": "disk usage",
    "free": "memory usage",
    "ifconfig": "network interface config",
    "ipconfig": "network interface config",
    "ping": "network reachability test",
    "netstat": "network connections",
    "ss": "socket statistics",
    "traceroute": "network route trace",
    "dig": "dns lookup",
    "nslookup": "dns lookup",
    "git": "version control",
    "docker": "container runtime",
    "kubectl": "kubernetes control",
    "helm": "kubernetes package manager",
    "npm": "node package manager",
    "yarn": "node package manager",
    "pip": "python package installer",
    "pipenv": "python environment",
    "poetry": "python dependency manager",
    "cargo": "rust package manager",
    "go": "golang",
    "make": "build tool",
    "cmake": "build system",
    "pytest": "python test runner",
    "jest": "javascript test runner",
    "mocha": "javascript test runner",
    "tox": "python test environments",
    "ansible": "configuration automation",
    "terraform": "infrastructure as code",
    "vagrant": "development environment",
    "systemctl": "service control",
    "journalctl": "system logs",
    "crontab": "scheduled tasks",
    "openssl": "ssl tls crypto",
}

# Maps abstract conceptual query terms to the concrete vocabulary that
# typically appears in commands/commits. Used only for query expansion in
# `_expand_query()` — never used for FTS/BM25 retrieval.
_TOPIC_EXPANSIONS: Dict[str, List[str]] = {
    "containerization": ["docker", "container", "kubernetes", "k8s", "compose", "pod"],
    "container": ["docker", "pod", "kubernetes", "compose"],
    "containers": ["docker", "pod", "kubernetes", "compose"],
    "orchestration": ["kubernetes", "k8s", "helm", "compose", "swarm"],
    "deployment": ["deploy", "release", "kubernetes", "helm", "ci", "cd", "rollout"],
    "deploy": ["deployment", "release", "rollout", "kubernetes"],
    "ci": ["continuous integration", "github actions", "gitlab ci", "jenkins"],
    "cd": ["continuous deployment", "release", "rollout"],
    "testing": ["test", "pytest", "unittest", "mock", "fixture", "jest"],
    "test": ["pytest", "unittest", "mock", "fixture", "jest", "mocha"],
    "tests": ["pytest", "unittest", "mock", "fixture", "jest"],
    "debugging": ["debug", "pdb", "traceback", "breakpoint", "gdb", "lldb"],
    "debug": ["pdb", "breakpoint", "traceback", "gdb"],
    "database": ["sql", "postgres", "mysql", "sqlite", "migration", "query", "db"],
    "migration": ["migrate", "alembic", "schema", "database"],
    "refactoring": ["refactor", "cleanup", "rename", "extract", "simplify", "restructure"],
    "refactor": ["cleanup", "rename", "extract", "restructure"],
    "performance": ["profile", "benchmark", "optimize", "perf", "slow", "latency"],
    "optimization": ["optimize", "profile", "benchmark", "perf"],
    "profiling": ["profile", "perf", "benchmark", "flamegraph"],
    "security": ["auth", "vulnerability", "sanitize", "encrypt", "token", "jwt", "cve"],
    "vulnerability": ["cve", "security", "patch", "sanitize"],
    "documentation": ["docs", "readme", "sphinx", "mkdocs", "docstring", "javadoc"],
    "docs": ["readme", "sphinx", "mkdocs", "docstring"],
    "networking": ["http", "tcp", "socket", "api", "request", "endpoint", "rest"],
    "network": ["http", "tcp", "socket", "api", "endpoint"],
    "auth": ["login", "token", "jwt", "session", "oauth", "password", "authenticate"],
    "authentication": ["login", "token", "jwt", "oauth", "session"],
    "authorization": ["permission", "rbac", "role", "access"],
    "git": ["commit", "branch", "merge", "rebase", "push", "pull", "checkout"],
    "version control": ["git", "commit", "branch", "merge", "svn"],
    "build": ["compile", "make", "cmake", "webpack", "babel", "tsc"],
    "compilation": ["compile", "gcc", "clang", "make", "cmake"],
    "linting": ["lint", "flake8", "pylint", "eslint", "ruff"],
    "formatting": ["format", "black", "prettier", "gofmt", "rustfmt"],
    "logging": ["log", "logger", "logging", "loguru", "structured log"],
    "monitoring": ["monitor", "prometheus", "grafana", "datadog", "metric"],
    "observability": ["trace", "span", "otel", "opentelemetry", "metric", "log"],
}


class BM25:
    """
    A lightweight, self-contained Python implementation of BM25 ranking.
    """
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lengths) / (self.corpus_size + 1e-9)
        self.doc_freqs = {}
        self.idf = {}
        self.doc_term_freqs = []

        # Initialize frequencies
        for doc in corpus:
            tf = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.doc_term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Calculate IDF
        for term, freq in self.doc_freqs.items():
            # Standard BM25 IDF formula with smoothing to avoid negative values
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0 + 1e-9)

    def get_score(self, doc_index: int, query_terms: List[str]) -> float:
        score = 0.0
        tf = self.doc_term_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index]

        for term in query_terms:
            if term not in tf:
                continue
            f = tf[term]
            idf_val = self.idf.get(term, 0.0)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf_val * (numerator / denominator)
        return score


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into a list of lowercase alphanumeric words.
    """
    return re.findall(r'\w+', text.lower())


def _expand_query(query: str) -> str:
    """
    Expands a (potentially zero-keyword) query with semantically related
    anchor terms before embedding.

    Background
    ----------
    all-MiniLM-L6-v2 is a general-purpose sentence embedding model. When the
    user issues a conceptual query like "containerization" against a corpus
    of literal shell commands (`docker ps -a`) and short commit subjects,
    the cosine similarity between the query embedding and the document
    embedding is often near zero — not because the documents are irrelevant,
    but because the vocabularies don't overlap.

    This function bridges that gap by appending concrete anchor vocabulary
    to the query. The original query is always preserved verbatim at the
    front so the embedding stays anchored to the user's actual phrasing;
    the appended terms just give the model additional signal.

    Scope
    -----
    This expansion is used ONLY for the semantic embedding step. The FTS /
    BM25 retrieval paths still operate on the raw `query`. This keeps the
    lexical scoring deterministic and unaffected by the expansion heuristic.

    Security
    --------
    No LLM is invoked. The expansion map is static and lives in source. The
    returned string is only ever fed to the local sentence-transformers
    model, so there is no prompt-injection surface.
    """
    if not query:
        return query
    query_lower = query.lower()
    tokens = re.findall(r'\w+', query_lower)
    expansions: List[str] = []
    seen = set(tokens)

    # Single-token anchors (e.g. "containerization" -> ["docker", "k8s", ...])
    for token in tokens:
        for anchor in _TOPIC_EXPANSIONS.get(token, []):
            if anchor not in seen:
                expansions.append(anchor)
                seen.add(anchor)

    # Multi-word phrase anchors (e.g. "version control" -> ["git", ...]).
    # Only triggered when the phrase appears verbatim in the query.
    for phrase, anchors in _TOPIC_EXPANSIONS.items():
        if ' ' in phrase and phrase in query_lower:
            for anchor in anchors:
                if anchor not in seen:
                    expansions.append(anchor)
                    seen.add(anchor)

    if not expansions:
        return query
    # Cap the expansion to keep the embedding anchored to the original query.
    return f"{query} (related: {', '.join(expansions[:8])})"


def _preprocess_command_text(command: str) -> str:
    """
    Preprocesses a raw shell command into a more natural-language-like form
    before embedding.

    Why
    ---
    all-MiniLM-L6-v2 is trained on natural-language text. A raw command
    string like `ps -a` produces a near-meaningless embedding because the
    token "ps" carries no semantic signal to the model. Expanding such
    shorthands to natural-language phrases (`ps` -> `process status`)
    gives the model a much stronger signal to match against conceptual
    queries like "list running processes".

    Scope
    -----
    Applied only at embedding time, inside `_build_session_document()`. The
    raw command string is preserved everywhere else (FTS index, BM25
    corpus tokenization is unaffected because `tokenize()` strips non-word
    chars anyway, and `matching_commands` is computed against raw commands).
    """
    if not command:
        return ""
    tokens = command.split()
    expanded: List[str] = []
    for tok in tokens:
        # Strip leading dashes for lookup, but preserve the original token
        # in the output so the raw command form is still recoverable.
        bare = tok.lstrip('-').lower()
        mapped = _COMMAND_VERB_MAP.get(bare)
        if mapped:
            expanded.append(f"{tok} ({mapped})")
        else:
            expanded.append(tok)
    return " ".join(expanded)


def _build_session_document(session: Dict) -> str:
    """
    Builds a natural-language-rich document representation of a session for
    embedding. Commands are preprocessed via `_preprocess_command_text()` to
    expand common shorthands; commit messages are kept as-is because they
    are already natural language. The project name and AI summary are
    included verbatim to give the embedding model the same context a human
    reviewer would have.
    """
    cmd_str = " ".join(
        _preprocess_command_text(c) for c in session.get("all_commands", [])
    )
    commit_str = " ".join(
        c.get("message", "") for c in session.get("all_commits", [])
    )
    return (
        f"Project: {session.get('project_name', '')}\n"
        f"Summary: {session.get('ai_summary') or ''}\n"
        f"Commands: {cmd_str}\n"
        f"Commits: {commit_str}"
    )


def get_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Any:
    """
    Generates sentence embeddings using the sentence-transformers library.

    This function caches loaded model instances by model_name to avoid expensive
    reloading on repeated calls.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "The 'sentence-transformers' package is required for semantic search. "
            "Please install it using: pip install sentence-transformers"
        )
    model = _model_cache.get(model_name)
    if model is None:
        new_model = SentenceTransformer(model_name)
        with _model_cache_lock:
            if model_name not in _model_cache:
                _model_cache[model_name] = new_model
            model = _model_cache[model_name]
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(v1: Any, v2: Any) -> float:
    """
    Computes cosine similarity between two 1D numpy arrays.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


def get_semantic_scores(query: str, documents: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Computes semantic similarity (cosine similarity) of a query against a list of documents.

    The `query` is expected to already be preprocessed (e.g. via `_expand_query()`)
    by the caller if zero-keyword enrichment is desired.
    """
    if not documents:
        return []
    embeddings = get_embeddings([query] + documents, model_name=model_name)
    query_emb = embeddings[0]
    doc_embs = embeddings[1:]

    scores = []
    for doc_emb in doc_embs:
        sim = cosine_similarity(query_emb, doc_emb)
        scores.append(sim)
    return scores


def hybrid_search(
    db,
    query: str,
    project_filter: Optional[str] = None,
    since_ts: Optional[int] = None,
    until_ts: Optional[int] = None,
    tag_filters: Optional[List[str]] = None,
    alpha: float = 0.5,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: Optional[int] = 20,
    semantic_candidate_k: Optional[int] = None,
) -> List[Dict]:
    """
    Performs a hybrid search (BM25 + Cosine Similarity) over terminal sessions.

    Uses FTS-backed candidate retrieval first, but falls back to a bounded
    non-FTS candidate set when no FTS matches are found so semantic
    reranking still works for zero-keyword queries.

    Zero-keyword query handling
    ---------------------------
    A "zero-keyword" query is one whose tokens have NO lexical overlap with
    any candidate document (e.g. querying "containerization" against a
    corpus that only contains `docker ps -a`). The BM25 score for every
    candidate will be 0 in this case. Without special handling, min-max
    normalization would map all candidates to the same BM25 value (0),
    which makes the `(1 - alpha) * bm25` term collapse to a constant —
    effectively turning the hybrid score into pure semantic ranking anyway,
    but with the appearance of a hybrid score.

    We detect this condition explicitly and:
      1. Set the normalized BM25 score to a neutral 0.5 (so the alpha
         blend remains meaningful and the user's `alpha` choice is
         respected).
      2. Mark each result with `zero_keyword=True` so the caller / UI can
         surface the fact that matches are semantic-only.
      3. Use `_expand_query()` to enrich the query with topic anchors
         before embedding, dramatically improving recall for conceptual
         queries against a literal command/commit corpus.

    Candidate pool sizing
    ---------------------
    When FTS misses entirely, the fallback pool size is
    `max(top_k, semantic_candidate_k)` (default `semantic_candidate_k` =
    `max(top_k * 5, 100)`). This gives the semantic ranker enough room to
    surface non-keyword matches without resorting to an unbounded full
    scan, which would be too slow on large histories.

    Semantic-only mode
    ------------------
    When `alpha >= 0.999` the FTS step is skipped entirely and ranking is
    done purely on cosine similarity over the bounded fallback pool. This
    is useful for power users who know their query is conceptual and want
    to bypass FTS entirely.

    Security
    --------
    This function does not invoke any LLM. All embeddings are computed
    locally via sentence-transformers. The returned session dicts are safe
    to feed into LLM-facing code paths because they originate from the
    SQLite database (which is already subject to the existing sanitization
    pipeline in `sanitizer.py`) — no new unsanitized user input is added.

    If sentence-transformers is not installed, raises an ImportError.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "The 'sentence-transformers' package is required for semantic search. "
            "Please install it using: pip install sentence-transformers"
        )

    if not query:
        # Nothing to search for — return an empty list rather than pulling
        # the entire corpus through the embedding pipeline.
        return []

    # Semantic-only mode: skip FTS entirely and rank a bounded candidate
    # pool purely by cosine similarity.
    semantic_only = alpha >= 0.999

    # Default semantic candidate pool: 5x top_k, clamped to a floor of 100
    # so even small top_k values yield a meaningfully-sized reranking pool.
    effective_top_k = top_k if top_k is not None else 20
    if semantic_candidate_k is None:
        semantic_candidate_k = max(effective_top_k * 5, 100)

    # --- Stage 1: FTS-backed candidate retrieval (skipped in semantic-only mode) ---
    from termstory.search import advanced_search
    candidate_sessions: List[Dict] = []
    if not semantic_only:
        candidate_sessions = advanced_search(
            db,
            query=query,
            project_filter=project_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            tag_filters=tag_filters,
            fts=True,
            limit=top_k
        )

    # --- Stage 2: Zero-keyword fallback / augmentation ---
    # If FTS returned no matches (zero-keyword query), OR if we're in
    # semantic-only mode, pull a bounded non-FTS candidate set so semantic
    # similarity has something to rank. The pool is enlarged to
    # `semantic_candidate_k` to give the semantic ranker room to surface
    # non-keyword matches.
    if not candidate_sessions and query:
        candidate_sessions = advanced_search(
            db,
            query=None,
            project_filter=project_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            tag_filters=tag_filters,
            limit=max(effective_top_k, semantic_candidate_k)
        )

    if not candidate_sessions:
        return []

    # --- Stage 3: Document construction with command preprocessing ---
    # Commands are expanded via `_preprocess_command_text()` so the embedding
    # model has natural-language signal to work with. This is critical for
    # zero-keyword recall: `ps -a` alone produces a near-zero signal, but
    # `ps (process status) -a (all)` matches a query like "list processes"
    # much more reliably.
    documents = [_build_session_document(s) for s in candidate_sessions]

    # --- Stage 4: BM25 (lexical) scoring ---
    # BM25 still operates on the raw query (NOT the expanded query) so the
    # lexical signal remains deterministic and unaffected by the topic
    # expansion heuristic.
    tokenized_corpus = [tokenize(doc) for doc in documents]
    query_terms = tokenize(query)
    bm25 = BM25(tokenized_corpus)
    bm25_scores = [bm25.get_score(i, query_terms) for i in range(len(candidate_sessions))]

    # Detect the zero-keyword condition: every BM25 score is zero, meaning
    # no query token appears in any candidate document.
    zero_keyword = bool(bm25_scores) and all(s <= 0.0 for s in bm25_scores)
    if not bm25_scores:
        zero_keyword = True

    # --- Stage 5: Semantic scoring with query expansion ---
    # The expanded query is used ONLY for embedding, never for BM25. This
    # is what makes zero-keyword queries work: "containerization" gets
    # enriched with "docker, kubernetes, k8s, compose, pod" before encoding,
    # so the cosine similarity against a `docker ps -a` session is high.
    expanded_query = _expand_query(query)
    semantic_scores = get_semantic_scores(expanded_query, documents, model_name=model_name)

    # --- Stage 6: Min-Max normalization ---
    # For zero-keyword queries, BM25 is uninformative (every score is 0).
    # Rather than letting min-max collapse to 0 for everyone (which would
    # make the alpha blend meaningless), we set normalized BM25 to a
    # neutral 0.5 so the semantic signal dominates as intended by `alpha`
    # without artificially penalizing every candidate.
    if zero_keyword:
        normalized_bm25 = [0.5] * len(candidate_sessions)
    else:
        min_bm25 = min(bm25_scores) if bm25_scores else 0.0
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
        bm25_range = max_bm25 - min_bm25
        if bm25_range == 0.0:
            bm25_range = 1e-9
        normalized_bm25 = [(score - min_bm25) / bm25_range for score in bm25_scores]

    min_sem = min(semantic_scores) if semantic_scores else 0.0
    max_sem = max(semantic_scores) if semantic_scores else 0.0
    sem_range = max_sem - min_sem
    if sem_range == 0.0:
        sem_range = 1e-9
    normalized_sem = [(score - min_sem) / sem_range for score in semantic_scores]

    # --- Stage 7: Combine scores and populate result metadata ---
    scored_sessions = []
    for i, s in enumerate(candidate_sessions):
        h_score = alpha * normalized_sem[i] + (1.0 - alpha) * normalized_bm25[i]

        # matching_commands / matching_commits are computed against the
        # RAW query terms (not the expanded query). For zero-keyword
        # queries these lists will be empty, which is the correct
        # behavior: we surface the session via semantic similarity alone,
        # not via fake lexical matches against expansion anchors.
        matching_cmds = []
        for cmd in s.get("all_commands", []):
            if any(term in cmd.lower() for term in query_terms):
                matching_cmds.append(cmd)

        matching_commits = []
        for commit in s.get("all_commits", []):
            msg = commit.get("message", "").lower()
            if any(term in msg for term in query_terms):
                matching_commits.append(commit)

        s["matching_commands"] = matching_cmds
        s["matching_commits"] = matching_commits
        s["hybrid_score"] = h_score
        # Expose the raw sub-scores and zero-keyword flag so callers
        # (CLI, TUI, future LLM-facing code) can reason about why a
        # session was surfaced.
        s["semantic_score"] = float(semantic_scores[i]) if i < len(semantic_scores) else 0.0
        s["bm25_score"] = float(bm25_scores[i]) if i < len(bm25_scores) else 0.0
        s["zero_keyword"] = zero_keyword
        scored_sessions.append(s)

    # Sort sessions by hybrid score descending
    scored_sessions.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # Truncate the ranked pool back down to the requested top_k. The
    # fallback / semantic-only branches above intentionally pull a larger
    # candidate pool (`max(effective_top_k, semantic_candidate_k)`, floor
    # 100) so the semantic ranker has room to surface non-keyword matches
    # — but the *return* contract of this function is "at most top_k
    # sessions". Without this final slice, callers that don't re-truncate
    # (anything other than the CLI's `--limit` path) would receive up to
    # 100 sessions for zero-keyword / semantic-only queries regardless of
    # the `top_k` argument.
    return scored_sessions[:effective_top_k]
