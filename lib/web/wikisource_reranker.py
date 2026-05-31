"""
Wikisource-based anchor reranker.

Purpose:
- validate translated keywords against a Han-text corpus,
- reward keywords that co-occur with other translated candidates,
- penalize keywords that are too isolated or too generic.

Auth design:
- use normal env variables or .env via lib.config,
- if WIKISOURCE_BOT_USERNAME and WIKISOURCE_BOT_PASSWORD exist, login once,
- otherwise fallback to anonymous requests.
"""

import time
import requests
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any, Set

from lib.config import load_project_env, get_env


DEFAULT_WIKISOURCE_API = "https://zh.wikisource.org/w/api.php"
DEFAULT_USER_AGENT = "MyThesisBot/1.0 (research use)"

WIKISOURCE_API = DEFAULT_WIKISOURCE_API
HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
}

# One global session is enough. If bot credentials exist, it becomes logged-in.
WIKISOURCE_SESSION = requests.Session()
WIKISOURCE_AUTH_INITIALIZED = False
WIKISOURCE_LOGGED_IN = False
WIKISOURCE_USERNAME: Optional[str] = None

WIKISOURCE_REQUEST_COUNT = 0
WIKISOURCE_CACHE_HIT_COUNT = 0
WIKISOURCE_429_COUNT = 0


def init_wikisource_session(
    verbose: bool = False,
    timeout: int = 10,
    api_url: Optional[str] = None,
) -> bool:
    """
    Initialize the MediaWiki HTTP session used for co-occurrence reranking.

    Despite the historical "wikisource" naming, this is a generic MediaWiki
    search client. The endpoint is chosen per translation direction:
        - vi->zh : https://zh.wikisource.org/w/api.php (classical Han corpus)
        - zh->vi : https://vi.wikipedia.org/w/api.php (rich modern VI corpus)

    Endpoint precedence: api_url arg > WIKISOURCE_API env > default (zh.wikisource).

    Env variables loaded from shell or .env through lib.config:
        WIKISOURCE_API
        WIKISOURCE_USER_AGENT
        WIKISOURCE_BOT_USERNAME
        WIKISOURCE_BOT_PASSWORD

    Returns:
        True if logged in with bot credentials, False if anonymous.
    """
    global WIKISOURCE_API, WIKISOURCE_AUTH_INITIALIZED
    global WIKISOURCE_LOGGED_IN, WIKISOURCE_USERNAME

    if WIKISOURCE_AUTH_INITIALIZED:
        return WIKISOURCE_LOGGED_IN

    load_project_env()

    WIKISOURCE_API = (
        api_url
        or get_env("WIKISOURCE_API", DEFAULT_WIKISOURCE_API)
        or DEFAULT_WIKISOURCE_API
    )
    user_agent = get_env("WIKISOURCE_USER_AGENT", DEFAULT_USER_AGENT) or DEFAULT_USER_AGENT

    HEADERS["User-Agent"] = user_agent
    WIKISOURCE_SESSION.headers.update(HEADERS)

    username = get_env("WIKISOURCE_BOT_USERNAME")
    password = get_env("WIKISOURCE_BOT_PASSWORD")

    WIKISOURCE_AUTH_INITIALIZED = True

    if not username or not password:
        if verbose:
            print("[Wikisource auth] no bot credentials found; using anonymous session")
        return False

    try:
        # 1) Get login token.
        token_resp = WIKISOURCE_SESSION.get(
            WIKISOURCE_API,
            params={
                "action": "query",
                "meta": "tokens",
                "type": "login",
                "format": "json",
            },
            timeout=timeout,
        )
        token_resp.raise_for_status()
        login_token = token_resp.json()["query"]["tokens"]["logintoken"]

        # 2) Login with BotPassword.
        login_resp = WIKISOURCE_SESSION.post(
            WIKISOURCE_API,
            data={
                "action": "login",
                "lgname": username,
                "lgpassword": password,
                "lgtoken": login_token,
                "format": "json",
            },
            timeout=timeout,
        )
        login_resp.raise_for_status()
        login_data = login_resp.json().get("login", {})

        if login_data.get("result") == "Success":
            WIKISOURCE_LOGGED_IN = True
            WIKISOURCE_USERNAME = username
            if verbose:
                print(f"[Wikisource auth] logged in as {username}")
        else:
            WIKISOURCE_LOGGED_IN = False
            WIKISOURCE_USERNAME = None
            if verbose:
                reason = login_data.get("reason") or login_data.get("result")
                print(f"[Wikisource auth] login failed: {reason}; using anonymous session")

    except Exception as e:
        WIKISOURCE_LOGGED_IN = False
        WIKISOURCE_USERNAME = None
        if verbose:
            print(f"[Wikisource auth] login error: {e}; using anonymous session")

    return WIKISOURCE_LOGGED_IN


def get_wikisource_auth_status() -> Dict[str, Any]:
    return {
        "logged_in": WIKISOURCE_LOGGED_IN,
        "username": WIKISOURCE_USERNAME,
        "api": WIKISOURCE_API,
        "user_agent": HEADERS.get("User-Agent"),
    }


def get_wikisource_stats() -> Dict[str, int]:
    return {
        "requests": WIKISOURCE_REQUEST_COUNT,
        "cache_hits": WIKISOURCE_CACHE_HIT_COUNT,
        "rate_limits": WIKISOURCE_429_COUNT,
    }


def reset_wikisource_stats():
    global WIKISOURCE_REQUEST_COUNT, WIKISOURCE_CACHE_HIT_COUNT, WIKISOURCE_429_COUNT
    WIKISOURCE_REQUEST_COUNT = 0
    WIKISOURCE_CACHE_HIT_COUNT = 0
    WIKISOURCE_429_COUNT = 0


class WikisourceRateLimitError(RuntimeError):
    pass


def pair_hit_quality(hits: int) -> float:
    """
    Heuristic score for pair co-occurrence quality.

    Interpretation:
    - 0 hits      : unsupported
    - 1-3 hits    : highly specific, strongest signal
    - 4-10 hits   : strong
    - 11-30 hits  : moderate
    - 31-100 hits : weak, becoming generic
    - >100 hits   : very generic
    """
    if hits <= 0:
        return 0.0
    elif hits <= 3:
        return 1.0
    elif hits <= 10:
        return 0.8
    elif hits <= 30:
        return 0.5
    elif hits <= 100:
        return 0.25
    else:
        return 0.1


def search_pair_hits_wikisource(
    word1: str,
    word2: str,
    cache: Optional[Dict[Tuple[str, str], int]] = None,
    sleep_sec: float = 0.3,
    timeout: int = 10,
) -> int:
    """
    Search co-occurrence count of two Han terms in Wikisource.

    Notes:
        - Cache normal pair results, including hits=0.
        - Do NOT cache 429 responses because rate limiting is temporary.
        - Raise WikisourceRateLimitError on 429 so the caller can disable
          Wikisource reranking for the rest of the run.
        - Count actual HTTP requests, excluding cache hits.
    """
    global WIKISOURCE_REQUEST_COUNT, WIKISOURCE_CACHE_HIT_COUNT, WIKISOURCE_429_COUNT

    # Login once if bot env exists; otherwise this stays anonymous.
    init_wikisource_session(verbose=False, timeout=timeout)

    key = tuple(sorted([word1, word2]))

    if cache is not None and key in cache:
        WIKISOURCE_CACHE_HIT_COUNT += 1
        return cache[key]

    params = {
        "action": "query",
        "list": "search",
        "srsearch": f'"{word1}" "{word2}"',
        "format": "json",
        "srlimit": 1,
    }

    hits = 0

    try:
        WIKISOURCE_REQUEST_COUNT += 1
        request_no = WIKISOURCE_REQUEST_COUNT

        resp = WIKISOURCE_SESSION.get(
            WIKISOURCE_API,
            params=params,
            timeout=timeout,
        )

        if resp.status_code == 429:
            WIKISOURCE_429_COUNT += 1
            retry_after = resp.headers.get("Retry-After")

            print(
                f"[Wikisource 429] request_no={request_no}, "
                f"rate_limit_no={WIKISOURCE_429_COUNT}, "
                f"cache_hits={WIKISOURCE_CACHE_HIT_COUNT}, "
                f"pair=({word1}, {word2}), "
                f"Retry-After={retry_after}"
            )

            raise WikisourceRateLimitError(
                f"429 Too Many Requests at request_no={request_no}; "
                f"pair=({word1}, {word2}); "
                f"Retry-After={retry_after}"
            )

        resp.raise_for_status()

        data = resp.json()
        hits = data.get("query", {}).get("searchinfo", {}).get("totalhits", 0)

    except WikisourceRateLimitError:
        raise

    except Exception as e:
        print(
            f"[Wikisource API error] "
            f"request_no={WIKISOURCE_REQUEST_COUNT}, "
            f"pair=({word1}, {word2}): {e}"
        )
        hits = 0

    if cache is not None:
        cache[key] = hits

    if sleep_sec > 0:
        time.sleep(sleep_sec)

    return hits


def filter_candidates(
    keywords: List[Dict[str, Any]],
    trans_field: str = "trans",
    min_han_len: int = 2,
    allowed_pos: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build two candidate pools:

    1. all_candidates:
        All translated candidates. These are never discarded only because of
        POS or length constraints.

    2. rerank_candidates:
        Candidates eligible for Wikisource co-occurrence reranking.
        They must pass:
            - min_han_len
            - allowed_pos, if provided

    Non-rerankable candidates can still be used later as semantic-only fallback.
    """
    all_candidates = []
    rerank_candidates = []

    for item in keywords:
        word = item.get("word", "")
        han_word = item.get(trans_field, "") or item.get("han_word", "")
        han_word = str(han_word).strip()
        score = float(item.get("score", 0.0))
        pos = item.get("pos", "")

        if not han_word:
            continue

        cand = {
            "word": word,
            "han_word": han_word,
            "score": score,
            "pos": pos,
        }

        all_candidates.append(cand)

        valid_for_rerank = True

        if allowed_pos is not None:
            pos_tags = set(str(pos).split())
            if not pos_tags & allowed_pos:
                valid_for_rerank = False

        if len(han_word) < min_han_len:
            valid_for_rerank = False

        if valid_for_rerank:
            rerank_candidates.append(cand)

    return all_candidates, rerank_candidates


def score_anchor_candidates(
    keywords: List[Dict[str, Any]],
    trans_field: str = "trans",
    min_han_len: int = 2,
    allowed_pos: Optional[Set[str]] = None,
    min_pair_hit: int = 1,
    top_k: Optional[int] = None,
    sleep_sec: float = 0.3,
    semantic_weight: float = 0.10,
    pair_count_weight: float = 0.60,
    pair_quality_weight: float = 0.30,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], int]]:
    """
    Rerank translated keywords by Wikisource co-occurrence.

    Important behavior:
    - All translated candidates are kept.
    - Only candidates passing min_han_len + allowed_pos are eligible for Wikisource rerank.
    - If top_k is set, only the top_k eligible candidates by original score are sent to Wikisource.
    - Candidates not sent to Wikisource remain as semantic-only fallback.
    """
    all_candidates, rerank_candidates = filter_candidates(
        keywords,
        trans_field=trans_field,
        min_han_len=min_han_len,
        allowed_pos=allowed_pos,
    )

    if not all_candidates:
        return [], {}

    # Use all translated candidates for semantic normalization.
    max_sem = max(x["score"] for x in all_candidates) or 1.0

    # Initialize all candidates as semantic-only fallback.
    for cand in all_candidates:
        semantic_norm = cand["score"] / max_sem

        cand["support_count"] = 0
        cand["total_pair_hits"] = 0
        cand["max_pair_hits"] = 0
        cand["pair_hits_list"] = []
        cand["pair_quality_sum"] = 0.0
        cand["connected_words"] = []
        cand["anchor_score"] = semantic_weight * semantic_norm
        cand["rerank_source"] = "semantic_only_fallback"

    # IMPORTANT:
    # Cut eligible rerank candidates BEFORE calling Wikisource.
    # This controls the number of API requests:
    # top_k=4 -> C(4,2)=6 requests per query group.
    if top_k is not None:
        rerank_candidates = sorted(
            rerank_candidates,
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

    pair_matrix: Dict[Tuple[str, str], int] = {}

    if len(rerank_candidates) >= 2:
        cache: Dict[Tuple[str, str], int] = {}

        for cand in rerank_candidates:
            cand["rerank_source"] = "wikisource_bot" if WIKISOURCE_LOGGED_IN else "wikisource"

        for cand_a, cand_b in combinations(rerank_candidates, 2):
            han_a = cand_a["han_word"]
            han_b = cand_b["han_word"]

            hits = search_pair_hits_wikisource(
                han_a,
                han_b,
                cache=cache,
                sleep_sec=sleep_sec,
            )
            quality = pair_hit_quality(hits)

            pair_matrix[(han_a, han_b)] = hits
            pair_matrix[(han_b, han_a)] = hits

            for current, other in [(cand_a, cand_b), (cand_b, cand_a)]:
                current["pair_hits_list"].append(hits)
                current["total_pair_hits"] += hits
                current["max_pair_hits"] = max(current["max_pair_hits"], hits)
                current["pair_quality_sum"] += quality

                if hits >= min_pair_hit:
                    current["support_count"] += 1
                    current["connected_words"].append({
                        "word": other["word"],
                        "han_word": other["han_word"],
                        "hits": hits,
                        "quality": quality,
                    })

        for cand in rerank_candidates:
            cand["connected_words"] = sorted(
                cand["connected_words"],
                key=lambda x: (x["quality"], x["hits"]),
                reverse=True,
            )

        max_support = max(x["support_count"] for x in rerank_candidates) or 1
        max_quality = max(x["pair_quality_sum"] for x in rerank_candidates) or 1.0

        for item in rerank_candidates:
            semantic_norm = item["score"] / max_sem
            support_norm = item["support_count"] / max_support
            quality_norm = item["pair_quality_sum"] / max_quality

            item["anchor_score"] = (
                semantic_weight * semantic_norm
                + pair_count_weight * support_norm
                + pair_quality_weight * quality_norm
            )

    ranked_candidates = sorted(
        all_candidates,
        key=lambda x: (
            x["anchor_score"],
            x["score"],
        ),
        reverse=True,
    )

    return ranked_candidates, pair_matrix


def select_top_anchors(
    keywords: List[Dict[str, Any]],
    top_n: int = 3,
    trans_field: str = "trans",
    min_han_len: int = 2,
    allowed_pos: Optional[Set[str]] = None,
    min_pair_hit: int = 1,
    top_k_candidates: int = 8,
    sleep_sec: float = 0.3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, str], int]]:
    """
    Convenience wrapper:
    returns (top_anchors, ranked_candidates, pair_matrix)
    """
    ranked_candidates, pair_matrix = score_anchor_candidates(
        keywords=keywords,
        trans_field=trans_field,
        min_han_len=min_han_len,
        allowed_pos=allowed_pos,
        min_pair_hit=min_pair_hit,
        top_k=top_k_candidates,
        sleep_sec=sleep_sec,
    )

    return ranked_candidates[:top_n], ranked_candidates, pair_matrix
