# This file was generated using Claude Code

"""Scientific Contribution Graph Explorer — single-file FastAPI demo.

A browser-based UI for the Scientific Contribution Graph:
  1) Live-search papers by title.
  2) Pick a paper; the contributions appear immediately, with downstream
     impact metrics filling in as they finish computing.
  3) Pick a contribution; the forward/backward crawl renders in the main
     panel, with a sticky sidebar of live knobs (direction, layout, depth,
     strong-only, min-children). Adjusting a knob re-renders live.

The page is a single HTML/CSS/JS document served by FastAPI — no Gradio,
no React, no build step. The visualization is an inline SVG with pan/zoom
via the (CDN-loaded) `svg-pan-zoom` library.

CONFIGURATION
=============

The same `demo.py` runs locally and on Hugging Face Spaces. Settings are
resolved with precedence: CLI args > env vars > config file > defaults.

  1. **Config file** — `demo/demo_config.json` next to this script (or
     pass `--config /path/to/config.json`). See `demo_config.example.json`
     for the schema. Two modes:

       a) Local: set `"data_path"` to an unpacked release directory.
       b) HF Spaces: set `"bucket_uri"` to a `hf://buckets/...` URI; the
          dataset will be synced + extracted on first boot into
          `data_path` (default `./scg-release`).

  2. **Env vars** — `SCG_DATA_PATH`, `SCG_BUCKET_URI`,
     `SCG_SERVER_NAME`, `SCG_SERVER_PORT`.

  3. **CLI** — `--data-path`, `--bucket-uri`, `--host`, `--port`.

Run with:
    python demo/demo.py
or:
    python demo/demo.py --data-path /path/to/scg-release-root --port 7860
or:
    python demo/demo.py --bucket-uri hf://buckets/.../releases-tar/current
"""

import argparse
import asyncio
import html as htmllib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from queue import Empty as QueueEmpty
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response

from scicontgraph import ScientificContributionGraph
from scicontgraph.ScientificContributionGraphVisualization import (
    convert_crawl_results_to_dot,
    convert_crawl_results_to_dot_with_edge_nodes,
    export_crawl_results_to_radial_tree_svg,
)


DEFAULT_DATA_PATH = "/data-ssd2/scientific-contribution-graph/hf-release/releases/1.0/"
DEFAULT_BUCKET_URI = "hf://buckets/pajansen/scientific-contribution-graph/releases-tar/current"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "demo_config.json")

# UI-side timeouts (seconds). Defaults are tuned for a local workstation;
# on HF Spaces' slower vCPUs, override via demo_config.json (or env vars
# SCG_IMPACT_TIMEOUT_SECS / SCG_RENDER_TIMEOUT_SECS).
DEFAULT_IMPACT_TIMEOUT_SECS = 10
DEFAULT_RENDER_TIMEOUT_SECS = 10

# Globals initialized at startup.
GRAPH: Optional[ScientificContributionGraph] = None
WORK_DIR: Optional[str] = None
TITLE_INDEX: List[Tuple[str, str, str]] = []
GRAPH_STATS: Dict[str, int] = {}

# Effective settings — populated in main(); used by /admin/refresh to find
# the same data_path / bucket_uri the server was launched with.
SETTINGS: Dict[str, Any] = {}

# /admin/refresh job state.
REFRESH_STATE: Dict[str, Any] = {
    "status": "idle",           # idle | syncing | extracting | reloading | done | error
    "started_at": None,
    "finished_at": None,
    "message": "",
    "release": None,            # release marker after a successful refresh
}
REFRESH_LOCK = threading.Lock()

# Tunables.
IMPACT_MAX_DEPTH = 3            # Depth used for impact metric (slowest step).
AUTO_TRIM_TARGET_NODES = 30     # Auto-trim a crawl down to ~this many nodes.

# In-memory cache for impact metrics. Computing the metric on a popular
# paper takes a few seconds, so we keep results around for the session.
IMPACT_CACHE: Dict[str, dict] = {}

# Papers whose impact-metric computation timed out in the UI. Subsequent UI
# requests skip recomputation and immediately return "too large" — we don't
# want to repeatedly burn CPU on the same hopeless case. The mapping value
# is the timeout that was in effect at the time. (API callers without a
# timeout always bypass this cache and re-attempt the real computation.)
IMPACT_TIMED_OUT: Dict[str, float] = {}

# Same caches but keyed at the per-contribution level, used by the new
# /api/contribution/{cid}/impact endpoint that the UI hits one at a time
# so impact scores populate progressively rather than all-or-nothing.
CONTRIB_IMPACT_CACHE: Dict[str, dict] = {}
CONTRIB_IMPACT_TIMED_OUT: Dict[str, float] = {}


# ----------------------------------------------------------------------
# Title-index search
# ----------------------------------------------------------------------
def build_title_index(graph: ScientificContributionGraph) -> List[Tuple[str, str, str]]:
    index = []
    for title, corpus_id in graph.paper_title_to_corpus_id.items():
        if not title:
            continue
        index.append((title.lower(), title, str(corpus_id)))
    return index


# Graph stats are static per release. We hard-code the values for v1.0
# (verified by a full on-disk sweep) rather than recompute on every boot,
# which would burn ~2 minutes scanning 30 GB of paper JSON. If a future
# release ships a `<data_path>/data/metadata/demo_stats.json` file with
# `{"n_papers", "n_contributions", "n_edges"}`, we use those instead.
DEFAULT_GRAPH_STATS: Dict[str, int] = {
    "n_papers":        230_454,
    "n_contributions": 2_047_426,
    "n_edges":         12_524_458,
}


def compute_graph_stats(graph: ScientificContributionGraph,
                        data_path: Optional[str] = None) -> Dict[str, int]:
    """Return the documented totals. Prefer a sidecar override file at
    `<data_path>/data/metadata/demo_stats.json` if present."""
    if data_path:
        override = os.path.join(data_path, "data", "metadata", "demo_stats.json")
        if os.path.exists(override):
            try:
                import json as _json
                with open(override, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                return {
                    "n_papers":        int(data["n_papers"]),
                    "n_contributions": int(data["n_contributions"]),
                    "n_edges":         int(data["n_edges"]),
                }
            except Exception:
                traceback.print_exc()
    return dict(DEFAULT_GRAPH_STATS)


def live_search_titles(query: str, max_results: int = 25) -> List[Dict[str, str]]:
    """Substring + token-coverage matching over all paper titles."""
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []

    starts_with: List[Tuple[str, str]] = []
    word_start: List[Tuple[str, str]] = []
    contains: List[Tuple[str, str]] = []
    space_q = " " + q
    for lowered, original, corpus_id in TITLE_INDEX:
        if lowered.startswith(q):
            starts_with.append((original, corpus_id))
        elif space_q in lowered:
            word_start.append((original, corpus_id))
        elif q in lowered:
            contains.append((original, corpus_id))
        if (len(starts_with) + len(word_start) + len(contains)) >= max_results * 3:
            break
    ranked = starts_with + word_start + contains

    # Token-coverage fallback: titles that contain ALL the query tokens.
    if len(ranked) < max_results:
        tokens = [t for t in q.split() if len(t) >= 2]
        seen = {cid for _, cid in ranked}
        if len(tokens) >= 2:
            for lowered, original, corpus_id in TITLE_INDEX:
                if corpus_id in seen:
                    continue
                if all(t in lowered for t in tokens):
                    ranked.append((original, corpus_id))
                    if len(ranked) >= max_results:
                        break

    ranked = ranked[:max_results]
    return [{"title": t, "corpus_id": c} for t, c in ranked]


# ----------------------------------------------------------------------
# Auto-trim crawls so the first render is small enough to be readable.
# ----------------------------------------------------------------------
def _build_parent_to_children(crawl_results, direction):
    edges = crawl_results.get("edges", []) or []
    p2c = defaultdict(set)
    for edge in edges:
        source = edge.get("contribution_id")
        target = edge.get("used_by_contribution_id") or edge.get("prerequisite_for_contribution_id")
        if not source or not target:
            continue
        if direction == "forward":
            p2c[source].add(target)
        else:
            p2c[target].add(source)
    return p2c


def _trim_crawl_by_min_children(crawl_results, direction, min_c):
    if min_c <= 0:
        return crawl_results
    nodes = crawl_results.get("nodes", {}) or {}
    edges = crawl_results.get("edges", []) or []
    root = crawl_results.get("root_node")
    p2c = _build_parent_to_children(crawl_results, direction)

    keep = set()
    if root is not None:
        keep.add(root)
    for nid in nodes:
        if nid == root:
            continue
        if len(p2c.get(nid, set())) >= min_c:
            keep.add(nid)

    new_nodes = {nid: nodes[nid] for nid in keep if nid in nodes}
    new_edges = []
    for edge in edges:
        s = edge.get("contribution_id")
        t = edge.get("used_by_contribution_id") or edge.get("prerequisite_for_contribution_id")
        if s in keep and t in keep:
            new_edges.append(edge)
    return {**crawl_results, "nodes": new_nodes, "edges": new_edges}


def _auto_trim_to_target(crawl_results, direction, target=AUTO_TRIM_TARGET_NODES,
                         user_min_children=0):
    nodes = crawl_results.get("nodes", {}) or {}
    final_mc = max(int(user_min_children or 0), 0)

    if final_mc > 0:
        crawl_results = _trim_crawl_by_min_children(crawl_results, direction, final_mc)
        if len(crawl_results.get("nodes", {})) <= target:
            return crawl_results, final_mc

    if len(crawl_results.get("nodes", {})) <= target:
        return crawl_results, final_mc

    for min_c in range(max(final_mc, 0) + 1, 50):
        trimmed = _trim_crawl_by_min_children(crawl_results, direction, min_c)
        if len(trimmed.get("nodes", {})) <= target:
            return trimmed, min_c
    return crawl_results, final_mc


# ----------------------------------------------------------------------
# Visualization render helpers
# ----------------------------------------------------------------------
def _safe_name(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in text)


def _clear_prior_renders(prefix: str):
    base = os.path.basename(prefix)
    dir_path = os.path.dirname(prefix) or "."
    for filename in os.listdir(dir_path):
        if filename.startswith(base + ".") or filename.startswith(base + "-"):
            try:
                os.remove(os.path.join(dir_path, filename))
            except OSError:
                pass


def _flip_dot_arrows(dot_path: str, direction: str, layout: str) -> None:
    """Post-process a DOT file in place to flip arrow head positions.

    Background: the upstream `convert_crawl_results_to_dot*` helpers
    point arrows in the *citation* direction — newer-citing-older — by
    using `dir=back` on forward-crawl edges so the arrowhead lands on
    the prerequisite (the visual parent at the top, for forward layouts).
    Users of this demo expect the *causal* direction instead: arrows
    pointing from older toward newer, i.e. "A enabled B" rather than
    "B cites A". Since we're not allowed to modify the upstream viz
    code, we rewrite the DOT output after it's emitted but before we
    run `dot -Tsvg`, swapping the small set of attribute literals the
    helpers use:

      tree layout
        forward  → strip `[dir=back]` (arrowhead moves to the child)
        backward → add  `[dir=back]` (arrowhead moves to the parent)

      tree-with-edges layout
        forward  output uses `[dir=back, weight=3]` and `[dir=none, weight=3]`
        backward output uses `[dir=none, weight=3]` and `[weight=3]`
        We swap the two output families so each direction renders with
        the arrowheads the *other* direction was previously using —
        which is exactly the user's "flip them" request.

    Has no effect on node/graph attributes (they don't contain `dir=`
    or `weight=3` literals)."""
    if layout not in ("tree", "tree-with-edges"):
        return
    if direction not in ("forward", "backward"):
        return
    try:
        with open(dot_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return

    if layout == "tree":
        if direction == "forward":
            text = text.replace(" [dir=back];", ";")
        else:
            text = re.sub(
                r'^(    "[^"]+" -> "[^"]+");$',
                r"\1 [dir=back];",
                text, flags=re.MULTILINE,
            )
    else:  # tree-with-edges
        SENT_BACK  = "\x00__SCG_BACK__"
        SENT_NONE  = "\x00__SCG_NONE__"
        SENT_PLAIN = "\x00__SCG_PLAIN__"
        text = text.replace("[dir=back, weight=3]", SENT_BACK)
        text = text.replace("[dir=none, weight=3]", SENT_NONE)
        text = text.replace("[weight=3]",            SENT_PLAIN)
        if direction == "forward":
            text = text.replace(SENT_BACK,  "[dir=none, weight=3]")
            text = text.replace(SENT_NONE,  "[weight=3]")
            text = text.replace(SENT_PLAIN, "[weight=3]")
        else:
            text = text.replace(SENT_NONE,  "[dir=back, weight=3]")
            text = text.replace(SENT_PLAIN, "[dir=none, weight=3]")
            text = text.replace(SENT_BACK,  "[dir=back, weight=3]")

    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(text)


def render_crawl(
    contribution_id: str,
    direction: str,
    layout: str,
    max_depth: int,
    only_strong: bool,
    min_children: int,
    auto_trim: bool = True,
    layout_deadline_epoch: Optional[float] = None,
) -> Dict:
    """Crawl + render. Returns {svg, info, applied_min_children, ...}.

    `auto_trim` controls whether the backend will *raise* `min_children`
    beyond the user's value to keep the first render readable. The frontend
    sets it to True on the first render of a freshly-selected contribution
    and False thereafter — so once the user touches a knob, their values
    are respected exactly.

    `layout_deadline_epoch` (if set) is a wall-time deadline for the
    radial layout's contraction loop. When the loop would overrun, we
    stop it early and write the SVG with the partially-converged node
    positions. Other layouts ignore this. Used by the UI-mode wrapper
    so big graphs degrade gracefully instead of timing out wholesale.
    """
    print(f"[render] contribution={contribution_id} dir={direction} "
          f"layout={layout} depth={max_depth} strong={only_strong} "
          f"user_min_children={min_children} auto_trim={auto_trim}", flush=True)

    t0 = time.time()
    if direction == "forward":
        crawl_results = GRAPH.crawl_forwards_from_contribution(
            contribution_id=contribution_id, max_depth=int(max_depth),
            only_strong_connections=bool(only_strong), verbose_progress=False,
        )
    else:
        crawl_results = GRAPH.crawl_backwards_from_contribution(
            contribution_id=contribution_id, max_depth=int(max_depth),
            only_strong_connections=bool(only_strong), verbose_progress=False,
        )
    crawl_secs = time.time() - t0

    raw_nodes = len(crawl_results.get("nodes", {}) or {})
    raw_edges = len(crawl_results.get("edges", []) or [])

    user_mc = int(min_children or 0)
    if auto_trim:
        crawl_results, applied_mc = _auto_trim_to_target(
            crawl_results, direction,
            target=AUTO_TRIM_TARGET_NODES,
            user_min_children=user_mc,
        )
    else:
        # Honor the user's min_children exactly — no further trimming.
        crawl_results = _trim_crawl_by_min_children(crawl_results, direction, user_mc)
        applied_mc = user_mc
    num_nodes = len(crawl_results.get("nodes", {}) or {})
    num_edges = len(crawl_results.get("edges", []) or [])

    if num_nodes <= 1 or num_edges == 0:
        return {
            "svg": "",
            "num_nodes": num_nodes, "num_edges": num_edges,
            "raw_nodes": raw_nodes, "raw_edges": raw_edges,
            "applied_min_children": applied_mc,
            "crawl_secs": crawl_secs, "render_secs": 0.0,
            "empty_reason": (
                "No connected nodes at these settings. Try increasing depth, "
                "disabling “strong connections only”, or lowering “min children”."
            ),
        }

    prefix = os.path.join(WORK_DIR, _safe_name(f"crawl_{direction}_{contribution_id}"))
    _clear_prior_renders(prefix)
    svg_path = prefix + ".svg"

    t0 = time.time()
    partial_layout = False
    if layout == "radial":
        # Tuning for the force-directed contraction loop. Each iteration is
        # O(N²) for node-pair repulsion plus an expensive crossing-count
        # acceptance check (up to 4 retries per node per iteration). The
        # combination of half the iterations + larger per-step distance
        # gives roughly the same total "displacement budget", and the
        # less-frequent crossing check skips most of the per-node retry
        # work. End result: ~2-3× faster on slower vCPUs (e.g., HF Spaces),
        # with similar final layout quality on the kinds of graphs this
        # demo renders (≤30 nodes after auto-trim).
        #
        # Additionally: when a `layout_deadline_epoch` is set, we
        # monkey-patch `tqdm.tqdm` for the duration of this call so the
        # contraction loop *stops early* on overrun rather than running
        # all 200 iterations. The SVG-write step then completes with
        # whatever positions the loop reached — so users get a
        # partially-converged (but complete) graph instead of a timeout
        # error. Restored in `finally:` so this side-effect doesn't leak
        # to other code in the same process.
        budget_hit_ref = [False]
        tqdm_mod = sys.modules.get("tqdm")
        real_tqdm = getattr(tqdm_mod, "tqdm", None) if tqdm_mod else None
        patch_applied = False
        if (layout_deadline_epoch is not None
                and tqdm_mod is not None and real_tqdm is not None):
            def _budgeted_tqdm(iterable, *_a, **_kw):
                # We deliberately drop the kw args — we don't need the
                # progress bar, just a budget-aware iterator.
                for item in iterable:
                    if time.time() > layout_deadline_epoch:
                        budget_hit_ref[0] = True
                        return
                    yield item
            tqdm_mod.tqdm = _budgeted_tqdm
            patch_applied = True
        try:
            export_crawl_results_to_radial_tree_svg(
                crawl_results, svg_path,
                node_diameter_px=120, min_gap_px=30, radial_step_px=150, margin_px=40,
                contraction_iterations=200,
                gravity_k=4 * 0.055, spring_k=2 * 0.020,
                node_repulsion_k=1.5 * 18.0, edge_repulsion_k=1.5 * 10.0,
                max_move_px=28.0, node_edge_gap_px=30.0, node_center_min_distance_px=150.0,
                crossing_check_every_n_iterations=3,
                metadata_line_mode="paper_title",
                min_children=0, min_thresh=0,
                root_fill_color="#CCCCCC",
                trim_root_leaf_summary_nodes=False,
                crawl_direction=direction,
            )
        finally:
            if patch_applied:
                tqdm_mod.tqdm = real_tqdm
        partial_layout = budget_hit_ref[0]
        if partial_layout:
            print(f"[render] radial contraction stopped early (deadline hit)",
                  flush=True)
    elif layout == "tree-with-edges":
        convert_crawl_results_to_dot_with_edge_nodes(
            crawl_results, prefix, crawl_direction=direction,
            edge_label_width=54, edge_label_max_chars=400,
            group_parallel_edges=True, semantic_arrowheads=True,
        )
        _flip_dot_arrows(prefix + ".dot", direction, layout)
        engine = "sfdp" if num_nodes > 100 else "dot"
        subprocess.run([engine, "-Tsvg", prefix + ".dot", "-o", svg_path], check=True)
    else:
        convert_crawl_results_to_dot(crawl_results, prefix, crawl_direction=direction)
        _flip_dot_arrows(prefix + ".dot", direction, layout)
        engine = "sfdp" if num_nodes > 100 else "dot"
        subprocess.run([engine, "-Tsvg", prefix + ".dot", "-o", svg_path], check=True)
    render_secs = time.time() - t0

    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    print(f"[render] done: {num_nodes} nodes, SVG {len(svg_content)//1024} KB, "
          f"crawl {crawl_secs:.1f}s + render {render_secs:.1f}s, "
          f"applied_min_children={applied_mc}", flush=True)

    return {
        "svg": svg_content,
        "svg_path": svg_path,
        "num_nodes": num_nodes, "num_edges": num_edges,
        "raw_nodes": raw_nodes, "raw_edges": raw_edges,
        "applied_min_children": applied_mc,
        "crawl_secs": crawl_secs, "render_secs": render_secs,
        "partial_layout": partial_layout,
    }


# ----------------------------------------------------------------------
# Paper / impact helpers
# ----------------------------------------------------------------------
def paper_payload_fast(corpus_id: str) -> Optional[dict]:
    paper = GRAPH.load_paper(corpus_id)
    if paper is None:
        return None
    contributions = []
    for c in paper.contributions:
        contributions.append({
            "contribution_id": c.contribution_id,
            "name": c.name or "",
            "description": (c.description or "")[:400],
            "impact": None,
            "impact_dampened": None,
        })
    return {
        "corpus_id": str(paper.corpus_id),
        "title": paper.title,
        "year": paper.year,
        "n_contributions": len(paper.contributions),
        "contributions": contributions,
        "impact_ready": False,
        "overall_impact": None,
    }


def _impact_timed_out_payload(corpus_id: str, timeout_secs: float) -> Optional[dict]:
    """Same shape as paper_payload_with_impact, but with no scores —
    every contribution is marked 'large'."""
    base = paper_payload_fast(corpus_id)
    if base is None:
        return None
    base["impact_ready"] = False
    base["impact_timed_out"] = True
    base["impact_timeout_secs"] = float(timeout_secs)
    base["overall_impact"] = None
    return base


# ---- subprocess-based cancellable impact computation ----------------
# The impact metric is a recursive crawl with no built-in cancellation, so
# the only way to actually free the CPU when a UI request times out is to
# run it in a child process and kill that child. On Linux we use `fork`,
# which is cheap: the child inherits the loaded GRAPH via copy-on-write,
# so no re-loading and (effectively) no extra resident memory until the
# crawl starts writing pages.
_MP_CTX = None


def _get_fork_ctx():
    global _MP_CTX
    if _MP_CTX is None:
        try:
            _MP_CTX = multiprocessing.get_context("fork")
        except ValueError:
            _MP_CTX = None  # not available on this platform
    return _MP_CTX


def _impact_worker(corpus_id: str, queue) -> None:
    try:
        result = paper_payload_with_impact(corpus_id)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("err", f"{type(e).__name__}: {e}"))


def compute_contribution_impact(contribution_id: str) -> Optional[dict]:
    """One-contribution impact tally. Calls the graph's per-contribution
    method directly (the per-paper method internally calls this N times),
    so the UI can fire one request per contribution and watch each row
    populate independently — partial results survive a single
    contribution's timeout."""
    try:
        result = GRAPH.calculate_impact_metric_contribution(
            contribution_id=contribution_id, max_depth=IMPACT_MAX_DEPTH,
        )
    except Exception:
        traceback.print_exc()
        return None
    if result is None:
        return None
    return {
        "contribution_id": contribution_id,
        "contribution_name": result.get("contribution_name") or "",
        "impact_score": float(result.get("impact_score") or 0.0),
        "impact_score_dampened": float(result.get("impact_score_dampened") or 0.0),
        "ready": True,
        "timed_out": False,
    }


def _contrib_impact_worker(contribution_id: str, queue) -> None:
    try:
        result = compute_contribution_impact(contribution_id)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("err", f"{type(e).__name__}: {e}"))


def compute_contribution_impact_with_timeout(contribution_id: str,
                                             timeout_secs: float) -> dict:
    """Fork-killable version of compute_contribution_impact. On timeout,
    marks this contribution as 'too large' in the soft-cache and returns
    a dict the UI can render as a stand-in."""
    ctx = _get_fork_ctx()
    if ctx is None:
        return compute_contribution_impact(contribution_id) or {
            "contribution_id": contribution_id, "ready": False,
        }

    q = ctx.Queue()
    p = ctx.Process(target=_contrib_impact_worker,
                    args=(contribution_id, q), daemon=True)
    t0 = time.time()
    p.start()
    try:
        status, payload = q.get(timeout=float(timeout_secs))
    except QueueEmpty:
        elapsed = time.time() - t0
        print(f"[contrib-impact] timed out after {elapsed:.1f}s for "
              f"contribution_id={contribution_id} (killing pid={p.pid})",
              flush=True)
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
            p.join(timeout=2)
        CONTRIB_IMPACT_TIMED_OUT[contribution_id] = float(timeout_secs)
        return {
            "contribution_id": contribution_id,
            "ready": False,
            "timed_out": True,
            "timeout_secs": float(timeout_secs),
        }

    p.join(timeout=2)
    if p.is_alive():
        p.kill()

    if status == "ok":
        if payload is not None:
            CONTRIB_IMPACT_CACHE[contribution_id] = payload
        return payload or {
            "contribution_id": contribution_id, "ready": False,
        }
    raise RuntimeError(payload)


def _render_worker(args: dict, queue) -> None:
    try:
        result = render_crawl(**args)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("err", f"{type(e).__name__}: {e}"))


def compute_render_with_timeout(args: dict, timeout_secs: float) -> dict:
    """Fork a child for `render_crawl(**args)` with a wall-time budget.

    For **radial** layouts, the child applies the budget *softly* via a
    tqdm monkey-patch on the contraction loop — when the deadline hits
    the loop stops early and the SVG is written with the partially-
    converged positions. So the user gets a usable (if a bit messier)
    graph rather than a "timed out" screen.

    For **tree** / **tree-with-edges**, there's no equivalent hook —
    those layouts run `dot -Tsvg` as an opaque subprocess. They rely
    on the *hard* parent-side kill if they overrun.

    Either way, we keep a hard parent-side timeout (budget + grace) as
    a safety net against the child hanging on something unexpected.
    """
    ctx = _get_fork_ctx()
    if ctx is None:
        return render_crawl(**args)

    # Soft budget: reserved 3 s for the SVG-write step plus a tiny safety
    # margin. Floor of 5 s so very small budgets still do *some* layout.
    soft_budget = max(float(timeout_secs) - 3.0, 5.0)
    layout_deadline = time.time() + soft_budget
    # Hard cap so a truly-stuck child can't hang the worker indefinitely.
    hard_budget = float(timeout_secs) + 10.0

    child_args = {**args, "layout_deadline_epoch": layout_deadline}

    q = ctx.Queue()
    p = ctx.Process(target=_render_worker, args=(child_args, q), daemon=True)
    t0 = time.time()
    p.start()
    try:
        status, payload = q.get(timeout=hard_budget)
    except QueueEmpty:
        elapsed = time.time() - t0
        print(f"[render] hard kill after {elapsed:.1f}s for "
              f"contribution_id={args.get('contribution_id')} "
              f"(killing pid={p.pid})", flush=True)
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
            p.join(timeout=2)
        return {
            "svg": "",
            "render_timed_out": True,
            "timeout_secs": float(timeout_secs),
            "num_nodes": 0, "num_edges": 0,
            "raw_nodes": 0, "raw_edges": 0,
            "applied_min_children": int(args.get("min_children") or 0),
            "crawl_secs": 0.0, "render_secs": 0.0,
        }

    p.join(timeout=2)
    if p.is_alive():
        p.kill()

    if status == "ok":
        return payload
    raise RuntimeError(payload)


def compute_impact_with_timeout(corpus_id: str, timeout_secs: float) -> Optional[dict]:
    """Fork a child to compute the impact metric; kill it if it overruns."""
    ctx = _get_fork_ctx()
    if ctx is None:
        # No fork available — just compute synchronously without cancellation.
        return paper_payload_with_impact(corpus_id)

    q = ctx.Queue()
    p = ctx.Process(target=_impact_worker, args=(corpus_id, q), daemon=True)
    t0 = time.time()
    p.start()
    try:
        status, payload = q.get(timeout=float(timeout_secs))
    except QueueEmpty:
        elapsed = time.time() - t0
        print(f"[impact] timed out after {elapsed:.1f}s for corpus_id={corpus_id} "
              f"(killing pid={p.pid})", flush=True)
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
            p.join(timeout=2)
        IMPACT_TIMED_OUT[corpus_id] = float(timeout_secs)
        return _impact_timed_out_payload(corpus_id, timeout_secs)

    p.join(timeout=2)
    if p.is_alive():
        p.kill()

    if status == "ok":
        if payload is not None:
            IMPACT_CACHE[corpus_id] = payload
        return payload
    raise RuntimeError(payload)


def paper_payload_with_impact(corpus_id: str) -> Optional[dict]:
    if corpus_id in IMPACT_CACHE:
        return IMPACT_CACHE[corpus_id]

    base = paper_payload_fast(corpus_id)
    if base is None:
        return None
    try:
        impact_metric = GRAPH.calculate_impact_metric_paper(
            corpus_id=str(corpus_id), max_depth=IMPACT_MAX_DEPTH,
        )
    except Exception as exc:
        traceback.print_exc()
        base["impact_error"] = str(exc)
        return base

    contrib_scores = impact_metric.get("contribution_impact_scores", {}) or {}
    overall = impact_metric.get("overall_paper_impact_score") or {}

    by_id = {c["contribution_id"]: c for c in base["contributions"]}
    for contribution_id, info in contrib_scores.items():
        row = by_id.get(contribution_id)
        if row is None:
            continue
        row["impact"] = float(info.get("impact_score") or 0.0)
        row["impact_dampened"] = float(info.get("impact_score_dampened") or 0.0)

    base["contributions"].sort(
        key=lambda c: -(c.get("impact") or -1.0)
    )
    base["impact_ready"] = True
    base["overall_impact"] = {
        "impact_score": float(overall.get("impact_score") or 0.0),
        "impact_score_dampened": float(overall.get("impact_score_dampened") or 0.0),
        "max_depth": IMPACT_MAX_DEPTH,
    }
    IMPACT_CACHE[corpus_id] = base
    return base


# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="Scientific Contribution Graph Explorer")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/help", response_class=HTMLResponse)
def help_page():
    return HTMLResponse(HELP_HTML)


@app.get("/api/search")
def api_search(q: str = Query("", min_length=0), max_results: int = 25):
    return {"results": live_search_titles(q, max_results=max_results)}


@app.get("/api/stats")
def api_stats():
    base = GRAPH_STATS or {"n_papers": 0, "n_contributions": 0, "n_edges": 0}
    # Pass through the operator-configured UI timeouts so the frontend
    # doesn't need to hardcode them.
    return {
        **base,
        "impact_timeout_secs": SETTINGS.get(
            "impact_timeout_secs", DEFAULT_IMPACT_TIMEOUT_SECS),
        "render_timeout_secs": SETTINGS.get(
            "render_timeout_secs", DEFAULT_RENDER_TIMEOUT_SECS),
    }


@app.get("/api/refresh_status")
def api_refresh_status():
    """Public, unauthenticated read-only view of the refresh state.

    The browser polls this to drive the header's disk-icon + timer
    indicator and the current-release pill. Only coarse info is
    exposed — no bucket URI, no admin token, nothing that lets a
    visitor *trigger* a refresh (that's `/admin/refresh`)."""
    status = REFRESH_STATE.get("status", "idle")
    in_progress = status in ("syncing", "reloading")
    release = None
    data_path = SETTINGS.get("data_path") if SETTINGS else None
    if data_path:
        release = _read_release_marker(data_path)
    return {
        "in_progress": in_progress,
        "status": status,
        "started_at": REFRESH_STATE.get("started_at"),
        "finished_at": REFRESH_STATE.get("finished_at"),
        "release": release,
    }


# ----------------------------------------------------------------------
# Admin: refresh the dataset from the bucket and hot-reload the graph
# ----------------------------------------------------------------------
def _admin_authorized(token: Optional[str]) -> bool:
    """Compare against `SCG_ADMIN_TOKEN`. If the env var is unset the
    admin surface is disabled entirely — we never want random visitors
    to be able to kick off a 10-minute download on the public Space."""
    expected = os.environ.get("SCG_ADMIN_TOKEN")
    if not expected:
        return False
    return token is not None and token == expected


def _run_refresh_job(bucket_uri: str) -> None:
    """Background worker for `/admin/refresh`. Holds REFRESH_LOCK for
    the lifetime of the job so concurrent refresh requests serialize."""
    with REFRESH_LOCK:
        try:
            REFRESH_STATE.update({
                "status": "syncing",
                "started_at": time.time(),
                "finished_at": None,
                "message": f"Syncing from {bucket_uri} ...",
            })

            data_path = SETTINGS["data_path"]
            ensure_dataset_downloaded(data_path, bucket_uri, force=True)

            REFRESH_STATE.update({
                "status": "reloading",
                "message": "Loading the new graph into memory ...",
            })

            new_graph = ScientificContributionGraph(
                path=data_path, search_enabled=False, search_device="cpu",
            )
            new_index = build_title_index(new_graph)
            new_stats = compute_graph_stats(new_graph, data_path)

            global GRAPH, TITLE_INDEX, GRAPH_STATS, IMPACT_CACHE, IMPACT_TIMED_OUT
            GRAPH = new_graph
            TITLE_INDEX = new_index
            GRAPH_STATS = new_stats
            # In-flight caches are tied to the *old* graph's IDs — clear them.
            IMPACT_CACHE = {}
            IMPACT_TIMED_OUT = {}

            REFRESH_STATE.update({
                "status": "done",
                "finished_at": time.time(),
                "message": f"Reloaded: {new_stats}",
                "release": _read_release_marker(data_path),
            })
            print(f"[refresh] completed: {new_stats}", flush=True)
        except Exception as exc:
            traceback.print_exc()
            REFRESH_STATE.update({
                "status": "error",
                "finished_at": time.time(),
                "message": f"{type(exc).__name__}: {exc}",
            })


@app.post("/admin/refresh")
def admin_refresh(token: Optional[str] = None,
                  bucket_uri: Optional[str] = None):
    """Kick off a dataset refresh in the background. Returns immediately
    with `{"status": "started", ...}`. Poll `/admin/refresh/status` to
    follow progress.

    Auth: requires `?token=<X>` to match the `SCG_ADMIN_TOKEN` env var.
    If `SCG_ADMIN_TOKEN` is unset, the endpoint is disabled (returns 403).

    `bucket_uri` defaults to the URI the server was launched with."""
    if not _admin_authorized(token):
        raise HTTPException(status_code=403,
                            detail="Admin disabled or invalid token")

    target_bucket = bucket_uri or SETTINGS.get("bucket_uri") or DEFAULT_BUCKET_URI

    # Don't queue concurrent refreshes — bail if one is already running.
    if REFRESH_LOCK.locked():
        raise HTTPException(status_code=409,
                            detail=f"A refresh is already running: {REFRESH_STATE}")

    threading.Thread(target=_run_refresh_job, args=(target_bucket,),
                     name="scg-refresh", daemon=True).start()
    return {"status": "started", "bucket_uri": target_bucket}


@app.get("/admin/refresh/status")
def admin_refresh_status(token: Optional[str] = None):
    if not _admin_authorized(token):
        raise HTTPException(status_code=403,
                            detail="Admin disabled or invalid token")
    return REFRESH_STATE


@app.get("/api/paper/{corpus_id}")
def api_paper(corpus_id: str):
    data = paper_payload_fast(corpus_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Paper {corpus_id} not found")
    return data


@app.get("/api/paper/{corpus_id}/impact")
async def api_paper_impact(corpus_id: str, timeout: Optional[float] = None):
    """Impact-metric endpoint.

    - **No `timeout`**: behaves exactly as before. The request waits as long
      as the computation takes. Intended for direct API callers — they can
      wait.
    - **With `timeout=N`**: if the metric isn't ready in `N` seconds, kill
      the underlying computation and return the fast paper payload marked
      `impact_timed_out=true`. The browser UI uses this.
    """
    if corpus_id in IMPACT_CACHE:
        return IMPACT_CACHE[corpus_id]

    if timeout is None:
        # API mode — no timeout, no subprocess.
        data = await asyncio.to_thread(paper_payload_with_impact, corpus_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Paper {corpus_id} not found")
        return data

    # UI mode — if we've already timed out on this paper at >= this
    # timeout budget, don't re-burn CPU; just return the soft response.
    prior = IMPACT_TIMED_OUT.get(corpus_id)
    if prior is not None and float(timeout) <= prior:
        soft = _impact_timed_out_payload(corpus_id, float(timeout))
        if soft is None:
            raise HTTPException(status_code=404, detail=f"Paper {corpus_id} not found")
        return soft

    data = await asyncio.to_thread(
        compute_impact_with_timeout, corpus_id, float(timeout),
    )
    if data is None:
        raise HTTPException(status_code=404, detail=f"Paper {corpus_id} not found")
    return data


@app.get("/api/render")
async def api_render(
    contribution_id: str,
    direction: str = "forward",
    layout: str = "tree",
    max_depth: int = 1,
    only_strong: bool = True,
    min_children: int = 0,
    auto_trim: bool = True,
    timeout: Optional[float] = None,
):
    """Crawl + render endpoint.

    - **No `timeout`**: behaves as before; the request blocks for as long
      as the crawl + layout take. Intended for direct API callers.
    - **With `timeout=N`**: the work runs in a forked child so the parent
      can kill it after `N` seconds. Returns `render_timed_out=true` if it
      overruns. The browser UI uses this.
    """
    if direction not in ("forward", "backward"):
        raise HTTPException(status_code=400, detail="direction must be forward|backward")
    if layout not in ("tree", "tree-with-edges", "radial"):
        raise HTTPException(status_code=400, detail="layout must be tree|tree-with-edges|radial")
    args = {
        "contribution_id": contribution_id,
        "direction": direction,
        "layout": layout,
        "max_depth": int(max_depth),
        "only_strong": bool(only_strong),
        "min_children": int(min_children),
        "auto_trim": bool(auto_trim),
    }
    try:
        if timeout is None:
            result = await asyncio.to_thread(render_crawl, **args)
        else:
            result = await asyncio.to_thread(
                compute_render_with_timeout, args, float(timeout),
            )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    # Strip svg_path from response — the client doesn't need it.
    result.pop("svg_path", None)
    return JSONResponse(result)


@app.get("/api/contribution/{contribution_id}/impact")
async def api_contribution_impact(contribution_id: str,
                                  timeout: Optional[float] = None):
    """Per-contribution impact metric. The UI fires one request per
    contribution so the column populates progressively rather than
    all-or-nothing. API callers without `timeout` are never time-limited."""
    if contribution_id in CONTRIB_IMPACT_CACHE:
        return CONTRIB_IMPACT_CACHE[contribution_id]

    if timeout is None:
        data = await asyncio.to_thread(compute_contribution_impact,
                                       contribution_id)
        if data is None:
            raise HTTPException(status_code=404,
                                detail=f"Contribution {contribution_id} not found")
        return data

    # UI mode — if we've already given up on this contribution at >= this
    # timeout budget, return the "too large" stand-in immediately.
    prior = CONTRIB_IMPACT_TIMED_OUT.get(contribution_id)
    if prior is not None and float(timeout) <= prior:
        return {
            "contribution_id": contribution_id,
            "ready": False,
            "timed_out": True,
            "timeout_secs": float(timeout),
        }

    data = await asyncio.to_thread(
        compute_contribution_impact_with_timeout, contribution_id, float(timeout),
    )
    return data


@app.get("/api/contribution/{contribution_id}")
def api_contribution(contribution_id: str):
    c = GRAPH.get_contribution_by_id(contribution_id)
    paper_info = GRAPH.get_paper_info_by_contribution_id(contribution_id)
    if c is None:
        raise HTTPException(status_code=404, detail=f"Contribution {contribution_id} not found")
    return {
        "contribution_id": c.contribution_id,
        "name": c.name or "",
        "description": c.description or "",
        "types": [{"type": t.type, "explanation": t.explanation} for t in (c.types or [])],
        "sections": c.sections or [],
        "n_prerequisites": len(c.prerequisites or []),
        "paper_info": paper_info or {},
    }


@app.get("/api/download")
def api_download(
    contribution_id: str,
    direction: str = "forward",
    layout: str = "tree",
):
    """Return the most recently rendered SVG for this contribution as a download."""
    prefix = os.path.join(WORK_DIR, _safe_name(f"crawl_{direction}_{contribution_id}"))
    svg_path = prefix + ".svg"
    if not os.path.exists(svg_path):
        raise HTTPException(status_code=404, detail="No SVG yet — render first.")
    filename = f"scg_{direction}_{layout}_{_safe_name(contribution_id)}.svg"
    return FileResponse(svg_path, media_type="image/svg+xml", filename=filename)


# ----------------------------------------------------------------------
# Single-page HTML/CSS/JS — the entire frontend.
# ----------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Scientific Contribution Graph Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
      integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA=="
      crossorigin="anonymous" referrerpolicy="no-referrer" />
<style>
  :root {
    --bg: #fafafa;
    --panel: #ffffff;
    --ink: #1a1a1a;
    --muted: #6b7280;
    --muted-2: #9ca3af;
    --line: #e5e7eb;
    --line-2: #d1d5db;
    --accent: #2563eb;
    --accent-soft: #eff6ff;
    --row-hover: #f3f4f6;
    --row-selected: #dbeafe;
    --warn-bg: #fff7ed;
    --warn-line: #fed7aa;
    --warn-ink: #9a3412;
    --shadow: 0 1px 2px rgba(0,0,0,.04), 0 1px 1px rgba(0,0,0,.03);
    --radius: 8px;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: var(--ink); background: var(--bg);
    font-size: 14px; line-height: 1.45;
    height: 100%;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  button {
    font: inherit; color: inherit;
    background: var(--panel);
    border: 1px solid var(--line-2);
    border-radius: 6px;
    padding: 6px 10px;
    cursor: pointer;
  }
  button:hover { background: var(--row-hover); }
  button.primary {
    background: var(--accent); color: white; border-color: var(--accent);
  }
  button.primary:hover { background: #1d4ed8; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  input, select {
    font: inherit; color: inherit;
    background: var(--panel);
    border: 1px solid var(--line-2);
    border-radius: 6px;
    padding: 7px 10px;
    width: 100%;
  }
  input:focus, select:focus, button:focus {
    outline: 2px solid var(--accent); outline-offset: -1px;
  }
  .hide { display: none !important; }

  header {
    background: var(--panel);
    border-bottom: 1px solid var(--line);
    padding: 10px 20px;
    display: flex; align-items: center; gap: 16px;
  }
  header h1 {
    margin: 0; font-size: 17px; font-weight: 600; letter-spacing: -0.01em;
  }
  header .subtitle { color: var(--muted); font-size: 13px; }
  header .spacer { flex: 1; }
  header .stats {
    display: flex; align-items: center; gap: 14px;
    font-size: 12px; color: var(--muted);
  }
  header .stats .stat { display: inline-flex; align-items: center; gap: 5px; }
  header .stats .stat .num {
    color: var(--ink); font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  header .stats .stat i { color: var(--muted-2); width: 13px; text-align: center; }
  header .paper-link {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 9px; border: 1px solid var(--line-2); border-radius: 6px;
    color: var(--accent); font-size: 12px; font-weight: 500;
    background: var(--panel);
  }
  header .paper-link:hover { background: var(--accent-soft); text-decoration: none; }

  /* "Update in progress" indicator — disk icon + live mm:ss timer.
     Hidden when status === "idle"; visible during sync / reload. */
  header .update-indicator {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 4px 10px;
    background: #fff7ed; border: 1px solid #fed7aa;
    color: #9a3412; border-radius: 6px;
    font-size: 12px; font-weight: 500;
  }
  header .update-indicator i {
    animation: scg-blink 1.4s ease-in-out infinite;
  }
  @keyframes scg-blink {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.35; }
  }
  /* Brief "just finished" flash; auto-fades back to hidden. */
  header .update-indicator.done {
    background: #ecfdf5; border-color: #a7f3d0; color: #065f46;
  }
  header .update-indicator.done i { animation: none; }
  header .update-indicator.error {
    background: #fef2f2; border-color: #fecaca; color: #991b1b;
  }
  header .update-indicator.error i { animation: none; }

  /* Compact pill showing the currently-installed release name. */
  header .release-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; color: var(--muted-2);
    font-family: ui-monospace, monospace;
  }
  header .release-pill:empty { display: none; }

  /* Grid: search column | content column. */
  .layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    grid-template-rows: 1fr;
    height: calc(100vh - 49px);
    min-height: 0;
  }

  /* Left column: search + paper picker. */
  .left {
    border-right: 1px solid var(--line);
    background: var(--panel);
    display: flex; flex-direction: column; min-height: 0;
  }
  .left .search-bar { padding: 12px; border-bottom: 1px solid var(--line); }
  .left .search-meta { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .left .results { flex: 1; overflow-y: auto; min-height: 0; }
  .result-row {
    padding: 9px 14px; border-bottom: 1px solid var(--line);
    cursor: pointer; line-height: 1.4;
  }
  .result-row:hover { background: var(--row-hover); }
  .result-row.selected { background: var(--row-selected); }
  .result-row .title { font-weight: 500; }
  .result-row .meta { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .empty {
    padding: 24px 14px; color: var(--muted); font-style: italic; font-size: 13px;
  }

  /* Right column: paper details + viz. */
  .right {
    display: grid;
    grid-template-rows: auto 1fr;
    min-height: 0;
  }

  .paper-panel {
    background: var(--panel);
    border-bottom: 1px solid var(--line);
    padding: 10px 16px;
    display: flex; flex-direction: column; gap: 6px;
    max-height: 28vh;
  }
  .paper-panel.empty-state {
    display: flex; align-items: center; justify-content: center;
    min-height: 80px;
  }
  .paper-title {
    font-size: 15px; font-weight: 600; line-height: 1.35;
  }
  .paper-meta { font-size: 12px; color: var(--muted); }
  .paper-meta .pill {
    background: var(--accent-soft); color: var(--accent);
    padding: 1px 6px; border-radius: 4px; font-weight: 500;
    margin-left: 6px;
  }
  .paper-meta .pill.pending {
    background: #fef3c7; color: #92400e;
    display: inline-flex; align-items: center; gap: 6px;
  }
  /* Tiny inline spinner that lives inside the .pill.pending — shows
     that per-contribution impact calculations are streaming in. */
  .pill.pending .mini-spinner {
    width: 11px; height: 11px;
    border: 2px solid #fde68a;
    border-top-color: #92400e;
    border-radius: 50%;
    animation: scg-spin 0.9s linear infinite;
    display: inline-block;
  }
  /* And a slow pulse on the impact cells that are still pending, so the
     user sees the table is actively working its way down the list. */
  table.contribs .impact-cell.pending {
    color: var(--muted-2);
    animation: scg-impact-pulse 1.6s ease-in-out infinite;
  }
  @keyframes scg-impact-pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.35; }
  }
  .contribs-wrap {
    flex: 1; min-height: 0; overflow-y: auto;
    border: 1px solid var(--line); border-radius: 6px;
  }
  table.contribs {
    width: 100%; border-collapse: collapse; font-size: 13px;
  }
  table.contribs thead th {
    position: sticky; top: 0; background: #f9fafb;
    text-align: left; font-weight: 600; color: var(--muted);
    padding: 8px 10px; border-bottom: 1px solid var(--line);
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
  }
  table.contribs tbody td {
    padding: 8px 10px; border-bottom: 1px solid var(--line);
    vertical-align: top;
  }
  table.contribs tbody tr { cursor: pointer; }
  table.contribs tbody tr:hover { background: var(--row-hover); }
  table.contribs tbody tr.selected { background: var(--row-selected); }
  table.contribs .impact-cell {
    text-align: right; font-variant-numeric: tabular-nums;
    white-space: nowrap; color: var(--muted);
  }
  table.contribs .impact-cell.has { color: var(--ink); font-weight: 500; }
  table.contribs .cid { color: var(--muted-2); font-size: 11px; font-family: ui-monospace, monospace; }

  /* Bottom-right: viz area + sidebar of knobs. */
  .viz-area {
    display: grid;
    grid-template-columns: 1fr 260px;
    grid-template-rows: 1fr;
    min-height: 0;
  }
  .viz-main {
    position: relative; background: white;
    border-right: 1px solid var(--line);
    overflow: hidden;
  }
  .viz-info {
    position: absolute; top: 8px; left: 12px; right: 12px;
    background: rgba(255,255,255,0.95);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px; color: var(--muted);
    z-index: 5;
    box-shadow: var(--shadow);
  }
  .viz-info .strong { color: var(--ink); font-weight: 500; }
  .viz-toolbar {
    position: absolute; top: 8px; right: 12px; z-index: 6;
    background: rgba(255,255,255,0.95);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 3px 4px;
    display: flex; gap: 2px;
    box-shadow: var(--shadow);
  }
  .viz-toolbar button { padding: 4px 9px; font-size: 13px; }
  .viz-hint {
    position: absolute; bottom: 8px; left: 12px;
    background: rgba(255,255,255,0.85);
    color: var(--muted); font-size: 11px;
    padding: 3px 7px; border-radius: 4px;
    z-index: 5;
  }
  .viz-host {
    width: 100%; height: 100%;
    display: flex; align-items: center; justify-content: center;
  }
  .viz-host svg.viz-svg {
    width: 100%; height: 100%; display: block;
  }
  .viz-empty {
    color: var(--muted); font-style: italic; padding: 40px; text-align: center;
    line-height: 1.6;
  }
  .viz-empty .arrow { font-size: 28px; display:block; margin-bottom: 10px; opacity: 0.5; }

  /* Sidebar. */
  .sidebar {
    background: var(--panel);
    padding: 14px 16px;
    display: flex; flex-direction: column; gap: 14px;
    overflow-y: auto; min-height: 0;
    font-size: 13px;
  }
  .sidebar h3 {
    margin: 0; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--muted); font-weight: 600;
  }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field > label { font-weight: 500; }
  .field .hint { font-size: 11px; color: var(--muted); }
  .slider-row { display: flex; align-items: center; gap: 10px; }
  .slider-row input[type=range] { flex: 1; }
  .slider-row .value {
    min-width: 22px; text-align: right;
    font-variant-numeric: tabular-nums; font-weight: 500;
  }
  .checkbox-row {
    display: flex; align-items: center; gap: 8px;
  }
  .checkbox-row input { width: auto; }
  .checkbox-row label { font-weight: 500; cursor: pointer; }
  .sidebar .actions {
    display: flex; flex-direction: column; gap: 8px;
    margin-top: 4px;
  }
  .sidebar .actions button { width: 100%; }
  .sidebar .status {
    font-size: 11px; color: var(--muted);
    border-top: 1px solid var(--line);
    padding-top: 10px;
    min-height: 16px;
  }
  .sidebar .status.busy { color: var(--warn-ink); }

  /* Spinner. */
  .spinner-overlay {
    position: absolute; inset: 0;
    background: rgba(255,255,255,0.85);
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 14px;
    z-index: 10;
  }
  .spinner {
    width: 56px; height: 56px;
    border: 5px solid var(--line);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  .spinner-msg { color: var(--ink); font-weight: 500; font-size: 14px; text-align: center; max-width: 520px; padding: 0 16px; }
  .spinner-sub { color: var(--muted); font-size: 12px; margin-top: -8px; text-align: center; max-width: 520px; padding: 0 16px; }
  .spinner-hint { color: var(--muted-2); font-size: 11px; font-style: italic; margin-top: -4px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Responsive tweak. */
  @media (max-width: 980px) {
    .layout { grid-template-columns: 1fr; grid-template-rows: 300px 1fr; }
    .left { border-right: none; border-bottom: 1px solid var(--line); }
    .viz-area { grid-template-columns: 1fr; grid-template-rows: 1fr auto; }
    .sidebar { border-right: none; border-top: 1px solid var(--line); max-height: 240px; }
  }
</style>
</head>
<body>
<header>
  <h1>Scientific Contribution Graph Explorer</h1>
  <span class="subtitle">Search a paper → pick a contribution → see what it built on and what built on it.</span>
  <span class="spacer"></span>
  <div class="update-indicator hide" id="update-indicator"
       title="A dataset refresh is in progress. The UI keeps serving the current graph until the new one is loaded.">
    <i class="fa-solid fa-hard-drive"></i>
    <span id="update-text">Updating data — 0:00</span>
  </div>
  <span class="release-pill" id="release-pill" title="Currently-installed release"></span>
  <div class="stats" id="stats">
    <span class="stat" title="Papers"><i class="fa-regular fa-file-lines"></i><span class="num" id="stat-papers">…</span> papers</span>
    <span class="stat" title="Scientific contributions"><i class="fa-solid fa-lightbulb"></i><span class="num" id="stat-contribs">…</span> contributions</span>
    <span class="stat" title="Citation-style edges"><i class="fa-solid fa-diagram-project"></i><span class="num" id="stat-edges">…</span> edges</span>
  </div>
  <a class="paper-link"
     href="https://arxiv.org/abs/2605.15011"
     target="_blank" rel="noopener"
     title="Read the paper on arXiv">
    <i class="fa-solid fa-book-open"></i>
    Paper
  </a>
  <a class="paper-link"
     href="https://github.com/cognitiveailab/scientific-contribution-graph"
     target="_blank" rel="noopener"
     title="Source code &amp; full API on GitHub">
    <i class="fa-brands fa-github"></i>
    GitHub
  </a>
  <a class="paper-link"
     href="/help" target="_blank" rel="noopener"
     title="What this is, how to use it, common issues">
    <i class="fa-regular fa-circle-question"></i>
    Help
  </a>
</header>

<div class="layout">

  <!-- LEFT: search + results -->
  <aside class="left">
    <div class="search-bar">
      <input id="search-input" type="text"
             placeholder="Search papers by title…"
             autocomplete="off" autocorrect="off" spellcheck="false" autofocus />
      <div class="search-meta" id="search-meta">Type at least 2 characters.</div>
    </div>
    <div class="results" id="results">
      <div class="empty">No search yet.</div>
    </div>
  </aside>

  <!-- RIGHT: paper details (top) + viz area (bottom) -->
  <section class="right">

    <div class="paper-panel empty-state" id="paper-panel">
      <div class="empty">Select a paper from the left to see its contributions.</div>
    </div>

    <div class="viz-area">
      <div class="viz-main" id="viz-main">
        <div class="viz-host" id="viz-host">
          <div class="viz-empty">
            <span class="arrow">↑</span>
            Pick a contribution above to render its citation graph here.
          </div>
        </div>
        <div class="viz-info hide" id="viz-info"></div>
        <div class="viz-toolbar hide" id="viz-toolbar">
          <button id="viz-zoom-in"  title="Zoom in">+</button>
          <button id="viz-zoom-out" title="Zoom out">−</button>
          <button id="viz-fit"      title="Fit to view">⤢</button>
        </div>
        <div class="viz-hint hide" id="viz-hint">scroll to zoom · drag to pan</div>
        <div class="spinner-overlay hide" id="spinner">
          <div class="spinner"></div>
          <div class="spinner-msg" id="spinner-msg">Rendering…</div>
          <div class="spinner-sub" id="spinner-sub"></div>
          <div class="spinner-hint">This may take up to a minute, depending on the size of the graph…</div>
        </div>
      </div>

      <aside class="sidebar">
        <h3>Visualization</h3>

        <div class="field">
          <label for="direction">Direction</label>
          <select id="direction">
            <option value="backward" selected>Backward (what this built on)</option>
            <option value="forward">Forward (what built on this)</option>
          </select>
        </div>

        <div class="field">
          <label for="layout">Layout</label>
          <select id="layout">
            <option value="tree" selected>Tree</option>
            <option value="tree-with-edges">Tree (with edge labels)</option>
            <option value="radial">Radial</option>
          </select>
        </div>

        <div class="field">
          <label for="max-depth">Depth <span class="hint">(higher = bigger, slower)</span></label>
          <div class="slider-row">
            <input id="max-depth" type="range" min="1" max="4" step="1" value="1" />
            <span class="value" id="max-depth-val">1</span>
          </div>
        </div>

        <div class="field">
          <label for="min-children">Min children <span class="hint">(trim leaves)</span></label>
          <div class="slider-row">
            <input id="min-children" type="range" min="0" max="10" step="1" value="0" />
            <span class="value" id="min-children-val">0</span>
          </div>
          <div class="hint">Auto-bumped on first render to keep graphs readable.</div>
        </div>

        <div class="checkbox-row">
          <input id="only-strong" type="checkbox" checked />
          <label for="only-strong">Strong connections only</label>
        </div>

        <div class="actions">
          <button class="primary" id="rerender-btn" disabled>Re-render</button>
          <button id="download-btn" disabled>Download SVG</button>
        </div>

        <div class="status" id="render-status">Pick a contribution to begin.</div>
      </aside>
    </div>

  </section>
</div>

<script>
(() => {
  // --- state ---
  const state = {
    selectedCorpusId: null,
    selectedContributionId: null,
    selectedContributionName: "",
    impactReady: false,
    panZoom: null,
    renderSeq: 0,         // bumped on each render to ignore stale responses
    searchSeq: 0,         // same idea for the search
    pendingRender: null,  // {timer, ...}
    impactSeq: 0,
    // True only for the very first render of a freshly-picked contribution.
    // Lets the backend auto-bump min_children once for a readable first
    // view; after that, all user knob values are respected exactly.
    autoTrimNextRender: false,
    // UI timeouts. Initialized with sane defaults; the first /api/stats
    // response overwrites them with the operator's configured values.
    timeouts: { impact: 10, render: 30 },
  };

  // --- DOM ---
  const $ = (id) => document.getElementById(id);
  const els = {
    search: $("search-input"),
    searchMeta: $("search-meta"),
    results: $("results"),
    paperPanel: $("paper-panel"),
    vizMain: $("viz-main"),
    vizHost: $("viz-host"),
    vizInfo: $("viz-info"),
    vizToolbar: $("viz-toolbar"),
    vizHint: $("viz-hint"),
    spinner: $("spinner"),
    spinnerMsg: $("spinner-msg"),
    spinnerSub: $("spinner-sub"),
    direction: $("direction"),
    layout: $("layout"),
    maxDepth: $("max-depth"),
    maxDepthVal: $("max-depth-val"),
    minChildren: $("min-children"),
    minChildrenVal: $("min-children-val"),
    onlyStrong: $("only-strong"),
    rerenderBtn: $("rerender-btn"),
    downloadBtn: $("download-btn"),
    renderStatus: $("render-status"),
    statPapers: $("stat-papers"),
    statContribs: $("stat-contribs"),
    statEdges: $("stat-edges"),
    updateIndicator: $("update-indicator"),
    updateText: $("update-text"),
    releasePill: $("release-pill"),
  };

  // ---- Refresh-status indicator (header disk icon + live mm:ss timer) ----
  // Polls /api/refresh_status; while a refresh is running, shows a pulsing
  // disk icon with elapsed-minutes counter. Briefly flashes "done"/error
  // when a refresh completes, then collapses.
  const refreshIndicator = (() => {
    let lastSeenFinishedAt = null;
    let startedAt = null;
    let tickHandle = null;
    let doneFlashTimer = null;
    // First poll: latch whatever finished_at the server reports without
    // flashing, so opening the page well after a past refresh doesn't
    // pop up a stale "Update failed" / "Data updated" banner.
    let firstPoll = true;

    const fmtTimer = (secs) => {
      if (!Number.isFinite(secs) || secs < 0) secs = 0;
      const m = Math.floor(secs / 60);
      const s = secs % 60;
      return `${m}:${String(s).padStart(2, "0")}`;
    };

    const tick = () => {
      if (startedAt == null) return;
      els.updateText.textContent =
        `Updating data — ${fmtTimer(Math.floor(Date.now() / 1000 - startedAt))}`;
    };

    const startTicker = () => {
      tick();
      if (!tickHandle) tickHandle = setInterval(tick, 1000);
    };
    const stopTicker = () => {
      if (tickHandle) { clearInterval(tickHandle); tickHandle = null; }
    };

    const flashDone = (status, message) => {
      if (doneFlashTimer) clearTimeout(doneFlashTimer);
      els.updateIndicator.classList.remove("hide");
      els.updateIndicator.classList.toggle("done",  status === "done");
      els.updateIndicator.classList.toggle("error", status === "error");
      els.updateText.textContent = status === "done"
        ? "Data updated"
        : `Update failed${message ? `: ${message}` : ""}`;
      doneFlashTimer = setTimeout(() => {
        els.updateIndicator.classList.add("hide");
        els.updateIndicator.classList.remove("done", "error");
      }, status === "done" ? 8000 : 15000);
    };

    const poll = async () => {
      try {
        const r = await fetch("/api/refresh_status");
        const s = await r.json();

        // Release pill: shows currently-installed tarball name (truncated).
        if (s.release && s.release.tarball) {
          const t = s.release.tarball.replace(/\.tar\.gz$/, "");
          els.releasePill.innerHTML =
            `<i class="fa-solid fa-tag"></i> ${esc(t)}`;
        }

        // Live indicator
        if (s.in_progress) {
          if (doneFlashTimer) { clearTimeout(doneFlashTimer); doneFlashTimer = null; }
          els.updateIndicator.classList.remove("hide", "done", "error");
          startedAt = s.started_at || (Date.now() / 1000);
          startTicker();
          lastSeenFinishedAt = null;
        } else {
          stopTicker();
          startedAt = null;
          if (firstPoll) {
            // Latch whatever the server is reporting on the first poll —
            // don't pop a stale flash from a previous session's refresh.
            lastSeenFinishedAt = s.finished_at || null;
          } else if (s.finished_at && s.finished_at !== lastSeenFinishedAt &&
                     (s.status === "done" || s.status === "error")) {
            // A refresh just completed (or errored) since the last poll.
            lastSeenFinishedAt = s.finished_at;
            flashDone(s.status, "");
            if (s.status === "done") {
              // Stats may have changed — re-fetch.
              fetch("/api/stats").then(r => r.json()).then(st => {
                els.statPapers.textContent   = fmtNum(st.n_papers);
                els.statContribs.textContent = fmtNum(st.n_contributions);
                els.statEdges.textContent    = fmtNum(st.n_edges);
              }).catch(() => {});
            }
          } else if (!doneFlashTimer) {
            els.updateIndicator.classList.add("hide");
          }
        }
        firstPoll = false;
      } catch (e) { /* keep last state on transient failure */ }
    };

    return { poll };
  })();
  // Polling kicked off below, once the helpers (esc, fmtNum) are in scope.

  const fmtNum = (n) => {
    if (n == null) return "?";
    return n.toLocaleString();
  };

  fetch("/api/stats")
    .then(r => r.json())
    .then(s => {
      els.statPapers.textContent   = fmtNum(s.n_papers);
      els.statContribs.textContent = fmtNum(s.n_contributions);
      els.statEdges.textContent    = fmtNum(s.n_edges);
      if (s.impact_timeout_secs) state.timeouts.impact = Number(s.impact_timeout_secs);
      if (s.render_timeout_secs) state.timeouts.render = Number(s.render_timeout_secs);
    })
    .catch(() => {});

  // Kick off the refresh-status poller (now that esc/fmtNum are defined).
  refreshIndicator.poll();
  setInterval(() => refreshIndicator.poll(), 10000);

  // --- helpers ---
  const esc = (s) => String(s == null ? "" : s)
      .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

  const debounce = (fn, ms) => {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  };

  const fmtImpact = (v) => {
    if (v == null) return "…";
    if (v === 0) return "0";
    if (v >= 100) return v.toFixed(0);
    if (v >= 10) return v.toFixed(1);
    return v.toFixed(2);
  };

  const setStatus = (text, busy = false) => {
    els.renderStatus.textContent = text || "";
    els.renderStatus.classList.toggle("busy", !!busy);
  };

  const showSpinner = (msg, sub = "") => {
    els.spinnerMsg.textContent = msg;
    els.spinnerSub.textContent = sub;
    els.spinner.classList.remove("hide");
  };
  const hideSpinner = () => els.spinner.classList.add("hide");

  // --- search ---
  const renderResults = (results) => {
    if (!results.length) {
      els.results.innerHTML = '<div class="empty">No matches.</div>';
      return;
    }
    els.results.innerHTML = results.map(r => `
      <div class="result-row ${r.corpus_id === state.selectedCorpusId ? 'selected' : ''}"
           data-corpus-id="${esc(r.corpus_id)}">
        <div class="title">${esc(r.title)}</div>
        <div class="meta">corpus ${esc(r.corpus_id)}</div>
      </div>
    `).join("");
    els.results.querySelectorAll(".result-row").forEach(row => {
      row.addEventListener("click", () => selectPaper(row.dataset.corpusId));
    });
  };

  const doSearch = async (q) => {
    const seq = ++state.searchSeq;
    q = (q || "").trim();
    if (q.length < 2) {
      els.searchMeta.textContent = "Type at least 2 characters.";
      els.results.innerHTML = '<div class="empty">No search yet.</div>';
      return;
    }
    els.searchMeta.textContent = "Searching…";
    try {
      const r = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      const data = await r.json();
      if (seq !== state.searchSeq) return; // stale
      const n = data.results.length;
      els.searchMeta.textContent = n === 0 ? "No matches." :
        `${n} match${n === 1 ? "" : "es"}.`;
      renderResults(data.results);
    } catch (e) {
      els.searchMeta.textContent = "Search failed.";
    }
  };

  els.search.addEventListener("input", debounce((e) => doSearch(e.target.value), 120));

  // --- paper picker ---
  const renderPaperPanel = (paper) => {
    if (!paper) {
      els.paperPanel.className = "paper-panel empty-state";
      els.paperPanel.innerHTML =
        '<div class="empty">Select a paper from the left to see its contributions.</div>';
      return;
    }
    const impactPill = paper.impact_ready
      ? `<span class="pill">impact ${fmtImpact(paper.overall_impact?.impact_score)}
         · dampened ${fmtImpact(paper.overall_impact?.impact_score_dampened)}
         · depth ${paper.overall_impact?.max_depth ?? ''}</span>`
      : paper.impact_timed_out
        ? `<span class="pill pending" title="Crawl exceeded ${paper.impact_timeout_secs ?? state.timeouts.impact}s — terminated to keep the UI responsive">impact too large to compute quickly</span>`
        : `<span class="pill pending"><span class="mini-spinner"></span>computing impact ${paper.impact_progress_done ?? 0}/${paper.n_contributions ?? '?'}…</span>`;
    els.paperPanel.className = "paper-panel";
    els.paperPanel.innerHTML = `
      <div class="paper-title">${esc(paper.title || "(untitled)")}</div>
      <div class="paper-meta">
        corpus <code>${esc(paper.corpus_id)}</code>
        · year ${esc(paper.year ?? "?")}
        · ${paper.n_contributions} contribution${paper.n_contributions === 1 ? "" : "s"}
        ${impactPill}
      </div>
      <div class="contribs-wrap">
        <table class="contribs">
          <thead>
            <tr>
              <th style="width:55%;">Contribution</th>
              <th style="width:15%;" class="impact-cell">Impact</th>
              <th style="width:15%;" class="impact-cell">Dampened</th>
              <th style="width:15%;">ID</th>
            </tr>
          </thead>
          <tbody id="contribs-tbody">
            ${paper.contributions.map((c, i) => {
              const hasImpact = c.impact != null;
              const hasDamped = c.impact_dampened != null;
              const isTimedOut = c.impact_timed_out === true;
              const impactText = isTimedOut
                ? '<span title="Crawl exceeded the UI timeout — try the API or take the contribution-level forward crawl manually">large</span>'
                : (hasImpact ? fmtImpact(c.impact) : '…');
              const dampedText = isTimedOut
                ? '<span>large</span>'
                : (hasDamped ? fmtImpact(c.impact_dampened) : '…');
              const impactCls  = (!hasImpact && !isTimedOut) ? 'pending' : (hasImpact || isTimedOut ? 'has' : '');
              const dampedCls  = (!hasDamped && !isTimedOut) ? 'pending' : (hasDamped || isTimedOut ? 'has' : '');
              return `
              <tr data-cid="${esc(c.contribution_id)}"
                  data-cname="${esc(c.name)}"
                  class="${c.contribution_id === state.selectedContributionId ? 'selected' : ''}">
                <td>${esc(c.name || "(unnamed)")}</td>
                <td class="impact-cell ${impactCls}">${impactText}</td>
                <td class="impact-cell ${dampedCls}">${dampedText}</td>
                <td><span class="cid">${esc(c.contribution_id)}</span></td>
              </tr>
            `;}).join("")}
          </tbody>
        </table>
      </div>
    `;
    document.querySelectorAll("#contribs-tbody tr").forEach(tr => {
      tr.addEventListener("click", () => {
        selectContribution(tr.dataset.cid, tr.dataset.cname);
      });
    });
  };

  // Helper: update one contribution row's impact cells in-place without
  // re-rendering the whole table. Lets the user keep their selection /
  // scroll position while impacts stream in.
  const updateContribRow = (cid, info) => {
    const tr = document.querySelector(`#contribs-tbody tr[data-cid="${CSS.escape(cid)}"]`);
    if (!tr) return;
    const [, impactCell, dampedCell] = tr.children;
    if (info.timed_out) {
      impactCell.className = "impact-cell has";
      dampedCell.className = "impact-cell has";
      impactCell.innerHTML =
        '<span title="Crawl exceeded the UI timeout — try the API for the exact number">large</span>';
      dampedCell.innerHTML = '<span>large</span>';
    } else if (info.ready) {
      impactCell.className = "impact-cell has";
      dampedCell.className = "impact-cell has";
      impactCell.textContent = fmtImpact(info.impact_score);
      dampedCell.textContent = fmtImpact(info.impact_score_dampened);
    }
  };

  const selectPaper = async (corpusId) => {
    if (!corpusId) return;
    state.selectedCorpusId = corpusId;
    state.selectedContributionId = null;
    state.selectedContributionName = "";
    state.impactReady = false;
    document.querySelectorAll(".result-row").forEach(r => {
      r.classList.toggle("selected", r.dataset.corpusId === corpusId);
    });
    clearViz();
    setStatus("Pick a contribution to begin.");

    // Fast call first — contributions appear immediately, as placeholders.
    const seq = ++state.impactSeq;
    let paper;
    try {
      const r = await fetch(`/api/paper/${encodeURIComponent(corpusId)}`);
      if (seq !== state.impactSeq) return;
      paper = await r.json();
      paper.impact_progress_done = 0;
      renderPaperPanel(paper);
    } catch (e) {
      els.paperPanel.innerHTML = `<div class="empty">Failed to load paper.</div>`;
      return;
    }

    // Per-contribution impact loop. Sequential — multiple parallel fork+crawl
    // jobs on a 2-vCPU HF Space would just queue at the OS level anyway, and
    // sequential gives steady visible progress. If the user clicks away to a
    // different paper, the `seq` check aborts the loop immediately.
    let totalImpact = 0, totalDampened = 0;
    let anySucceeded = false;
    for (let i = 0; i < paper.contributions.length; i++) {
      if (seq !== state.impactSeq) return;
      const c = paper.contributions[i];
      let info;
      try {
        const r = await fetch(
          `/api/contribution/${encodeURIComponent(c.contribution_id)}/impact?timeout=${state.timeouts.impact}`,
        );
        info = await r.json();
      } catch (e) {
        info = { timed_out: true };
      }
      if (seq !== state.impactSeq) return;

      if (info.ready) {
        c.impact = info.impact_score;
        c.impact_dampened = info.impact_score_dampened;
        c.impact_timed_out = false;
        totalImpact   += info.impact_score   || 0;
        totalDampened += info.impact_score_dampened || 0;
        anySucceeded = true;
      } else if (info.timed_out) {
        c.impact = null;
        c.impact_dampened = null;
        c.impact_timed_out = true;
      }
      updateContribRow(c.contribution_id, info);

      // Update the header progress counter without nuking the table body.
      paper.impact_progress_done = i + 1;
      const progressPill = document.querySelector(".paper-panel .pill.pending");
      if (progressPill) {
        if (i + 1 < paper.contributions.length) {
          progressPill.innerHTML =
            `<span class="mini-spinner"></span>computing impact ${i + 1}/${paper.contributions.length}…`;
        } else {
          // All done — replace the pill with a final summary.
          paper.impact_ready = true;
          paper.overall_impact = {
            impact_score: totalImpact,
            impact_score_dampened: totalDampened,
            max_depth: null,
          };
          progressPill.outerHTML = anySucceeded
            ? `<span class="pill">impact ${fmtImpact(totalImpact)}
                · dampened ${fmtImpact(totalDampened)}</span>`
            : `<span class="pill pending" title="Every contribution's crawl exceeded the UI timeout — try the API">impact too large to compute quickly</span>`;
        }
      }
    }
    state.impactReady = true;
  };

  // --- contribution picker / render ---
  const clearViz = () => {
    if (state.panZoom) {
      try { state.panZoom.destroy(); } catch (e) {}
      state.panZoom = null;
    }
    els.vizHost.innerHTML = `
      <div class="viz-empty">
        <span class="arrow">↑</span>
        Pick a contribution above to render its citation graph here.
      </div>`;
    els.vizInfo.classList.add("hide");
    els.vizToolbar.classList.add("hide");
    els.vizHint.classList.add("hide");
    els.downloadBtn.disabled = true;
    els.rerenderBtn.disabled = true;
  };

  const showSvg = (svgText) => {
    if (state.panZoom) {
      try { state.panZoom.destroy(); } catch (e) {}
      state.panZoom = null;
    }
    // Inject the SVG with our class, stripping any fixed sizes so it scales.
    const svgWithClass = svgText.replace(
      /<svg\b/,
      '<svg class="viz-svg" preserveAspectRatio="xMidYMid meet"',
    );
    els.vizHost.innerHTML = svgWithClass;
    const svg = els.vizHost.querySelector("svg.viz-svg");
    if (!svg) return;
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    try {
      state.panZoom = svgPanZoom(svg, {
        controlIconsEnabled: false, fit: true, center: true, contain: true,
        zoomScaleSensitivity: 0.35, minZoom: 0.05, maxZoom: 30,
      });
    } catch (e) { console.warn(e); }
    els.vizToolbar.classList.remove("hide");
    els.vizHint.classList.remove("hide");
  };

  const renderNow = async () => {
    if (!state.selectedContributionId) return;
    const seq = ++state.renderSeq;
    const autoTrim = !!state.autoTrimNextRender;
    const params = new URLSearchParams({
      contribution_id: state.selectedContributionId,
      direction: els.direction.value,
      layout: els.layout.value,
      max_depth: els.maxDepth.value,
      only_strong: els.onlyStrong.checked ? "true" : "false",
      min_children: els.minChildren.value,
      auto_trim: autoTrim ? "true" : "false",
      // UI deadline; backend kills the underlying crawl on overrun.
      // Configured via demo_config.json (render_timeout_secs), but the
      // user can opt into 120 s for the rest of the session via the
      // "I'll wait" button on the timeout screen — that mutates
      // state.timeouts.render so all subsequent renders inherit it.
      timeout: String(state.timeouts.render),
    });
    const dirLabel = els.direction.value;
    showSpinner(
      `Crawling ${dirLabel} from contribution…`,
      `${state.selectedContributionName.slice(0, 70)}`,
    );
    setStatus("Rendering…", true);
    els.rerenderBtn.disabled = true;
    els.downloadBtn.disabled = true;
    try {
      const r = await fetch(`/api/render?${params}`);
      if (seq !== state.renderSeq) return;
      if (!r.ok) {
        const err = await r.json().catch(() => ({detail: r.statusText}));
        hideSpinner();
        els.vizHost.innerHTML = `
          <div class="viz-empty">Render failed: ${esc(err.detail)}</div>`;
        setStatus(`Failed: ${err.detail}`);
        els.rerenderBtn.disabled = false;
        return;
      }
      const data = await r.json();
      if (seq !== state.renderSeq) return;
      hideSpinner();

      // Render timed out — backend killed the worker, so CPU is back.
      if (data.render_timed_out) {
        const secs = data.timeout_secs ?? 30;
        const offerLonger = state.timeouts.render < 120;
        els.vizInfo.classList.add("hide");
        els.vizToolbar.classList.add("hide");
        els.vizHint.classList.add("hide");
        els.vizHost.innerHTML = `
          <div class="viz-empty">
            <span class="arrow"><i class="fa-solid fa-hourglass-end"></i></span>
            <strong>Timed out after ${secs}s — graph too large for this web demo.</strong><br>
            <span style="display:inline-block; margin-top:8px;">
              For very deep / popular contributions, please use the full API:
              <a href="https://github.com/cognitiveailab/scientific-contribution-graph"
                 target="_blank" rel="noopener">
                <i class="fa-brands fa-github"></i> cognitiveailab/scientific-contribution-graph
              </a>.
            </span><br>
            <span style="display:inline-block; margin-top:8px; font-size:12px;">
              Or try reducing <em>Depth</em>, raising <em>Min children</em>,
              or enabling <em>Strong connections only</em>.
            </span>
            ${offerLonger ? `
              <div style="margin-top:16px;">
                <button class="primary" id="extend-timeout-btn"
                        title="Raise the render timeout to 2 minutes for the rest of this session and retry.">
                  <i class="fa-solid fa-hourglass-half"></i>
                  I'll wait — try again, increasing the render limit to 2 minutes
                </button>
              </div>` : ''}
          </div>`;
        if (offerLonger) {
          document.getElementById("extend-timeout-btn").addEventListener("click", () => {
            // Session-level bump — all subsequent renders inherit this.
            state.timeouts.render = 120;
            renderNow();
          });
        }
        setStatus(`Timed out after ${secs}s.`);
        els.downloadBtn.disabled = true;
        els.rerenderBtn.disabled = false;
        state.autoTrimNextRender = false;
        return;
      }

      // Auto-trim may have moved the min-children slider, but ONLY on the
      // first render for a contribution. After that, the slider is the
      // user's — never override it.
      const userMc = parseInt(els.minChildren.value, 10) || 0;
      if (autoTrim && data.applied_min_children != null &&
          data.applied_min_children !== userMc) {
        els.minChildren.value = data.applied_min_children;
        els.minChildrenVal.textContent = data.applied_min_children;
      }
      // From here on, the user is in control.
      state.autoTrimNextRender = false;

      const info = (
        `<span class="strong">${esc(state.selectedContributionId)}</span> · ` +
        `${esc(state.selectedContributionName)}<br>` +
        `${esc(dirLabel)} crawl · ${esc(els.layout.value)} · depth ${els.maxDepth.value} · ` +
        `strong-only ${els.onlyStrong.checked} · ` +
        `<span class="strong">${data.num_nodes}</span> nodes / ` +
        `<span class="strong">${data.num_edges}</span> edges` +
        (data.applied_min_children !== userMc
          ? ` · auto-trim min_children=${data.applied_min_children} (${data.raw_nodes}→${data.num_nodes})`
          : "") +
        ` · crawl ${data.crawl_secs.toFixed(1)}s` +
        (data.render_secs ? ` + render ${data.render_secs.toFixed(1)}s` : "") +
        (data.partial_layout
          ? ` · <span title="The radial contraction loop stopped early to stay within the time budget — positions are partially-converged. Click 'I'll wait' on the next timeout, or switch to a tree layout for a fully-converged view.">partial layout</span>`
          : "")
      );
      els.vizInfo.innerHTML = info;
      els.vizInfo.classList.remove("hide");

      if (!data.svg) {
        els.vizHost.innerHTML = `
          <div class="viz-empty">${esc(data.empty_reason || "Nothing to render.")}</div>`;
        els.vizToolbar.classList.add("hide");
        els.vizHint.classList.add("hide");
        setStatus("Empty graph at these settings.");
      } else {
        showSvg(data.svg);
        setStatus(
          `Done · ${data.num_nodes} nodes / ${data.num_edges} edges · ` +
          `${data.crawl_secs.toFixed(1)}s + ${data.render_secs.toFixed(1)}s`,
        );
        els.downloadBtn.disabled = false;
      }
      els.rerenderBtn.disabled = false;
    } catch (e) {
      hideSpinner();
      els.vizHost.innerHTML = `
        <div class="viz-empty">Render error: ${esc(e.message)}</div>`;
      setStatus(`Error: ${e.message}`);
      els.rerenderBtn.disabled = false;
    }
  };

  const scheduleRender = debounce(renderNow, 250);

  const selectContribution = (cid, name) => {
    state.selectedContributionId = cid;
    state.selectedContributionName = name || "";
    document.querySelectorAll("#contribs-tbody tr").forEach(tr => {
      tr.classList.toggle("selected", tr.dataset.cid === cid);
    });
    // Reset to safe initial knob values + let the backend auto-trim ONCE
    // so the first render is fast and readable.
    els.maxDepth.value = 1;
    els.maxDepthVal.textContent = "1";
    els.minChildren.value = 0;
    els.minChildrenVal.textContent = "0";
    state.autoTrimNextRender = true;
    renderNow();
  };

  // --- knob wiring ---
  els.maxDepth.addEventListener("input", () => {
    els.maxDepthVal.textContent = els.maxDepth.value;
  });
  els.minChildren.addEventListener("input", () => {
    els.minChildrenVal.textContent = els.minChildren.value;
  });

  // Live re-render when knobs change (debounced).
  const liveTriggers = [els.direction, els.layout, els.onlyStrong];
  liveTriggers.forEach(el => el.addEventListener("change", scheduleRender));
  // Sliders fire on every step — debounce harder.
  [els.maxDepth, els.minChildren].forEach(el =>
    el.addEventListener("change", scheduleRender),
  );

  els.rerenderBtn.addEventListener("click", renderNow);
  els.downloadBtn.addEventListener("click", () => {
    if (!state.selectedContributionId) return;
    const params = new URLSearchParams({
      contribution_id: state.selectedContributionId,
      direction: els.direction.value,
      layout: els.layout.value,
    });
    window.location.href = `/api/download?${params}`;
  });

  // Toolbar.
  $("viz-zoom-in").addEventListener("click", () => state.panZoom?.zoomIn());
  $("viz-zoom-out").addEventListener("click", () => state.panZoom?.zoomOut());
  $("viz-fit").addEventListener("click", () => {
    if (state.panZoom) {
      state.panZoom.resize();
      state.panZoom.fit();
      state.panZoom.center();
    }
  });

})();
</script>
</body>
</html>
"""


HELP_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Help · Scientific Contribution Graph Explorer</title>
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
      integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA=="
      crossorigin="anonymous" referrerpolicy="no-referrer" />
<style>
  :root {
    --bg: #fafafa; --panel: #ffffff;
    --ink: #1a1a1a; --muted: #6b7280; --muted-2: #9ca3af;
    --line: #e5e7eb; --line-2: #d1d5db;
    --accent: #2563eb; --accent-soft: #eff6ff;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: var(--ink); background: var(--bg);
    font-size: 15px; line-height: 1.6;
  }
  header {
    background: var(--panel);
    border-bottom: 1px solid var(--line);
    padding: 12px 24px;
    display: flex; align-items: center; gap: 14px;
  }
  header h1 {
    margin: 0; font-size: 17px; font-weight: 600; letter-spacing: -0.01em;
  }
  header .subtitle { color: var(--muted); font-size: 13px; }
  header .spacer { flex: 1; }
  header a.back {
    display: inline-flex; align-items: center; gap: 6px;
    color: var(--accent); font-size: 13px; font-weight: 500;
    padding: 5px 10px; border: 1px solid var(--line-2); border-radius: 6px;
    text-decoration: none; background: var(--panel);
  }
  header a.back:hover { background: var(--accent-soft); }

  main {
    max-width: 820px; margin: 0 auto; padding: 32px 28px 80px;
  }
  h2 {
    font-size: 22px; font-weight: 600; margin: 36px 0 10px;
    letter-spacing: -0.01em;
    border-bottom: 1px solid var(--line); padding-bottom: 6px;
  }
  h2:first-of-type { margin-top: 0; }
  h2 i { color: var(--muted-2); margin-right: 9px; font-size: 17px; }
  h3 {
    font-size: 16px; font-weight: 600; margin: 20px 0 6px;
  }
  p { margin: 8px 0; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  ul, ol { padding-left: 22px; margin: 6px 0; }
  li { margin: 4px 0; }
  code, kbd {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: #f3f4f6;
    padding: 1px 5px; border-radius: 4px;
    font-size: 0.9em;
  }
  kbd {
    border: 1px solid var(--line-2);
    box-shadow: 0 1px 0 var(--line-2);
  }
  .lead {
    font-size: 16px; color: var(--ink);
    background: var(--accent-soft);
    border-left: 3px solid var(--accent);
    padding: 12px 16px; border-radius: 0 6px 6px 0;
    margin-bottom: 24px;
  }
  .note {
    background: #fff7ed; border-left: 3px solid #fb923c;
    padding: 10px 14px; border-radius: 0 6px 6px 0;
    margin: 12px 0; font-size: 14px;
  }
  .note i { color: #c2410c; margin-right: 6px; }
  .step {
    display: flex; gap: 14px; padding: 12px 0;
    border-bottom: 1px dashed var(--line);
  }
  .step:last-child { border-bottom: none; }
  .step .num {
    flex: 0 0 30px; height: 30px;
    background: var(--accent); color: white;
    border-radius: 50%; font-weight: 600; font-size: 14px;
    display: flex; align-items: center; justify-content: center;
  }
  .step .body { flex: 1; }
  .step .body strong { font-size: 15px; }
  dl.issues dt {
    font-weight: 600; margin-top: 14px;
    color: var(--ink);
  }
  dl.issues dt i { color: var(--muted-2); margin-right: 8px; font-size: 14px; }
  dl.issues dd { margin: 4px 0 0; color: var(--muted); }
  dl.issues dd, dl.issues dd code { color: var(--ink); }
  .footer-note {
    margin-top: 40px; padding-top: 18px;
    border-top: 1px solid var(--line);
    font-size: 13px; color: var(--muted);
  }
</style>
</head>
<body>
<header>
  <h1>Help — Scientific Contribution Graph Explorer</h1>
  <span class="subtitle">What this is, how to use it, common issues.</span>
  <span class="spacer"></span>
  <a class="back" href="/" title="Back to the explorer">
    <i class="fa-solid fa-arrow-left"></i> Back to explorer
  </a>
</header>

<main>

<div class="lead">
  This is a browser-based explorer for the <strong>Scientific Contribution Graph</strong>:
  a large knowledge graph that maps how each scientific contribution
  builds on prior contributions, and how later contributions in turn
  build on it.
</div>

<h2><i class="fa-solid fa-circle-info"></i>What is this?</h2>

<p>
  The Scientific Contribution Graph extracts the <em>scientific
  contributions</em> from each paper in a large corpus and links them
  together by <em>prerequisite relationships</em>. For example, if a
  recent paper develops some contribution X, and X was built in part
  from contributions A, B, and C in earlier papers, the graph records
  each of those A→X, B→X, C→X links — so you can trace how later
  technologies were built from earlier ones. The current release covers
  <strong>~230k papers</strong>, <strong>~2M contributions</strong>, and
  <strong>~12.5M prerequisite edges</strong>, drawn mostly from
  open-access NLP and AI literature.
</p>

<p>
  Each contribution has a name, a description, a list of types
  (e.g. <em>dataset</em>, <em>method</em>, <em>analysis</em>), and a list
  of prerequisite contributions it builds on. Each prerequisite has a
  <em>strength</em> (<strong>strong</strong> if the prerequisite is
  directly required; <strong>weak</strong> if it's only loosely related).
</p>

<p>
  This explorer lets you:
</p>
<ul>
  <li>Search any paper by title.</li>
  <li>See every contribution in that paper, ranked by <em>downstream
      impact</em> (how many later contributions build on it, directly or
      indirectly).</li>
  <li>Render a forward or backward citation graph for any contribution,
      with live-adjustable knobs.</li>
</ul>

<p>
  For background on the data, methodology, and impact metric, see the
  <a href="https://arxiv.org/abs/2605.15011" target="_blank" rel="noopener">paper</a>,
  or the
  <a href="https://github.com/cognitiveailab/scientific-contribution-graph"
     target="_blank" rel="noopener">GitHub repo</a> (which also exposes
  the full Python API, beyond what this demo surfaces).
</p>


<h2><i class="fa-solid fa-compass"></i>How to use it</h2>

<div class="step">
  <div class="num">1</div>
  <div class="body">
    <strong>Search for a paper.</strong>
    Type at least two characters in the <kbd>Search papers by title…</kbd>
    box on the left. Matches appear live; click a row to select it.
  </div>
</div>

<div class="step">
  <div class="num">2</div>
  <div class="body">
    <strong>Pick a contribution.</strong>
    The top of the main panel lists every contribution in the selected
    paper. The <em>Impact</em> column shows the total count of later
    contributions that build on it; <em>Dampened</em> applies a
    reciprocal-rank-by-depth weighting (direct citations count as 1,
    depth-2 contributions count as 0.5, depth-3 as 0.33, …). Click any
    row to render its citation graph.
  </div>
</div>

<div class="step">
  <div class="num">3</div>
  <div class="body">
    <strong>Tune the visualization.</strong>
    The sidebar on the right has live knobs — changes re-render
    automatically.
    <ul>
      <li><strong>Direction</strong> — <em>Backward</em> shows what this
          contribution was built on; <em>Forward</em> shows what was
          built on it.</li>
      <li><strong>Layout</strong> — <em>Tree</em> (Graphviz, fastest),
          <em>Tree (with edge labels)</em> (annotates each edge with a
          short summary of the prerequisite relation), or
          <em>Radial</em> (force-directed; better for medium graphs).</li>
      <li><strong>Depth</strong> — how many hops to traverse. Higher
          values mean exponentially larger graphs.</li>
      <li><strong>Min children</strong> — drop nodes whose subtree has
          fewer than this many children. Useful for trimming peripheral
          leaves in big graphs. The first render of each contribution
          auto-bumps this to keep the initial view readable; once you
          touch the slider, your value is honored exactly.</li>
      <li><strong>Strong connections only</strong> — restrict to
          prerequisites that are directly required (excludes weak
          / loosely-related links).</li>
    </ul>
  </div>
</div>

<div class="step">
  <div class="num">4</div>
  <div class="body">
    <strong>Pan, zoom, and download.</strong>
    Scroll-wheel to zoom, drag to pan; the <kbd>+</kbd> / <kbd>−</kbd> /
    <kbd>⤢</kbd> toolbar in the top-right of the viz area zooms in/out
    and fits the view. <strong>Download SVG</strong> in the sidebar
    saves a vector copy of the current render.
  </div>
</div>


<h2><i class="fa-solid fa-triangle-exclamation"></i>Common issues</h2>

<dl class="issues">

  <dt><i class="fa-solid fa-hourglass-end"></i>"Impact too large to compute quickly"</dt>
  <dd>
    For very highly-cited contributions (e.g., BERT, Transformers), the
    impact metric crawl can take much longer than the UI's configured
    timeout budget. When that happens the contributions table shows
    <strong>large</strong> in the impact columns rather than freezing
    the page. You can still click any contribution to render its
    citation graph — the timeout only affects the impact tally. For
    exact numbers, use the
    <a href="https://github.com/cognitiveailab/scientific-contribution-graph"
       target="_blank" rel="noopener">Python API</a> directly, which
    has no timeout.
  </dd>

  <dt><i class="fa-solid fa-hourglass-end"></i>"Timed out — graph too large for this web demo"</dt>
  <dd>
    The render has a configurable timeout budget. If a crawl plus layout exceeds
    that, the underlying worker is killed and you see this message. To
    fit within the budget, try one or more of:
    <ul>
      <li>Lower <strong>Depth</strong> (depth 1 or 2 is usually enough).</li>
      <li>Raise <strong>Min children</strong> (trims peripheral nodes).</li>
      <li>Enable <strong>Strong connections only</strong>.</li>
      <li>Switch to the <em>Tree</em> layout (it's the fastest of the three).</li>
    </ul>
    Again, for very large graphs the Python API has no timeout — you
    can render them locally with full control over layout parameters.
  </dd>

  <dt><i class="fa-solid fa-magnifying-glass"></i>Why can't I find a specific paper?</dt>
  <dd>
    The current corpus contains about a <strong>quarter million open-access
    papers</strong>, centred on natural language processing — so if the
    paper you're looking for is closed-access, sits in a different
    subfield, or is very new, it may simply not be in the release yet.
    The title search is also a token-coverage match (not a semantic
    search): try just a few distinctive words from the title. If you
    have the paper's Semantic Scholar <code>corpus_id</code>, you can
    also call <code>GET /api/paper/{corpus_id}</code> directly.
  </dd>

  <dt><i class="fa-solid fa-arrow-trend-down"></i>Why does the impact score look lower than I'd expect?</dt>
  <dd>
    The downstream-impact tally only counts <em>contributions in the
    current corpus</em> that build on the target. Papers outside the
    open-access NLP focus, or that haven't been crawled yet, won't be
    represented — so a foundational paper that is heavily cited in
    closed-access venues or in adjacent subfields will look smaller
    here than it really is. Coverage will continue to fill in as the
    crawl expands.
  </dd>

  <dt><i class="fa-solid fa-puzzle-piece"></i>Why doesn't this paper show contributions I know it built on (or its full impact)?</dt>
  <dd>
    Three reasons are usually at play:
    <ul>
      <li><strong>Open-access focus.</strong> The crawl is over open-access
          papers, so coverage of the <a href="https://aclanthology.org"
          target="_blank" rel="noopener">ACL Anthology</a> is good, but
          coverage of the broader AI / ML literature is thinner.
          Prerequisites that point to closed-access work won't have
          matched-contribution links into the graph.</li>
      <li><strong>The original crawl ran in the backward (citation) direction.</strong>
          That is, for each paper it identifies the prior contributions
          that paper was <em>built on</em>. This is optimised for the
          "what does this build on?" question — the technological
          roadmapping direction — rather than for forward-impact
          measurement, so forward-impact may be systematically
          undercounted in this initial release.</li>
      <li><strong>The current (continuing) crawl mixes both directions.</strong>
          Subsequent expansion passes interleave forward and backward
          crawls to balance technological roadmapping and impact
          assessment — so coverage in the forward direction will keep
          improving with each release.</li>
    </ul>
  </dd>

  <dt><i class="fa-solid fa-database"></i>Where is all the data for each contribution?</dt>
  <dd>
    Each contribution has a rich schema — name, description, types,
    sections, prerequisites with their own descriptions, explanations,
    strengths, matched references, and more — and only a small slice of
    that is shown in the visualizations and tables here. For the full
    per-contribution payload, use the Python API on
    <a href="https://github.com/cognitiveailab/scientific-contribution-graph"
       target="_blank" rel="noopener">GitHub</a>, or download the
    release and read the source files directly — they're stored as
    easily-readable JSON.
  </dd>

  <dt><i class="fa-solid fa-arrows-spin"></i>Is the crawl ongoing?</dt>
  <dd>
    Yes — the corpus is still being actively crawled and expanded, and
    we expect to update the public release at regular milestones. If a
    paper or contribution is missing today, it may well be present in
    a future release; check back periodically, or watch the
    <a href="https://github.com/cognitiveailab/scientific-contribution-graph"
       target="_blank" rel="noopener">GitHub repo</a> for release
    announcements.
  </dd>

  <dt><i class="fa-solid fa-diagram-project"></i>The graph looks empty or has only one node</dt>
  <dd>
    Some contributions have no downstream/upstream contributions at the
    current depth, or all their links are weak and you've enabled
    <strong>Strong connections only</strong>. Try increasing
    <strong>Depth</strong>, lowering <strong>Min children</strong>, or
    turning off the strong-only checkbox.
  </dd>

  <dt><i class="fa-solid fa-gauge-high"></i>The first render after picking a contribution is slow</dt>
  <dd>
    The first crawl for a fresh contribution does the actual graph
    traversal; subsequent re-renders only re-layout. If you're going to
    explore a contribution thoroughly, expect the first render to take
    a few seconds, then later knob changes to be much faster.
  </dd>

</dl>

<div class="footer-note">
  Built on the
  <a href="https://github.com/cognitiveailab/scientific-contribution-graph"
     target="_blank" rel="noopener">Scientific Contribution Graph</a> library.
  Bug reports and feature requests welcome on the GitHub issue tracker.
</div>

</main>
</body>
</html>
"""


# ----------------------------------------------------------------------
# Settings resolution (CLI > env > config file > defaults)
# ----------------------------------------------------------------------
def _load_config_file(path: str) -> Dict[str, Any]:
    """Read `demo_config.json` if it exists; otherwise return {}.

    Unknown keys are ignored. Comments are not supported in standard JSON;
    keep the file minimal."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            print(f"Warning: {path} is not a JSON object; ignoring.",
                  file=sys.stderr)
            return {}
        print(f"Loaded config from: {path}")
        return cfg
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}", file=sys.stderr)
        return {}


def _first_non_none(*values):
    for v in values:
        if v is not None and v != "":
            return v
    return None


def resolve_settings(args) -> Dict[str, Any]:
    """Merge CLI args > env vars > config file > built-in defaults."""
    cfg_path = args.config or os.environ.get("SCG_CONFIG") or DEFAULT_CONFIG_PATH
    cfg = _load_config_file(cfg_path)

    env = os.environ
    return {
        "data_path": _first_non_none(
            args.data_path, env.get("SCG_DATA_PATH"),
            cfg.get("data_path"), DEFAULT_DATA_PATH,
        ),
        "bucket_uri": _first_non_none(
            args.bucket_uri, env.get("SCG_BUCKET_URI"),
            cfg.get("bucket_uri"),
        ),
        "host": _first_non_none(
            args.host, env.get("SCG_SERVER_NAME"),
            cfg.get("host"), "127.0.0.1",
        ),
        "port": int(_first_non_none(
            args.port, env.get("SCG_SERVER_PORT"),
            cfg.get("port"), 7860,
        )),
        "impact_timeout_secs": float(_first_non_none(
            env.get("SCG_IMPACT_TIMEOUT_SECS"),
            cfg.get("impact_timeout_secs"),
            DEFAULT_IMPACT_TIMEOUT_SECS,
        )),
        "render_timeout_secs": float(_first_non_none(
            env.get("SCG_RENDER_TIMEOUT_SECS"),
            cfg.get("render_timeout_secs"),
            DEFAULT_RENDER_TIMEOUT_SECS,
        )),
    }


# Sidecar file written next to the unpacked data; lets `ensure_dataset_downloaded`
# decide cheaply whether a future bucket sync actually contains a NEW release
# vs. the same one we already extracted (compares tarball name + size).
RELEASE_MARKER_NAME = ".scg_release_marker.json"


def _read_release_marker(data_root: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(data_root, RELEASE_MARKER_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_release_marker(data_root: str, marker: Dict[str, Any]) -> None:
    path = os.path.join(data_root, RELEASE_MARKER_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2)


def ensure_dataset_downloaded(data_root: str, bucket_uri: str,
                              force: bool = False) -> str:
    """Make sure `data_root/data/papers/` exists. If not (or `force=True`),
    sync the release tarball from `bucket_uri` (an `hf://buckets/...` URI)
    and extract it.

    On every refresh we run `hf buckets sync`, which is cheap if nothing
    changed (the bucket sync is metadata-driven). We then compare the
    incoming tarball's name + size against `RELEASE_MARKER_NAME` written
    on the previous install; if it matches, we delete the just-fetched
    tarball and skip extraction. If it's different, we wipe the old
    `data/` directory and extract the new one, then update the marker.

    Idempotent. Used both at startup and by the `/admin/refresh` endpoint."""
    target_data_dir = os.path.join(data_root, "data")
    target_papers_dir = os.path.join(target_data_dir, "papers")
    already_installed = (os.path.isdir(target_data_dir)
                         and os.path.isdir(target_papers_dir))

    # Fast path: at startup, when data is already installed and the caller
    # didn't ask to force, we don't even hit the bucket.
    if already_installed and not force:
        print(f"Dataset already present at: {target_data_dir}")
        return data_root

    os.makedirs(data_root, exist_ok=True)
    print(f"Syncing dataset from {bucket_uri} into {data_root} ...")
    subprocess.run(["hf", "buckets", "sync", bucket_uri, data_root, "--verbose"],
                   check=True)

    tar_files = [f for f in os.listdir(data_root) if f.endswith(".tar.gz")]
    if not tar_files:
        if already_installed:
            print("No tarball in bucket; existing data is current.")
            return data_root
        raise RuntimeError(f"No .tar.gz found in {data_root} after bucket sync.")

    tarball = os.path.join(data_root, sorted(tar_files)[-1])
    tar_stat = os.stat(tarball)
    new_marker = {
        "tarball": os.path.basename(tarball),
        "size_bytes": tar_stat.st_size,
        "extracted_at": time.time(),
        "bucket_uri": bucket_uri,
    }

    current_marker = _read_release_marker(data_root)
    same_release = (
        current_marker is not None
        and current_marker.get("tarball") == new_marker["tarball"]
        and current_marker.get("size_bytes") == new_marker["size_bytes"]
    )
    if already_installed and same_release:
        print(f"Local data already matches latest release "
              f"({new_marker['tarball']}, {new_marker['size_bytes']} B); "
              "skipping extract.")
        try:
            os.remove(tarball)
        except OSError:
            pass
        return data_root

    # We've got a new release (or no marker, or force). Wipe the old
    # data and extract the new tarball. We do delete-then-extract rather
    # than extract-to-temp-and-swap because a 50 GB persistent disk
    # can't comfortably hold 60 GB during the swap.
    if os.path.isdir(target_data_dir):
        print(f"Removing old data at {target_data_dir} ...")
        shutil.rmtree(target_data_dir)

    print(f"Extracting {tarball} ...")
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=data_root)

    if not os.path.isdir(target_data_dir):
        raise RuntimeError(f"Expected {target_data_dir} to exist after extraction.")

    _write_release_marker(data_root, new_marker)

    # Free the 7 GB tarball — important on small persistent disks.
    try:
        os.remove(tarball)
        print(f"Removed tarball: {tarball}")
    except OSError as e:
        print(f"Warning: could not remove tarball {tarball}: {e}",
              file=sys.stderr)

    print(f"Dataset ready at: {target_data_dir} "
          f"(release {new_marker['tarball']})")
    return data_root


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help=f"Path to JSON config file (default: {DEFAULT_CONFIG_PATH} if present)")
    parser.add_argument("--data-path", default=None,
                        help="Local path to an unpacked Scientific Contribution Graph release")
    parser.add_argument("--bucket-uri", default=None,
                        help="hf://buckets/... URI to sync the release from on first boot")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-sync + re-extract from the bucket "
                             "even if data is already present locally.")
    args = parser.parse_args()

    settings = resolve_settings(args)
    print(f"Effective settings: {settings}")

    global GRAPH, WORK_DIR, TITLE_INDEX, GRAPH_STATS, SETTINGS
    SETTINGS = settings

    if not shutil.which("dot"):
        print("Warning: Graphviz `dot` not on PATH; tree layouts won't render. "
              "Install with `sudo apt-get install graphviz`.", file=sys.stderr)

    # HF Spaces mode: bucket_uri set → fetch dataset first.
    if settings["bucket_uri"]:
        ensure_dataset_downloaded(settings["data_path"], settings["bucket_uri"],
                                  force=args.force_refresh)
    elif args.force_refresh:
        print("Warning: --force-refresh has no effect without a bucket_uri.",
              file=sys.stderr)
    print(f"Loading Scientific Contribution Graph from: {settings['data_path']}")
    GRAPH = ScientificContributionGraph(
        path=settings["data_path"], search_enabled=False, search_device="cpu",
    )
    TITLE_INDEX = build_title_index(GRAPH)
    print(f"Built title index over {len(TITLE_INDEX)} papers.")
    GRAPH_STATS = compute_graph_stats(GRAPH, settings["data_path"])
    print(f"Graph stats: {GRAPH_STATS}")
    WORK_DIR = tempfile.mkdtemp(prefix="scg_demo_")
    print(f"Visualization files will be written to: {WORK_DIR}")

    import uvicorn
    print(f"Starting server on http://{settings['host']}:{settings['port']} ...")
    uvicorn.run(app, host=settings["host"], port=settings["port"],
                log_level="info")


if __name__ == "__main__":
    main()
