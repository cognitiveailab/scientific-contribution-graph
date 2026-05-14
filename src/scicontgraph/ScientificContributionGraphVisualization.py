# ScientificContributionGraphVisualization.py
# Helpers for visualizing crawls of subsets of the scientific contribution graph

#
#   Visualization helpers/wrappers
#


def _crawl_edge_dependency_pair(edge: dict):
    """Return (prerequisite_id, dependent_id) for one crawl edge, or (None, None)."""
    contribution_id = edge.get("contribution_id", None)
    if contribution_id is None:
        return None, None

    contribution_id = str(contribution_id)

    if edge.get("used_by_contribution_id", None) is not None:
        return contribution_id, str(edge.get("used_by_contribution_id"))

    if edge.get("prerequisite_for_contribution_id", None) is not None:
        return contribution_id, str(edge.get("prerequisite_for_contribution_id"))

    if edge.get("prerequisite_contribution_id", None) is not None:
        return str(edge.get("prerequisite_contribution_id")), contribution_id

    return None, None


def _infer_crawl_direction(crawl_results: dict) -> str:
    """Infer whether a crawl result should be visualized as forward or backward."""
    from collections import defaultdict, deque

    root_node = crawl_results.get("root_node", None)
    nodes = crawl_results.get("nodes", {}) or {}
    edges = crawl_results.get("edges", []) or []

    if root_node is None or root_node not in nodes:
        return "forward"

    def reachable_count(direction: str) -> int:
        parent_to_children = defaultdict(set)

        for edge in edges:
            prerequisite_id, dependent_id = _crawl_edge_dependency_pair(edge)
            if prerequisite_id is None or dependent_id is None:
                continue
            if prerequisite_id not in nodes or dependent_id not in nodes:
                continue

            if direction == "forward":
                parent_id = prerequisite_id
                child_id = dependent_id
            elif direction == "backward":
                parent_id = dependent_id
                child_id = prerequisite_id
            else:
                raise RuntimeError(f"Unknown direction: {direction}")

            parent_to_children[parent_id].add(child_id)

        visited = set([root_node])
        q = deque([root_node])

        while len(q) > 0:
            parent_id = q.popleft()
            for child_id in parent_to_children.get(parent_id, set()):
                if child_id in visited:
                    continue
                visited.add(child_id)
                q.append(child_id)

        return len(visited)

    forward_reachable = reachable_count("forward")
    backward_reachable = reachable_count("backward")

    if backward_reachable > forward_reachable:
        return "backward"

    return "forward"


def _normalize_crawl_results_for_visualization(crawl_results: dict, crawl_direction: str = "auto") -> dict:
    """
    Return a copy of crawl_results whose edges are visual parent -> visual child.

    The normalized edge format is:
        contribution_id          = visual parent
        used_by_contribution_id = visual child

    For forward crawls, visual parent -> child is prerequisite -> dependent.
    For backward crawls, visual parent -> child is dependent -> prerequisite.
    """
    import copy

    if crawl_direction == "auto":
        crawl_direction = _infer_crawl_direction(crawl_results)

    if crawl_direction not in ["forward", "backward"]:
        raise RuntimeError("crawl_direction must be one of: 'auto', 'forward', 'backward'")

    nodes = crawl_results.get("nodes", {}) or {}
    normalized = copy.deepcopy(crawl_results)
    normalized["nodes"] = copy.deepcopy(nodes)
    normalized_edges = []

    for edge in crawl_results.get("edges", []) or []:
        prerequisite_id, dependent_id = _crawl_edge_dependency_pair(edge)
        if prerequisite_id is None or dependent_id is None:
            continue
        if prerequisite_id not in nodes or dependent_id not in nodes:
            continue

        if crawl_direction == "forward":
            parent_id = prerequisite_id
            child_id = dependent_id
        else:
            parent_id = dependent_id
            child_id = prerequisite_id

        normalized_edge = copy.deepcopy(edge)
        normalized_edge["contribution_id"] = parent_id
        normalized_edge["used_by_contribution_id"] = child_id

        # Remove alternate endpoint spellings so downstream code has one source of truth.
        normalized_edge.pop("prerequisite_for_contribution_id", None)
        normalized_edge.pop("prerequisite_contribution_id", None)

        normalized_edges.append(normalized_edge)

    normalized["edges"] = normalized_edges
    normalized["visualization_crawl_direction"] = crawl_direction
    return normalized


def _sanitize_label_text(text, placeholder: str = "") -> str:
    import html
    import re

    if text is None:
        text = placeholder

    text = str(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) == 0:
        text = placeholder

    return text


#
#   DOT-based visualization (with square nodes and orthogonal edges)
#

def convert_crawl_results_to_dot(crawl_results: dict, filename_out_prefix: str, crawl_direction: str = "auto"):
    import html
    import subprocess
    import textwrap
    from collections import defaultdict, deque

    crawl_results = _normalize_crawl_results_for_visualization(crawl_results, crawl_direction=crawl_direction)
    crawl_direction = crawl_results.get("visualization_crawl_direction", "forward")

    root_node = crawl_results.get("root_node", None)
    nodes = crawl_results.get("nodes", {}) or {}
    edges = crawl_results.get("edges", []) or []

    def wrap_html(text: str, width: int = 50, placeholder: str = "(none)") -> str:
        text = _sanitize_label_text(text, placeholder=placeholder)
        lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]
        return "<BR/>".join(html.escape(line) for line in lines)

    def escape_dot_id(text: str) -> str:
        return str(text).replace("\\", "\\\\").replace('"', '\\"')

    def compute_min_depths():
        depths = {node_id: None for node_id in nodes.keys()}
        if root_node is None or root_node not in nodes:
            return depths

        parent_to_children = defaultdict(list)
        for edge in edges:
            parent_id = edge.get("contribution_id", None)
            child_id = edge.get("used_by_contribution_id", None)
            if parent_id is None or child_id is None:
                continue
            if parent_id in nodes and child_id in nodes:
                parent_to_children[parent_id].append(child_id)

        q = deque([root_node])
        depths[root_node] = 0

        while len(q) > 0:
            parent_id = q.popleft()
            parent_depth = depths[parent_id]
            for child_id in parent_to_children.get(parent_id, []):
                new_depth = parent_depth + 1
                if depths.get(child_id, None) is None or new_depth < depths[child_id]:
                    depths[child_id] = new_depth
                    q.append(child_id)

        return depths

    def depth_to_fillcolor(depth):
        palette = {
            0: "#fff2b2",
            1: "#fde0c5",
            2: "#f9d5e5",
            3: "#e3d5ff",
            4: "#d6eaff",
            5: "#d9f7d6",
            6: "#eeeeee",
        }
        if depth is None:
            return "#ffffff"
        return palette.get(depth, "#f5f5f5")

    depths = compute_min_depths()

    dot_str = """digraph G {
    graph [
        rankdir=TB,
        splines=ortho,
        overlap=false,
        concentrate=true,
        nodesep=0.7,
        ranksep=1.1,
        pad=0.2
    ];
    node [
        shape=box,
        style="rounded,filled",
        color="gray40",
        fontname="Helvetica"
    ];
    edge [
        color="gray55",
        penwidth=1.2,
        arrowsize=0.8
    ];
"""

    for contribution_id, node in nodes.items():
        contribution_obj = node.get("contribution_obj", {}) or {}

        contribution_name = contribution_obj.get("name") or "No contribution name available."
        description = contribution_obj.get("description") or "No description available."
        paper_title = node.get("paper_title") or "No paper title available."

        contribution_id_html = wrap_html(contribution_id, 50, "(missing id)")
        contribution_name_html = wrap_html(contribution_name, 50, "(no contribution name)")
        description_html = wrap_html(description, 50, "(no description)")
        paper_title_html = wrap_html(paper_title, 50, "(no paper title)")

        label = f"""
<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="4">
  <TR><TD><FONT POINT-SIZE="10" COLOR="gray40">{contribution_id_html}</FONT></TD></TR>
  <TR><TD><B><FONT POINT-SIZE="14">{contribution_name_html}</FONT></B></TD></TR>
  <TR><TD><I><FONT POINT-SIZE="11" COLOR="gray25">{paper_title_html}</FONT></I></TD></TR>
  <TR><TD><FONT POINT-SIZE="11">{description_html}</FONT></TD></TR>
</TABLE>
""".strip()

        contribution_id_escaped = escape_dot_id(contribution_id)
        fillcolor = depth_to_fillcolor(depths.get(contribution_id, None))

        if contribution_id == root_node:
            dot_str += f'    "{contribution_id_escaped}" [label=<{label}>, style="rounded,filled,bold", penwidth=2.5, fillcolor="{fillcolor}"];\n'
        else:
            dot_str += f'    "{contribution_id_escaped}" [label=<{label}>, fillcolor="{fillcolor}"];\n'

    seen_edges = set()
    for edge in edges:
        parent_id = edge.get("contribution_id", None)
        child_id = edge.get("used_by_contribution_id", None)
        if parent_id is None or child_id is None:
            continue
        if parent_id not in nodes or child_id not in nodes:
            continue

        edge_key = (parent_id, child_id)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        parent_id_escaped = escape_dot_id(parent_id)
        child_id_escaped = escape_dot_id(child_id)

        # if (crawl_direction == "forward"):
        #     dot_str += f'    "{parent_id_escaped}" -> "{child_id_escaped}";\n'
        # else:
        #     dot_str += f'    "{parent_id_escaped}" <- "{child_id_escaped}";\n'

        if (crawl_direction == "forward"):
            dot_str += f'    "{parent_id_escaped}" -> "{child_id_escaped}" [dir=back];\n'
        else:
            dot_str += f'    "{parent_id_escaped}" -> "{child_id_escaped}";\n'

    dot_str += "}\n"

    dot_path = filename_out_prefix + ".dot"
    output_pdf = filename_out_prefix + ".pdf"

    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_str)

    subprocess.run(["dot", "-Tpdf", dot_path, "-o", output_pdf], check=True)

    reachable_count = sum(1 for depth in depths.values() if depth is not None)
    print(f"DOT graph saved to {output_pdf} ({reachable_count}/{len(nodes)} nodes reachable; direction={crawl_results.get('visualization_crawl_direction')})")

    return {
        "dot_path": dot_path,
        "output_pdf": output_pdf,
        "depths": depths,
        "visualization_crawl_direction": crawl_results.get("visualization_crawl_direction"),
    }


#
#   DOT-based visualization (that also labels edges with intermediate nodes)
#
def convert_crawl_results_to_dot_with_edge_nodes(crawl_results: dict, filename_out_prefix: str, crawl_direction: str = "auto", edge_label_width: int = 54, edge_label_max_chars: int | None = 900, group_parallel_edges: bool = True, semantic_arrowheads: bool = True):
    import html
    import re
    import subprocess
    import textwrap
    from collections import OrderedDict, defaultdict, deque

    root_node = crawl_results.get("root_node", None)
    nodes = dict(crawl_results.get("nodes", {}) or {})
    raw_edges = list(crawl_results.get("edges", []) or [])

    def sanitize_label_text(text, placeholder: str = "") -> str:
        try:
            return _sanitize_label_text(text, placeholder=placeholder)
        except NameError:
            text = "" if text is None else str(text)
            text = html.unescape(text)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                text = placeholder
            return text

    def wrap_html(text, width: int = 50, placeholder: str = "(none)") -> str:
        text = sanitize_label_text(text, placeholder=placeholder)
        if edge_label_max_chars is not None and len(text) > edge_label_max_chars:
            text = text[:max(0, edge_label_max_chars - 1)].rstrip() + "…"
        lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]
        return "<BR ALIGN=\"LEFT\"/>".join(html.escape(line) for line in lines)

    def wrap_node_html(text, width: int = 50, placeholder: str = "(none)") -> str:
        text = sanitize_label_text(text, placeholder=placeholder)
        lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]
        return "<BR/>".join(html.escape(line) for line in lines)

    def escape_dot_id(text: str) -> str:
        return str(text).replace("\\", "\\\\").replace('"', '\\"')

    def get_raw_source_target(edge: dict):
        source_id = edge.get("contribution_id", None)
        target_id = edge.get("used_by_contribution_id", None)
        if target_id is None:
            target_id = edge.get("prerequisite_for_contribution_id", None)
        if target_id is None:
            target_id = edge.get("dependent_contribution_id", None)
        if target_id is None:
            target_id = edge.get("target_contribution_id", None)
        return source_id, target_id

    def detect_crawl_direction() -> str:
        if crawl_direction in {"forward", "backward"}:
            return crawl_direction
        root_as_source = 0
        root_as_target = 0
        for edge in raw_edges:
            source_id, target_id = get_raw_source_target(edge)
            if source_id == root_node and target_id in nodes:
                root_as_source += 1
            if target_id == root_node and source_id in nodes:
                root_as_target += 1
        if root_as_source > root_as_target:
            return "forward"
        if root_as_target > root_as_source:
            return "backward"
        return "forward"

    detected_crawl_direction = detect_crawl_direction()

    normalized_edges = []
    for edge_index, edge in enumerate(raw_edges):
        source_id, target_id = get_raw_source_target(edge)
        if source_id is None or target_id is None:
            continue
        if source_id not in nodes or target_id not in nodes:
            continue
        if detected_crawl_direction == "forward":
            parent_id = source_id
            child_id = target_id
        else:
            parent_id = target_id
            child_id = source_id
        edge_copy = dict(edge)
        edge_copy["__edge_index"] = edge_index
        edge_copy["__visual_parent_id"] = parent_id
        edge_copy["__visual_child_id"] = child_id
        edge_copy["__raw_source_id"] = source_id
        edge_copy["__raw_target_id"] = target_id
        normalized_edges.append(edge_copy)

    def compute_min_depths():
        depths = {node_id: None for node_id in nodes.keys()}
        if root_node is None or root_node not in nodes:
            return depths
        parent_to_children = defaultdict(list)
        for edge in normalized_edges:
            parent_id = edge["__visual_parent_id"]
            child_id = edge["__visual_child_id"]
            parent_to_children[parent_id].append(child_id)
        q = deque([root_node])
        depths[root_node] = 0
        while len(q) > 0:
            parent_id = q.popleft()
            parent_depth = depths[parent_id]
            for child_id in parent_to_children.get(parent_id, []):
                new_depth = parent_depth + 1
                if depths.get(child_id, None) is None or new_depth < depths[child_id]:
                    depths[child_id] = new_depth
                    q.append(child_id)
        return depths

    def depth_to_fillcolor(depth):
        palette = {
            0: "#fff2b2",
            1: "#fde0c5",
            2: "#f9d5e5",
            3: "#e3d5ff",
            4: "#d6eaff",
            5: "#d9f7d6",
            6: "#eeeeee",
        }
        if depth is None:
            return "#ffffff"
        return palette.get(depth, "#f5f5f5")

    def unique_nonempty_texts(values):
        seen = set()
        out = []
        for value in values:
            text = sanitize_label_text(value, placeholder="")
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def strength_fillcolor(strengths):
        strengths_lc = {str(x).lower() for x in strengths}
        if "strong" in strengths_lc and "weak" not in strengths_lc:
            return "#e8f4ea"
        if "weak" in strengths_lc and "strong" not in strengths_lc:
            return "#fff6df"
        if len(strengths_lc) > 0:
            return "#eef2f7"
        return "#f7f7f7"

    def make_edge_node_label(edge_group):
        strengths = []
        depths = []
        descriptions = []
        explanations = []
        for edge in edge_group:
            edge_strengths = edge.get("strengths", [])
            if isinstance(edge_strengths, list):
                strengths.extend(edge_strengths)
            elif edge_strengths is not None:
                strengths.append(edge_strengths)
            if edge.get("depth", None) is not None:
                depths.append(edge.get("depth"))
            descriptions.append(edge.get("prerequisite_description", None))
            descriptions.append(edge.get("description", None))
            explanations.append(edge.get("prerequisite_explanation", None))
            explanations.append(edge.get("explanation", None))

        strengths = sorted({str(x) for x in strengths if str(x).strip()})
        depths = sorted({str(x) for x in depths if str(x).strip()})
        descriptions = unique_nonempty_texts(descriptions)
        explanations = unique_nonempty_texts(explanations)

        rows = []
        header_bits = []
        if strengths:
            header_bits.append("strength: " + ", ".join(strengths))
        if depths:
            header_bits.append("depth: " + ", ".join(depths))
        if not header_bits:
            header_bits.append("relationship")

        rows.append(f'<TR><TD ALIGN="LEFT"><B><FONT POINT-SIZE="10">Prerequisite Edge Description</FONT></B></TD></TR>')

        if descriptions:
            for i, description in enumerate(descriptions, start=1):
                label = "Description" if len(descriptions) == 1 else f"Description {i}"
                rows.append(f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="gray35"><B>{html.escape(label)}:</B></FONT><BR ALIGN="LEFT"/><FONT POINT-SIZE="9">{wrap_html(description, edge_label_width, "")}</FONT></TD></TR>')

        if explanations:
            for i, explanation in enumerate(explanations, start=1):
                label = "Explanation" if len(explanations) == 1 else f"Explanation {i}"
                rows.append(f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="gray35"><B>{html.escape(label)}:</B></FONT><BR ALIGN="LEFT"/><FONT POINT-SIZE="9">{wrap_html(explanation, edge_label_width, "")}</FONT></TD></TR>')

        if len(rows) == 1:
            rows.append('<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="gray40">No edge metadata available.</FONT></TD></TR>')

        rows.append(f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10">{html.escape("; ".join(header_bits))}</FONT></TD></TR>')

        return "<\n<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLPADDING=\"4\">\n" + "\n".join(rows) + "\n</TABLE>\n>"

    depths = compute_min_depths()

    dot_str = """digraph G {
    graph [
        rankdir=TB,
        splines=ortho,
        overlap=false,
        concentrate=false,
        nodesep=0.75,
        ranksep=1.25,
        pad=0.2
    ];
    node [
        shape=box,
        style="rounded,filled",
        color="gray40",
        fontname="Helvetica"
    ];
    edge [
        color="gray55",
        penwidth=1.2,
        arrowsize=0.8
    ];
"""

    for contribution_id, node in nodes.items():
        contribution_obj = node.get("contribution_obj", {}) or {}
        contribution_name = contribution_obj.get("name") or "No contribution name available."
        description = contribution_obj.get("description") or "No description available."
        paper_title = node.get("paper_title") or "No paper title available."

        contribution_id_html = wrap_node_html(contribution_id, 50, "(missing id)")
        contribution_name_html = wrap_node_html(contribution_name, 50, "(no contribution name)")
        description_html = wrap_node_html(description, 50, "(no description)")
        paper_title_html = wrap_node_html(paper_title, 50, "(no paper title)")

        label = f"""
<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="4">
  <TR><TD><FONT POINT-SIZE="10" COLOR="gray40">{contribution_id_html}</FONT></TD></TR>
  <TR><TD><B><FONT POINT-SIZE="14">{contribution_name_html}</FONT></B></TD></TR>
  <TR><TD><I><FONT POINT-SIZE="11" COLOR="gray25">{paper_title_html}</FONT></I></TD></TR>
  <TR><TD><FONT POINT-SIZE="11">{description_html}</FONT></TD></TR>
</TABLE>
""".strip()

        contribution_id_escaped = escape_dot_id(contribution_id)
        fillcolor = depth_to_fillcolor(depths.get(contribution_id, None))
        if contribution_id == root_node:
            dot_str += f'    "{contribution_id_escaped}" [label=<{label}>, style="rounded,filled,bold", penwidth=2.5, fillcolor="{fillcolor}"];\n'
        else:
            dot_str += f'    "{contribution_id_escaped}" [label=<{label}>, fillcolor="{fillcolor}"];\n'

    grouped_edges = OrderedDict()
    for edge in normalized_edges:
        parent_id = edge["__visual_parent_id"]
        child_id = edge["__visual_child_id"]
        if group_parallel_edges:
            edge_key = (parent_id, child_id)
        else:
            edge_key = (parent_id, child_id, edge["__edge_index"])
        grouped_edges.setdefault(edge_key, []).append(edge)

    for edge_node_index, (edge_key, edge_group) in enumerate(grouped_edges.items()):
        parent_id = edge_group[0]["__visual_parent_id"]
        child_id = edge_group[0]["__visual_child_id"]
        edge_node_id = f"__edge_metadata_{edge_node_index}"

        strengths = []
        for edge in edge_group:
            edge_strengths = edge.get("strengths", [])
            if isinstance(edge_strengths, list):
                strengths.extend(edge_strengths)
            elif edge_strengths is not None:
                strengths.append(edge_strengths)

        edge_label = make_edge_node_label(edge_group)
        edge_node_id_escaped = escape_dot_id(edge_node_id)
        parent_id_escaped = escape_dot_id(parent_id)
        child_id_escaped = escape_dot_id(child_id)
        edge_fillcolor = strength_fillcolor(strengths)

        dot_str += f'    "{edge_node_id_escaped}" [label={edge_label}, shape=box, style="rounded,filled,dashed", color="gray60", fillcolor="{edge_fillcolor}", fontname="Helvetica", margin="0.04,0.03"];\n'

        if semantic_arrowheads:
            if detected_crawl_direction == "forward":
                dot_str += f'    "{parent_id_escaped}" -> "{edge_node_id_escaped}" [dir=back, weight=3];\n'
                dot_str += f'    "{edge_node_id_escaped}" -> "{child_id_escaped}" [dir=none, weight=3];\n'
            else:
                dot_str += f'    "{parent_id_escaped}" -> "{edge_node_id_escaped}" [dir=none, weight=3];\n'
                dot_str += f'    "{edge_node_id_escaped}" -> "{child_id_escaped}" [weight=3];\n'
        else:
            dot_str += f'    "{parent_id_escaped}" -> "{edge_node_id_escaped}" [weight=3];\n'
            dot_str += f'    "{edge_node_id_escaped}" -> "{child_id_escaped}" [weight=3];\n'

    dot_str += "}\n"

    dot_path = filename_out_prefix + ".dot"
    output_pdf = filename_out_prefix + ".pdf"

    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_str)

    subprocess.run(["dot", "-Tpdf", dot_path, "-o", output_pdf], check=True)

    reachable_count = sum(1 for depth in depths.values() if depth is not None)
    print(f"DOT graph with edge nodes saved to {output_pdf} ({reachable_count}/{len(nodes)} contribution nodes reachable; {len(grouped_edges)} edge-label nodes; direction={detected_crawl_direction})")

    return {
        "dot_path": dot_path,
        "output_pdf": output_pdf,
        "depths": depths,
        "visualization_crawl_direction": detected_crawl_direction,
        "num_edge_label_nodes": len(grouped_edges),
    }



#
#   Radial force-directed visualization (with circular nodes and straight edges)
#
def export_crawl_results_to_radial_tree_svg(crawl_results: dict, filename_out: str, node_diameter_px: int = 110, min_gap_px: int = 26, radial_step_px: int = 140, angular_slack: float = 0.12, margin_px: int = 80, contraction_iterations: int = 1400, gravity_k: float = 0.055, spring_k: float = 0.020, node_repulsion_k: float = 18.0, edge_repulsion_k: float = 14.0, max_move_px: float = 8.0, cooling: float = 0.996, crossing_check_every_n_iterations: int = 1, node_edge_gap_px: float | None = None, node_center_min_distance_px: float | None = None, metadata_line_mode: str = "paper_title", min_children: int = 0, min_thresh: int = 1, root_fill_color: str = "#b5a6c9", trim_root_leaf_summary_nodes: bool = False, crawl_direction: str = "auto"):
    import math
    import html
    import textwrap
    import os
    import re
    from collections import defaultdict, deque
    from tqdm import tqdm

    root_node = crawl_results.get("root_node", None)
    nodes = crawl_results.get("nodes", {})
    edges = crawl_results.get("edges", [])

    if (root_node is None) or (root_node not in nodes):
        raise RuntimeError("crawl_results must contain a root_node that exists in nodes")

    if metadata_line_mode not in ["none", "paper_title", "first_author_year"]:
        raise RuntimeError("metadata_line_mode must be one of: 'none', 'paper_title', 'first_author_year'")

    def get_raw_edge_endpoint_ids(edge: dict):
        source_id = edge.get("contribution_id", None)
        target_id = edge.get("used_by_contribution_id", None)

        # Older/comments/examples sometimes used this name for the same dependent side.
        if target_id is None:
            target_id = edge.get("prerequisite_for_contribution_id", None)

        return source_id, target_id

    def count_reachable_nodes_for_direction(direction: str) -> int:
        parent_to_children_for_direction = defaultdict(set)

        for edge in edges:
            source_id, target_id = get_raw_edge_endpoint_ids(edge)
            if (source_id is None) or (target_id is None):
                continue
            if (source_id not in nodes) or (target_id not in nodes):
                continue

            if direction == "forward":
                parent_id = source_id
                child_id = target_id
            elif direction == "backward":
                parent_id = target_id
                child_id = source_id
            else:
                raise RuntimeError("direction must be 'forward' or 'backward'")

            parent_to_children_for_direction[parent_id].add(child_id)

        seen_for_direction = set([root_node])
        q_for_direction = deque([root_node])

        while len(q_for_direction) > 0:
            parent_id = q_for_direction.popleft()
            for child_id in parent_to_children_for_direction.get(parent_id, set()):
                if child_id in seen_for_direction:
                    continue
                seen_for_direction.add(child_id)
                q_for_direction.append(child_id)

        return len(seen_for_direction)

    def detect_crawl_direction() -> str:
        if crawl_direction not in ["auto", "forward", "backward"]:
            raise RuntimeError("crawl_direction must be one of: 'auto', 'forward', 'backward'")

        if crawl_direction != "auto":
            return crawl_direction

        root_as_source = 0
        root_as_target = 0
        depth0_root_as_source = 0
        depth0_root_as_target = 0

        for edge in edges:
            source_id, target_id = get_raw_edge_endpoint_ids(edge)
            if (source_id is None) or (target_id is None):
                continue
            if (source_id not in nodes) or (target_id not in nodes):
                continue

            if source_id == root_node:
                root_as_source += 1
            if target_id == root_node:
                root_as_target += 1

            edge_depth = edge.get("depth", None)
            if str(edge_depth) == "0":
                if source_id == root_node:
                    depth0_root_as_source += 1
                if target_id == root_node:
                    depth0_root_as_target += 1

        # Best signal for these crawl outputs: depth-0 edges touch the root.
        # Forward crawl:  root appears as contribution_id.
        # Backward crawl: root appears as used_by_contribution_id / prerequisite_for_contribution_id.
        if depth0_root_as_source > depth0_root_as_target:
            return "forward"
        if depth0_root_as_target > depth0_root_as_source:
            return "backward"

        # Good fallback when depth is missing.
        if root_as_source > root_as_target:
            return "forward"
        if root_as_target > root_as_source:
            return "backward"

        # Last resort: choose the orientation that reaches more nodes from the root.
        forward_reachable = count_reachable_nodes_for_direction("forward")
        backward_reachable = count_reachable_nodes_for_direction("backward")

        if backward_reachable > forward_reachable:
            return "backward"
        return "forward"

    detected_crawl_direction = detect_crawl_direction()
    print("Radial visualization direction: " + str(detected_crawl_direction))

    raw_parent_to_children = defaultdict(set)

    for edge in edges:
        source_id, target_id = get_raw_edge_endpoint_ids(edge)

        if (source_id is None) or (target_id is None):
            continue
        if (source_id not in nodes) or (target_id not in nodes):
            continue

        if detected_crawl_direction == "forward":
            parent_id = source_id
            child_id = target_id
        else:
            parent_id = target_id
            child_id = source_id

        raw_parent_to_children[parent_id].add(child_id)

    hidden_count_by_parent = defaultdict(int)

    if (min_children > 0):
        nodes_before = len(nodes)
        edges_before = len(edges)

        keep_nodes = set()
        for node_id in nodes.keys():
            if (node_id == root_node):
                keep_nodes.add(node_id)
                continue

            num_children = len(raw_parent_to_children.get(node_id, set()))
            if (num_children >= min_children):
                keep_nodes.add(node_id)

        def count_hidden_subtree(node_id: str, seen: set) -> int:
            if node_id in seen:
                return 0
            if node_id not in nodes:
                return 0

            seen.add(node_id)
            total = 1

            for child_id in raw_parent_to_children.get(node_id, set()):
                total += count_hidden_subtree(child_id, seen)

            return total

        for parent_id, child_ids in raw_parent_to_children.items():
            if parent_id not in keep_nodes:
                continue

            seen_hidden_from_parent = set()
            for child_id in child_ids:
                if child_id not in keep_nodes:
                    hidden_count_by_parent[parent_id] += count_hidden_subtree(child_id, seen_hidden_from_parent)

        nodes = {
            node_id: node
            for node_id, node in nodes.items()
            if node_id in keep_nodes
        }

        raw_parent_to_children_pruned = defaultdict(set)
        for parent_id, child_ids in raw_parent_to_children.items():
            if parent_id not in nodes:
                continue
            for child_id in child_ids:
                if child_id in nodes:
                    raw_parent_to_children_pruned[parent_id].add(child_id)

        raw_parent_to_children = raw_parent_to_children_pruned
        edges_after = sum(len(child_ids) for child_ids in raw_parent_to_children.values())
        hidden_total = sum(hidden_count_by_parent.values())

        print(f"Pruned radial SVG export with min_children={min_children}: nodes {nodes_before} -> {len(nodes)}, edges {edges_before} -> {edges_after}, hidden contributions found={hidden_total}")

    parent_to_children = defaultdict(list)
    child_to_parent = {}

    q = deque([root_node])
    assigned_nodes = set([root_node])

    while len(q) > 0:
        parent_id = q.popleft()
        child_ids = sorted(list(raw_parent_to_children.get(parent_id, set())), key=lambda x: str(x))

        for child_id in child_ids:
            if child_id in assigned_nodes:
                continue
            if child_id not in nodes:
                continue

            assigned_nodes.add(child_id)
            child_to_parent[child_id] = parent_id
            parent_to_children[parent_id].append(child_id)
            q.append(child_id)

    root_leaf_nodes_trimmed = 0
    root_leaf_contributions_represented = 0

    if (trim_root_leaf_summary_nodes == True) and (min_children > 0):
        root_children = list(parent_to_children.get(root_node, []))
        root_children_kept = []

        for child_id in root_children:
            visible_child_count = len(parent_to_children.get(child_id, []))
            if (visible_child_count > 0):
                root_children_kept.append(child_id)
                continue

            hidden_count = hidden_count_by_parent.get(child_id, 0)
            represented_count = 1 + hidden_count

            hidden_count_by_parent[root_node] += represented_count
            hidden_count_by_parent[child_id] = 0

            if child_id in nodes:
                nodes.pop(child_id, None)
            if child_id in child_to_parent:
                child_to_parent.pop(child_id, None)
            if child_id in assigned_nodes:
                assigned_nodes.remove(child_id)
            if child_id in parent_to_children:
                parent_to_children.pop(child_id, None)

            root_leaf_nodes_trimmed += 1
            root_leaf_contributions_represented += represented_count

        parent_to_children[root_node] = root_children_kept

        if (root_leaf_nodes_trimmed > 0):
            print(f"Trimmed {root_leaf_nodes_trimmed} root leaf-summary branches, adding {root_leaf_contributions_represented} represented contributions to the root hidden-summary node")

    hidden_summary_nodes_added = 0
    hidden_contributions_represented = 0

    if (min_children > 0):
        for parent_id in sorted(list(assigned_nodes), key=lambda x: str(x)):
            hidden_count = hidden_count_by_parent.get(parent_id, 0)
            if (hidden_count <= 0):
                continue
            if (hidden_count < min_thresh):
                continue

            hidden_node_id = f"{parent_id}.__hidden_contributions"
            hidden_idx = 1
            while hidden_node_id in nodes:
                hidden_node_id = f"{parent_id}.__hidden_contributions_{hidden_idx}"
                hidden_idx += 1

            label_text = f"{hidden_count} hidden contributions"
            if hidden_count == 1:
                label_text = "1 hidden contribution"

            nodes[hidden_node_id] = {
                "contribution_id": hidden_node_id,
                "paper_title": None,
                "paper_corpus_id": None,
                "is_hidden_summary": True,
                "hidden_count": hidden_count,
                "contribution_obj": {
                    "name": label_text,
                    "description": label_text,
                },
            }

            parent_to_children[parent_id].append(hidden_node_id)
            child_to_parent[hidden_node_id] = parent_id
            assigned_nodes.add(hidden_node_id)

            hidden_summary_nodes_added += 1
            hidden_contributions_represented += hidden_count

        if hidden_summary_nodes_added > 0:
            print(f"Added {hidden_summary_nodes_added} hidden-summary nodes representing {hidden_contributions_represented} hidden contributions with min_thresh={min_thresh}")

    for node_id in nodes.keys():
        parent_to_children[node_id] = sorted(parent_to_children[node_id], key=lambda x: str(x))

    seen = set()

    def dfs_check(node_id: str):
        if node_id in seen:
            return
        seen.add(node_id)
        for child_id in parent_to_children[node_id]:
            dfs_check(child_id)

    dfs_check(root_node)
    reachable_nodes = set(seen)

    print(f"Converted crawl graph to radial tree: nodes {len(nodes)} -> {len(reachable_nodes)}, edges {len(edges)} -> {len(child_to_parent)}")

    depth_by_node = {root_node: 0}

    def compute_depths(node_id: str):
        for child_id in parent_to_children[node_id]:
            depth_by_node[child_id] = depth_by_node[node_id] + 1
            compute_depths(child_id)

    compute_depths(root_node)

    leaf_span_px = float(node_diameter_px + min_gap_px)
    subtree_span_px = {}

    def compute_subtree_span(node_id: str) -> float:
        children = parent_to_children[node_id]
        if len(children) == 0:
            subtree_span_px[node_id] = leaf_span_px
            return subtree_span_px[node_id]

        total = 0.0
        for child_id in children:
            total += compute_subtree_span(child_id)

        total += float(min_gap_px) * float(max(0, len(children) - 1))
        subtree_span_px[node_id] = max(total, leaf_span_px)
        return subtree_span_px[node_id]

    compute_subtree_span(root_node)

    max_depth = max(depth_by_node.values()) if len(depth_by_node) > 0 else 0
    radius_by_depth = {0: 0.0}

    for depth in range(1, max_depth + 1):
        nodes_at_depth = [n for n in reachable_nodes if depth_by_node[n] == depth]
        if len(nodes_at_depth) == 0:
            radius_by_depth[depth] = radius_by_depth[depth - 1] + float(radial_step_px)
            continue

        required_circumference_px = sum(subtree_span_px[n] for n in nodes_at_depth) + float(min_gap_px) * float(len(nodes_at_depth))
        radius_from_circumference = required_circumference_px / (2.0 * math.pi)
        radius_from_previous = radius_by_depth[depth - 1] + float(radial_step_px)
        radius_by_depth[depth] = max(radius_from_circumference, radius_from_previous)

    angle_by_node = {root_node: 0.0}
    pos_by_node = {root_node: [0.0, 0.0]}

    def wrap_angle(theta: float) -> float:
        return math.atan2(math.sin(theta), math.cos(theta))

    def layout_children(parent_id: str, theta_start: float, theta_end: float):
        children = parent_to_children[parent_id]
        if len(children) == 0:
            return

        child_depth = depth_by_node[parent_id] + 1
        radius = radius_by_depth[child_depth]

        usable_start = theta_start + angular_slack
        usable_end = theta_end - angular_slack
        if usable_end <= usable_start:
            usable_start = theta_start
            usable_end = theta_end

        total_span = 0.0
        for child_id in children:
            total_span += subtree_span_px[child_id]
        total_span += float(min_gap_px) * float(max(0, len(children) - 1))

        child_intervals = []
        current_linear = 0.0
        for child_id in children:
            child_linear_width = subtree_span_px[child_id]
            start_linear = current_linear
            end_linear = current_linear + child_linear_width
            child_intervals.append((child_id, start_linear, end_linear))
            current_linear = end_linear + float(min_gap_px)

        total_arc = usable_end - usable_start
        parent_angle = angle_by_node[parent_id]

        interval_centers = []
        for child_id, start_linear, end_linear in child_intervals:
            center_linear = 0.5 * (start_linear + end_linear)
            frac = center_linear / total_span if total_span > 0 else 0.5
            theta = usable_start + frac * total_arc
            interval_centers.append((child_id, theta))

        center_shift = 0.0
        if len(interval_centers) > 0:
            nearest_idx = min(range(len(interval_centers)), key=lambda i: abs(wrap_angle(interval_centers[i][1] - parent_angle)))
            chosen_theta = interval_centers[nearest_idx][1]
            center_shift = parent_angle - chosen_theta

        for child_id, start_linear, end_linear in child_intervals:
            start_frac = start_linear / total_span if total_span > 0 else 0.0
            end_frac = end_linear / total_span if total_span > 0 else 1.0

            child_theta_start = usable_start + start_frac * total_arc + center_shift
            child_theta_end = usable_start + end_frac * total_arc + center_shift
            child_theta = 0.5 * (child_theta_start + child_theta_end)

            x = radius * math.cos(child_theta)
            y = radius * math.sin(child_theta)

            angle_by_node[child_id] = child_theta
            pos_by_node[child_id] = [x, y]

            layout_children(child_id, child_theta_start, child_theta_end)

    layout_children(root_node, -math.pi, math.pi)

    node_radius = float(node_diameter_px) / 2.0

    if node_edge_gap_px is None:
        node_edge_gap_px = node_radius

    if node_center_min_distance_px is None:
        node_center_min_distance_px = float(node_diameter_px * 2.5)

    edge_list = []
    for child_id, parent_id in child_to_parent.items():
        if child_id in reachable_nodes and parent_id in reachable_nodes:
            edge_list.append((child_id, parent_id))

    all_nodes_sorted = sorted(list(reachable_nodes), key=lambda x: (depth_by_node[x], str(x)))
    movable_nodes = [n for n in all_nodes_sorted if n != root_node]

    def point_segment_projection(px: float, py: float, ax: float, ay: float, bx: float, by: float):
        dx = bx - ax
        dy = by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 1e-12:
            return ax, ay, 0.0
        t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        qx = ax + t * dx
        qy = ay + t * dy
        return qx, qy, t

    def orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def segments_intersect(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float):
        o1 = orient(ax, ay, bx, by, cx, cy)
        o2 = orient(ax, ay, bx, by, dx, dy)
        o3 = orient(cx, cy, dx, dy, ax, ay)
        o4 = orient(cx, cy, dx, dy, bx, by)
        return (((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)))

    def crossing_count_for_node(node_id: str) -> int:
        incident_edges = []
        if node_id in child_to_parent:
            incident_edges.append((node_id, child_to_parent[node_id]))
        for child_id in parent_to_children.get(node_id, []):
            if child_id in reachable_nodes:
                incident_edges.append((child_id, node_id))

        count = 0
        for a, b in incident_edges:
            ax, ay = pos_by_node[a]
            bx, by = pos_by_node[b]
            for c, d in edge_list:
                if len({a, b, c, d}) < 4:
                    continue
                cx, cy = pos_by_node[c]
                dx, dy = pos_by_node[d]
                if segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                    count += 1
        return count

    d3_palette = [
        "#ff595e",
        "#ffca3a",
        "#8ac926",
        "#1982c4",
        "#6a4c93",
    ]

    branch_root_children = sorted(parent_to_children.get(root_node, []), key=lambda x: str(x))
    root_child_to_branch_color = {}
    for idx, child_id in enumerate(branch_root_children):
        root_child_to_branch_color[child_id] = d3_palette[idx % len(d3_palette)]

    node_branch_root_child = {root_node: None}

    def assign_branch(node_id: str, current_branch_root_child):
        node_branch_root_child[node_id] = current_branch_root_child
        for child_id in parent_to_children.get(node_id, []):
            if node_id == root_node:
                assign_branch(child_id, child_id)
            else:
                assign_branch(child_id, current_branch_root_child)

    assign_branch(root_node, None)

    def hex_to_rgb(hex_color: str):
        hex_color = hex_color.lstrip("#")
        return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))

    def rgb_to_hex(r: int, g: int, b: int):
        return "#{:02x}{:02x}{:02x}".format(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    def mix_with_white(hex_color: str, t: float):
        r, g, b = hex_to_rgb(hex_color)
        rr = int(round(r + (255 - r) * t))
        gg = int(round(g + (255 - g) * t))
        bb = int(round(b + (255 - b) * t))
        return rgb_to_hex(rr, gg, bb)

    def node_fill(node_id: str) -> str:
        node_obj = nodes.get(node_id, {})
        if node_obj.get("is_hidden_summary", False):
            return "#f4f4f4"

        if node_id == root_node:
            return root_fill_color

        branch_child = node_branch_root_child.get(node_id, None)
        if branch_child is None:
            return "#eeeeee"

        base = root_child_to_branch_color.get(branch_child, "#7f7f7f")
        depth = depth_by_node.get(node_id, 1)
        lighten = min(0.50 + 0.20 * float(max(0, depth - 1)), 0.96)
        return mix_with_white(base, lighten)

    def edge_stroke_color(child_id: str) -> str:
        branch_child = node_branch_root_child.get(child_id, None)
        if branch_child is None:
            return "#888888"
        base = root_child_to_branch_color.get(branch_child, "#7f7f7f")
        depth = depth_by_node.get(child_id, 1)
        lighten = min(0.50 + 0.10 * float(max(0, depth - 1)), 0.90)
        return mix_with_white(base, lighten)

    def edge_stroke_width(depth: int) -> float:
        if depth <= 1:
            return 8
        elif depth == 2:
            return 6.0
        elif depth == 3:
            return 4.0
        elif depth == 4:
            return 3.0
        else:
            return 3.0

    def edge_stroke_opacity(depth: int) -> float:
        if depth <= 1:
            return 0.95
        elif depth == 2:
            return 0.90
        elif depth == 3:
            return 0.85
        elif depth == 4:
            return 0.80
        else:
            return 0.75

    def write_svg_file(filename_out_current: str):
        min_x = min(pos_by_node[node_id][0] for node_id in reachable_nodes) - node_radius - margin_px
        max_x = max(pos_by_node[node_id][0] for node_id in reachable_nodes) + node_radius + margin_px
        min_y = min(pos_by_node[node_id][1] for node_id in reachable_nodes) - node_radius - margin_px
        max_y = max(pos_by_node[node_id][1] for node_id in reachable_nodes) + node_radius + margin_px

        width = max_x - min_x
        height = max_y - min_y

        def world_to_svg(x: float, y: float):
            sx = x - min_x
            sy = max_y - y
            return sx, sy

        def sanitize_metadata_text(text: str) -> str:
            if text is None:
                return ""

            text = str(text)
            text = html.unescape(text)
            text = re.sub(r"<[^>]+>", "", text)
            text = text.replace("**", "")
            text = text.replace("__", "")
            text = re.sub(r"\s+", " ", text).strip()

            return text

        def get_metadata_line(node_id: str, node_obj: dict) -> str:
            if metadata_line_mode == "none":
                return ""

            if metadata_line_mode == "paper_title":
                paper_title = sanitize_metadata_text(node_obj.get("paper_title", None))
                if (paper_title is None) or (len(str(paper_title).strip()) == 0):
                    return ""
                return paper_title

            if metadata_line_mode == "first_author_year":
                paper_first_author = node_obj.get("paper_first_author", None)
                paper_year = node_obj.get("paper_year", None)

                if isinstance(paper_first_author, dict):
                    last_name = paper_first_author.get("last_name", None)
                else:
                    last_name = None

                if (last_name is None) or (len(str(last_name).strip()) == 0):
                    return ""

                if paper_year is None:
                    return f"{str(last_name).strip()} et al."
                return f"{str(last_name).strip()} et al. ({paper_year})"

            return ""

        def make_node_label(node_id: str, node_obj: dict):
            if node_obj.get("is_hidden_summary", False):
                hidden_count = node_obj.get("hidden_count", 0)
                if hidden_count == 1:
                    main_lines = ["1 hidden", "contribution"]
                else:
                    main_lines = [f"{hidden_count} hidden", "contributions"]

                return {
                    "main_lines": main_lines,
                    "metadata_lines": [],
                }

            contribution_obj = node_obj.get("contribution_obj", {}) or {}
            contribution_name = contribution_obj.get("name", None)

            if (contribution_name is None) or (len(str(contribution_name).strip()) == 0):
                text = str(node_id)
            else:
                text = str(contribution_name).strip()

            main_lines = textwrap.wrap(text, width=22, break_long_words=False, break_on_hyphens=False)
            if len(main_lines) > 6:
                main_lines = main_lines[:6]
                if len(main_lines[-1]) > 19:
                    main_lines[-1] = main_lines[-1][:19] + "..."
                else:
                    main_lines[-1] = main_lines[-1] + "..."

            metadata_text = get_metadata_line(node_id, node_obj)
            metadata_lines = []
            if len(metadata_text) > 0:
                metadata_lines = textwrap.wrap(metadata_text, width=24, break_long_words=False, break_on_hyphens=False)
                if len(metadata_lines) > 2:
                    metadata_lines = metadata_lines[:2]
                    if len(metadata_lines[-1]) > 21:
                        metadata_lines[-1] = metadata_lines[-1][:21] + "..."
                    else:
                        metadata_lines[-1] = metadata_lines[-1] + "..."

            return {
                "main_lines": main_lines,
                "metadata_lines": metadata_lines,
            }

        svg_lines = []
        svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">')
        svg_lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

        for child_id, parent_id in sorted(child_to_parent.items(), key=lambda x: (depth_by_node[x[0]], angle_by_node.get(x[0], 0.0), str(x[0]))):
            if child_id not in reachable_nodes or parent_id not in reachable_nodes:
                continue

            x1, y1 = pos_by_node[child_id]
            x2, y2 = pos_by_node[parent_id]

            sx1, sy1 = world_to_svg(x1, y1)
            sx2, sy2 = world_to_svg(x2, y2)

            dx = sx2 - sx1
            dy = sy2 - sy1
            dist = math.hypot(dx, dy)

            child_is_hidden_summary = nodes.get(child_id, {}).get("is_hidden_summary", False)
            parent_is_hidden_summary = nodes.get(parent_id, {}).get("is_hidden_summary", False)

            if dist > 1e-6:
                ux = dx / dist
                uy = dy / dist

                if (not child_is_hidden_summary):
                    sx1 += ux * node_radius
                    sy1 += uy * node_radius

                if (not parent_is_hidden_summary):
                    sx2 -= ux * node_radius
                    sy2 -= uy * node_radius

            edge_depth = depth_by_node.get(child_id, 99)
            stroke_w = edge_stroke_width(edge_depth)
            stroke_op = edge_stroke_opacity(edge_depth)
            stroke_color = edge_stroke_color(child_id)

            if child_is_hidden_summary:
                stroke_op = min(stroke_op, 0.55)
                stroke_w = max(2.0, stroke_w * 0.55)

            svg_lines.append(f'<line x1="{sx1:.2f}" y1="{sy1:.2f}" x2="{sx2:.2f}" y2="{sy2:.2f}" stroke="{stroke_color}" stroke-width="{stroke_w:.2f}" stroke-opacity="{stroke_op:.2f}" stroke-linecap="round"/>')

        for node_id in sorted(reachable_nodes, key=lambda x: (depth_by_node[x], angle_by_node.get(x, 0.0), str(x))):
            x, y = pos_by_node[node_id]
            sx, sy = world_to_svg(x, y)
            fill = node_fill(node_id)
            node_obj = nodes.get(node_id, {})
            is_hidden_summary = node_obj.get("is_hidden_summary", False)

            if node_id == root_node:
                stroke_color = "#3f3f3f"
                stroke_width = 3.2
            else:
                branch_child = node_branch_root_child.get(node_id, None)
                if branch_child is None:
                    stroke_color = "#666666"
                else:
                    stroke_color = root_child_to_branch_color.get(branch_child, "#666666")
                stroke_width = 1.8

            if is_hidden_summary:
                square_side = node_diameter_px * 0.82
                rect_x = sx - square_side / 2.0
                rect_y = sy - square_side / 2.0
                svg_lines.append(f'<rect x="{rect_x:.2f}" y="{rect_y:.2f}" width="{square_side:.2f}" height="{square_side:.2f}" rx="8.00" ry="8.00" fill="{fill}" stroke="{stroke_color}" stroke-width="{stroke_width:.2f}" stroke-dasharray="5 4"/>')
            else:
                svg_lines.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="{node_radius:.2f}" fill="{fill}" stroke="{stroke_color}" stroke-width="{stroke_width:.2f}"/>')

            label_obj = make_node_label(node_id, node_obj)
            main_lines = label_obj["main_lines"]
            metadata_lines = label_obj["metadata_lines"]

            main_font_size = 11
            main_line_height = 12
            metadata_font_size = 9
            metadata_line_height = 10
            metadata_gap = 4 if len(metadata_lines) > 0 else 0

            if is_hidden_summary:
                main_font_size = 12
                main_line_height = 13

            total_height = 0.0
            if len(main_lines) > 0:
                total_height += (len(main_lines) - 1) * main_line_height
            if len(metadata_lines) > 0:
                total_height += metadata_gap + len(metadata_lines) * metadata_line_height

            cursor_y = sy - (total_height / 2.0)

            for i, line in enumerate(main_lines):
                escaped = html.escape(line)
                text_y = cursor_y + i * main_line_height
                font_weight = "600"
                fill_color = "#222222"
                if is_hidden_summary:
                    font_weight = "500"
                    fill_color = "#555555"
                svg_lines.append(f'<text x="{sx:.2f}" y="{text_y:.2f}" text-anchor="middle" dominant-baseline="middle" font-family="Helvetica, Arial, sans-serif" font-size="{main_font_size}" font-weight="{font_weight}" fill="{fill_color}">{escaped}</text>')

            if len(main_lines) > 0:
                cursor_y = cursor_y + (len(main_lines) - 1) * main_line_height + metadata_gap + metadata_line_height
            else:
                cursor_y = cursor_y + metadata_line_height

            for i, line in enumerate(metadata_lines):
                escaped = html.escape(line)
                text_y = cursor_y + i * metadata_line_height
                svg_lines.append(f'<text x="{sx:.2f}" y="{text_y:.2f}" text-anchor="middle" dominant-baseline="middle" font-family="Helvetica, Arial, sans-serif" font-size="{metadata_font_size}" font-weight="400" fill="#444444">{escaped}</text>')

        svg_lines.append('</svg>')

        with open(filename_out_current, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_lines) + "\n")

        print(f"Saved SVG to: {filename_out_current}")

        return {
            "width": width,
            "height": height,
        }

    step_cap = float(max_move_px)

    for iter_idx in tqdm(range(contraction_iterations), desc="Contracting"):
        disp = {}
        for node_id in movable_nodes:
            disp[node_id] = [0.0, 0.0]

        for node_id in movable_nodes:
            x, y = pos_by_node[node_id]
            disp[node_id][0] += -gravity_k * x
            disp[node_id][1] += -gravity_k * y

        for i in range(len(all_nodes_sorted)):
            a = all_nodes_sorted[i]
            ax, ay = pos_by_node[a]

            for j in range(i + 1, len(all_nodes_sorted)):
                b = all_nodes_sorted[j]
                bx, by = pos_by_node[b]

                dx = ax - bx
                dy = ay - by
                dist = math.hypot(dx, dy)

                if dist < 1e-9:
                    angle = 0.3141592653589793 * (i + 1)
                    dx = math.cos(angle) * 1e-6
                    dy = math.sin(angle) * 1e-6
                    dist = math.hypot(dx, dy)

                min_dist = node_center_min_distance_px
                if dist < min_dist:
                    overlap = min_dist - dist
                    ux = dx / dist
                    uy = dy / dist
                    force = node_repulsion_k * overlap

                    if a in disp:
                        disp[a][0] += ux * force
                        disp[a][1] += uy * force
                    if b in disp:
                        disp[b][0] -= ux * force
                        disp[b][1] -= uy * force

        for node_id in movable_nodes:
            px, py = pos_by_node[node_id]

            for c, d in edge_list:
                if c == node_id or d == node_id:
                    continue

                cx, cy = pos_by_node[c]
                dx, dy = pos_by_node[d]

                qx, qy, _ = point_segment_projection(px, py, cx, cy, dx, dy)
                vx = px - qx
                vy = py - qy
                dist = math.hypot(vx, vy)

                min_dist = node_radius + node_edge_gap_px
                if dist < 1e-9:
                    edge_dx = dx - cx
                    edge_dy = dy - cy
                    edge_len = math.hypot(edge_dx, edge_dy)
                    if edge_len < 1e-9:
                        continue
                    ux = -edge_dy / edge_len
                    uy = edge_dx / edge_len
                    force = edge_repulsion_k * min_dist
                    disp[node_id][0] += ux * force
                    disp[node_id][1] += uy * force
                    continue

                if dist < min_dist:
                    overlap = min_dist - dist
                    ux = vx / dist
                    uy = vy / dist
                    force = edge_repulsion_k * overlap
                    disp[node_id][0] += ux * force
                    disp[node_id][1] += uy * force

        target_edge_len = float(node_center_min_distance_px * 0.92)

        for child_id, parent_id in edge_list:
            cx, cy = pos_by_node[child_id]
            px, py = pos_by_node[parent_id]

            dx = cx - px
            dy = cy - py
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                continue

            ux = dx / dist
            uy = dy / dist
            force = spring_k * (dist - target_edge_len)

            if child_id in disp:
                disp[child_id][0] -= ux * force
                disp[child_id][1] -= uy * force
            if parent_id in disp:
                disp[parent_id][0] += ux * force
                disp[parent_id][1] += uy * force

        moved_any = False

        proposed_order = sorted(movable_nodes, key=lambda x: (-depth_by_node[x], str(x)))

        for node_id in proposed_order:
            dx = disp[node_id][0]
            dy = disp[node_id][1]
            mag = math.hypot(dx, dy)

            if mag < 1e-9:
                continue

            if mag > step_cap:
                dx = dx / mag * step_cap
                dy = dy / mag * step_cap

            old_x, old_y = pos_by_node[node_id]
            old_crossings = 0
            if (crossing_check_every_n_iterations > 0) and (((iter_idx + 1) % crossing_check_every_n_iterations) == 0):
                old_crossings = crossing_count_for_node(node_id)

            accepted = False
            for scale in [1.0, 0.5, 0.25, 0.125]:
                cand_x = old_x + dx * scale
                cand_y = old_y + dy * scale

                pos_by_node[node_id][0] = cand_x
                pos_by_node[node_id][1] = cand_y

                if (crossing_check_every_n_iterations > 0) and (((iter_idx + 1) % crossing_check_every_n_iterations) == 0):
                    new_crossings = crossing_count_for_node(node_id)
                    if new_crossings > old_crossings:
                        pos_by_node[node_id][0] = old_x
                        pos_by_node[node_id][1] = old_y
                        continue

                accepted = True
                moved_any = True
                break

            if not accepted:
                pos_by_node[node_id][0] = old_x
                pos_by_node[node_id][1] = old_y

        if (((iter_idx + 1) == 1) or (((iter_idx + 1) % 500) == 0)):
            filename_root, filename_ext = os.path.splitext(filename_out)
            iter_filename_out = f"{filename_root}-iter{iter_idx + 1}{filename_ext}"
            write_svg_file(iter_filename_out)

        if not moved_any:
            break

        step_cap = max(0.75, step_cap * cooling)

    final_svg_info = write_svg_file(filename_out)

    return {
        "filename_out": filename_out,
        "radius_by_depth": dict(radius_by_depth),
        "angle_by_node": dict(angle_by_node),
        "pos_by_node": {k: (v[0], v[1]) for k, v in pos_by_node.items()},
        "width": final_svg_info["width"],
        "height": final_svg_info["height"],
        "node_center_min_distance_px": node_center_min_distance_px,
        "node_edge_gap_px": node_edge_gap_px,
        "hidden_summary_nodes_added": hidden_summary_nodes_added,
        "hidden_contributions_represented": hidden_contributions_represented,
        "root_leaf_nodes_trimmed": root_leaf_nodes_trimmed,
        "root_leaf_contributions_represented": root_leaf_contributions_represented,
        "min_children": min_children,
        "min_thresh": min_thresh,
        "root_fill_color": root_fill_color,
        "trim_root_leaf_summary_nodes": trim_root_leaf_summary_nodes,
        "visualization_crawl_direction": detected_crawl_direction,
    }
