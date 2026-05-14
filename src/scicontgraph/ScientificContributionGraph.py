# ScientificContributionGraph.py

from __future__ import annotations

from typing import Annotated, Literal, Optional, Dict, List, Union
from pydantic import BaseModel, Field, ConfigDict

import os
import json
import orjson
import glob
import heapq
import time
import psutil
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed


# Embedding model
MODEL_NAME_SEARCH = "Qwen/Qwen3-Embedding-0.6B"
MAX_LENGTH_SEARCH = 512


#
#   Storage Classes
#

# An author of a paper
class AuthorName(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first_name: str
    last_name: str

# A type (classification) of contribution
class ContributionType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    explanation: str

# A specific link between a contribution mentioned in a prerequisite, and the actual source of that contribution.
# This is the main source of links between papers.
class Match(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contribution_id: str
    explanation: str
    match_type: Optional[Literal["strong", "weak"]] = None
    match_method: str


# Reference Union (3 types: PaperReference, InternalReference, OtherReference)
class PaperReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["paper"] = "paper"

    paper_title: Optional[str]
    paper_year: Optional[int]
    paper_first_author: Optional[AuthorName]
    paper_venue: Optional[str]

    corpus_id: Optional[str] = None
    corpus_id_match_confidence: Optional[float] = None
    corpus_id_match_method: Optional[str] = None

    matches: List[Match] = Field(default_factory=list)


class InternalReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["internal"] = "internal"

    contribution_name: str
    contribution_id: Optional[str]
    explanation: str


class OtherReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["other"] = "other"
    name: Optional[str] = None
    url: Optional[str] = None


Reference = Annotated[
    Union[PaperReference, InternalReference, OtherReference],
    Field(discriminator="type"),
]


# Prerequisite of a Contribution
class Prerequisite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    explanation: str
    core_or_peripheral: str  # if you have a fixed set, make this a Literal[...]
    references: List[Reference] = Field(default_factory=list)

# Contribution of a Paper
class Contribution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contribution_id: str
    name: str
    description: str
    types: list[ContributionType] = Field(default_factory=list)
    sections: List[str] = Field(default_factory=list)
    prerequisites: List[Prerequisite] = Field(default_factory=list)

# Paper (a list of contributions)
class Paper(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus_id: str
    title: Optional[str]
    year: Optional[int] = None
    publication_date: Optional[Dict[str, Optional[int]]] = None
    contributions: List[Contribution] = Field(default_factory=list)


#
#   Main Scientific Contribution Graph Class
#

class ScientificContributionGraph:
    def __init__(self, path: str, search_enabled: bool = False, search_device: str = "auto"):
        # Paths
        self.path = os.path.join(path, "data/")                     # Add /data/ to the path
        self.path_papers = os.path.join(self.path, "papers")
        self.path_embeddings = os.path.join(self.path, "embeddings")
        self.path_metadata = os.path.join(self.path, "metadata")

        # Keep track of load time for the graph
        load_time_start = time.time()

        # Get all corpus IDs in the graph
        self.all_corpus_ids = self.get_all_corpus_files()
        print("Found " + str(len(self.all_corpus_ids)) + " papers in the graph.")

        # Load the forward references
        self.forward_references_contrib_to_paper = {}
        self.forward_references_contrib_to_contrib = {}
        self.load_forward_references(os.path.join(self.path_metadata, "forward_references.json"))

        # Load the paper title to corpus id lookup
        self.paper_title_to_corpus_id = {}
        self.paper_title_normalized_to_corpus_id = {}
        self.paper_title_tokenized_to_corpus_id = []
        self.load_paper_title_to_corpus_id_lookup(os.path.join(self.path_metadata, "paper_title_to_corpus_id.json"))

        # Load the corpus id to paper year lookup
        self.corpus_id_to_paper_year = {}
        self.load_corpus_id_to_paper_year_lookup(os.path.join(self.path_metadata, "corpus_id_to_paper_year.json"))

        # Search-related
        self.search_enabled = search_enabled
        self.search_device_mode = search_device
        self.search_device = None
        self.search_tokenizer = None
        self.search_model = None
        self.search_shards = []              # list of (npy_path, lut_path)
        self.search_all_embeddings = []      # list of per-shard ndarrays (typically float16)
        self.search_all_ids = []             # list of per-shard List[Optional[str]]
        self.search_all_valid_idx = []       # list of per-shard np.int64 index arrays

        if self.search_enabled:
            self.initialize_search()

        # Report load time
        load_time_delta = time.time() - load_time_start
        print(f"ScientificContributionGraph initialized in {load_time_delta:.2f} seconds")

        # Report memory usage
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_usage_gb = mem_info.rss / (1024 ** 3)
        print(f"Current Python memory usage: {mem_usage_gb:.2f} GB")

    #
    #   Embedding Search
    #

    def get_search_device(self) -> str:
        if self.search_device_mode == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.search_device_mode == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested search_device='cuda' but CUDA is not available")
            return "cuda"
        if self.search_device_mode == "cpu":
            return "cpu"
        raise RuntimeError(f"Unknown search_device: {self.search_device_mode}")

    def search_log(self, msg: str):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] {msg}")

    def _madvise_no_hugepage(self, arr: np.ndarray) -> None:
        """Ask the kernel not to back this allocation with transparent huge pages.
        Defends against kcompactd / THP-compaction stalls on long-uptime hosts.
        Linux-only; silent no-op everywhere else."""
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            MADV_NOHUGEPAGE = 15
            libc.madvise(ctypes.c_void_p(arr.ctypes.data),
                        ctypes.c_size_t(arr.nbytes),
                        ctypes.c_int(MADV_NOHUGEPAGE))
        except Exception:
            pass

    def last_token_pool_search(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), lengths]

    def load_search_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_SEARCH, trust_remote_code=True, use_fast=True)
        dtype = torch.bfloat16 if self.search_device.startswith("cuda") else torch.float32
        model = AutoModel.from_pretrained(MODEL_NAME_SEARCH, trust_remote_code=True, dtype=dtype).to(self.search_device)
        model.eval()
        return tokenizer, model

    def embed_search_text(self, text: str) -> np.ndarray:
        tokens = self.search_tokenizer([text], padding=True, truncation=True, max_length=MAX_LENGTH_SEARCH, return_tensors="pt")
        tokens = {k: v.to(self.search_device, non_blocking=True) for k, v in tokens.items()}
        with torch.inference_mode():
            out = self.search_model(**tokens)
            emb = self.last_token_pool_search(out.last_hidden_state, tokens["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            emb = emb.to(torch.float32)
        return emb[0].detach().cpu().numpy()

    def search_shard_sort_key(self, npy_path: str) -> int:
        stem = os.path.splitext(os.path.basename(npy_path))[0]
        return int(stem.split("_")[1])

    def find_search_shards(self) -> List[tuple[str, str]]:
        if self.path_embeddings is None:
            raise RuntimeError("Search is enabled but path_embeddings is None")
        npy_files = sorted(glob.glob(os.path.join(self.path_embeddings, "shard_*.npy")), key=self.search_shard_sort_key)
        if not npy_files:
            raise RuntimeError(f"No shard_*.npy files found in {self.path_embeddings}")
        shards = []
        for npy_path in npy_files:
            base = os.path.splitext(os.path.basename(npy_path))[0]
            lut_path = os.path.join(self.path_embeddings, f"{base}.lut.json")
            if not os.path.exists(lut_path):
                raise RuntimeError(f"Missing LUT file for shard: {npy_path}")
            shards.append((npy_path, lut_path))
        return shards

    def load_search_ids_in_row_order(self, lut_path: str, expected_rows: int) -> List[Optional[str]]:
        with open(lut_path, "rb") as f:
            lut = orjson.loads(f.read())

        row_to_id: List[Optional[str]] = [None] * expected_rows
        out_of_range = 0

        for cid, row_idx in lut.items():
            if not isinstance(row_idx, int):
                continue
            if row_idx < 0 or row_idx >= expected_rows:
                out_of_range += 1
                continue
            row_to_id[row_idx] = cid

        missing = sum(1 for x in row_to_id if x is None)
        if missing > 0 or out_of_range > 0:
            self.search_log(
                f"[WARN] LUT {os.path.basename(lut_path)}: expected_rows={expected_rows} "
                f"lut_entries={len(lut)} missing_rows={missing} out_of_range={out_of_range}"
            )

        return row_to_id

    def _shard_scores(self, emb: np.ndarray, q32: np.ndarray, block_rows: int = 32768) -> np.ndarray:
        """Compute (emb @ q) as float32 scores.
        If emb is float16, convert one block at a time so the inner matmul goes
        through BLAS (NumPy has no BLAS path for float16). Block size is chosen
        to keep the temporary array (~128 MB at 1024-dim) in L3 cache."""
        if emb.dtype == np.float32:
            return emb @ q32
        n = emb.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, block_rows):
            block = emb[i:i + block_rows].astype(np.float32, copy=False)
            out[i:i + block_rows] = block @ q32
        return out

    def topk_from_search_scores(self, scores: np.ndarray, ids_in_order: List[Optional[str]], top_k: int, valid_idx: Optional[np.ndarray] = None) -> List[tuple[float, str]]:
        if scores.ndim != 1:
            raise RuntimeError(f"Expected 1D scores, got shape {scores.shape}")

        if valid_idx is None:
            valid_idx = np.fromiter((i for i, cid in enumerate(ids_in_order) if cid is not None), dtype=np.int64)
        else:
            valid_idx = np.asarray(valid_idx, dtype=np.int64)

        if valid_idx.size == 0:
            return []

        valid_scores = scores[valid_idx]
        k = min(top_k, valid_scores.shape[0])
        if k <= 0:
            return []

        local_idx = np.argpartition(valid_scores, -k)[-k:]
        local_idx = local_idx[np.argsort(valid_scores[local_idx])[::-1]]
        global_idx = valid_idx[local_idx]

        return [(float(scores[i]), ids_in_order[i]) for i in global_idx if ids_in_order[i] is not None]

    def merge_search_topk(self, global_heap: List[tuple[float, str]], candidates: List[tuple[float, str]], top_k: int) -> None:
        for score, cid in candidates:
            if len(global_heap) < top_k:
                heapq.heappush(global_heap, (score, cid))
            elif score > global_heap[0][0]:
                heapq.heapreplace(global_heap, (score, cid))

    def initialize_search(self):
        self.search_device = self.get_search_device()
        self.search_log(f"Using search device: {self.search_device}")
        self.search_log(f"Search embeddings path: {self.path_embeddings}")

        self.search_tokenizer, self.search_model = self.load_search_model_and_tokenizer()
        self.search_shards = self.find_search_shards()

        self.search_all_embeddings = []
        self.search_all_ids = []
        self.search_all_valid_idx = []

        # Pass 1: scan headers + LUTs without loading vector data.
        t0 = time.time()
        total_rows = 0
        total_valid_rows = 0
        total_disk_bytes = 0
        shard_dtypes: Dict[str, int] = {}
        shard_shapes: Dict[str, int] = {}

        self.search_log(f"[INDEX] scanning metadata for {len(self.search_shards)} shard(s)")
        for shard_num, (npy_path, lut_path) in enumerate(self.search_shards, start=1):
            emb_header = np.load(npy_path, mmap_mode="r")
            rows = emb_header.shape[0]
            dtype_name = str(emb_header.dtype)
            shape_name = str(tuple(emb_header.shape))
            file_bytes = os.path.getsize(npy_path)

            ids = self.load_search_ids_in_row_order(lut_path, rows)
            if rows != len(ids):
                raise RuntimeError(f"Row mismatch in {npy_path}: matrix rows={rows} ids={len(ids)}")

            valid_idx = np.fromiter((i for i, cid in enumerate(ids) if cid is not None), dtype=np.int64)

            self.search_all_ids.append(ids)
            self.search_all_valid_idx.append(valid_idx)

            total_rows += rows
            total_valid_rows += int(valid_idx.size)
            total_disk_bytes += file_bytes
            shard_dtypes[dtype_name] = shard_dtypes.get(dtype_name, 0) + 1
            shard_shapes[shape_name] = shard_shapes.get(shape_name, 0) + 1

            self.search_log(
                f"[INDEX] shard {shard_num}/{len(self.search_shards)}: "
                f"{os.path.basename(npy_path)} shape={emb_header.shape} dtype={emb_header.dtype} "
                f"valid_ids={valid_idx.size} disk={file_bytes / (1024 ** 3):.3f} GB"
            )

        self.search_log(
            f"[INDEX] metadata ready: shards={len(self.search_shards)} rows={total_rows} "
            f"valid_ids={total_valid_rows} dtypes={shard_dtypes} shapes={shard_shapes} "
            f"disk={total_disk_bytes / (1024 ** 3):.3f} GB in {time.time() - t0:.2f}s"
        )

        # Pass 2: load every shard into RAM, keeping native dtype (typically float16).
        # No astype, no vstack. Each shard is kept as its own ndarray in a list.
        t0 = time.time()
        self.search_log(f"[FULL] loading {len(self.search_shards)} shard(s) into memory")

        for shard_num, (npy_path, _) in enumerate(self.search_shards, start=1):
            t_shard = time.time()
            emb = np.load(npy_path)
            self._madvise_no_hugepage(emb)

            if emb.ndim != 2:
                raise RuntimeError(f"Expected 2D embedding matrix in {npy_path}, got shape={emb.shape}")
            expected_rows = len(self.search_all_ids[shard_num - 1])
            if emb.shape[0] != expected_rows:
                raise RuntimeError(f"Row mismatch in {npy_path}: matrix rows={emb.shape[0]} ids={expected_rows}")

            self.search_all_embeddings.append(emb)
            self.search_log(
                f"[FULL]   shard {shard_num}/{len(self.search_shards)} loaded "
                f"({os.path.basename(npy_path)}, dtype={emb.dtype}) in {time.time() - t_shard:.2f}s"
            )

        ram_gb = sum(emb.nbytes for emb in self.search_all_embeddings) / (1024 ** 3)
        rows = sum(emb.shape[0] for emb in self.search_all_embeddings)
        dtypes: Dict[str, int] = {}
        for emb in self.search_all_embeddings:
            dtypes[str(emb.dtype)] = dtypes.get(str(emb.dtype), 0) + 1
        self.search_log(
            f"[FULL] loaded: vectors={rows} RAM={ram_gb:.2f} GB dtypes={dtypes} in {time.time() - t0:.2f}s"
        )

        # Sanity check: cosine search assumes shards are L2-normalized.
        sample = self.search_all_embeddings[0][:128].astype(np.float32, copy=False)
        mean_norm = float(np.linalg.norm(sample, axis=1).mean())
        if not (0.98 <= mean_norm <= 1.02):
            self.search_log(
                f"[WARN] embeddings appear unnormalized (mean L2 of sample = {mean_norm:.4f}); "
                f"cosine results will be incorrect. Re-export shards with L2 normalization."
            )

    def search(self, query: str, top_n: int = 20, populate_names: bool = False) -> List[Dict]:
        if not self.search_enabled:
            raise RuntimeError("Search is disabled for this ScientificContributionGraph instance")
        if top_n <= 0:
            raise RuntimeError("top_n must be > 0")
        if not self.search_all_embeddings:
            raise RuntimeError("Search index not loaded in memory")

        query_vec = self.embed_search_text(query)
        self.search_log(f"Embedded query; dim={query_vec.shape[0]}")

        q32 = query_vec.astype(np.float32, copy=False)
        global_heap: List[tuple[float, str]] = []
        t0 = time.time()
        total_rows = 0
        total_valid_rows = 0

        self.search_log(f"[FULL] searching {len(self.search_all_embeddings)} shard(s)")
        for shard_num, emb in enumerate(self.search_all_embeddings, start=1):
            ids = self.search_all_ids[shard_num - 1]
            valid_idx = self.search_all_valid_idx[shard_num - 1]

            scores = self._shard_scores(emb, q32)
            shard_top = self.topk_from_search_scores(scores, ids, top_n, valid_idx=valid_idx)
            self.merge_search_topk(global_heap, shard_top, top_n)

            total_rows += emb.shape[0]
            total_valid_rows += int(valid_idx.size)

        ranked_list = [
            {
                "corpus_id": self.get_corpus_id_from_contribution(cid),
                "contribution_id": cid,
                "cosine": score,
            }
            for score, cid in sorted(global_heap, key=lambda x: x[0], reverse=True)
        ]

        self.search_log(
            f"[FULL] completed search over {total_rows} vectors ({total_valid_rows} with IDs) in {time.time() - t0:.2f}s"
        )

        if populate_names:
            # Cache loaded papers so two results from the same paper don't read it twice,
            # and so contribution-name + paper-info don't trigger two separate load_paper calls.
            paper_cache: Dict[str, Optional[Paper]] = {}
            for result in ranked_list:
                contribution_id = result.get("contribution_id")
                if contribution_id is None:
                    continue

                corpus_id = self.get_corpus_id_from_contribution(contribution_id)
                if corpus_id is None:
                    result["contribution_name"] = None
                    result["contribution_description"] = None
                    result["paper_info"] = None
                    continue

                if corpus_id not in paper_cache:
                    paper_cache[corpus_id] = self.load_paper(corpus_id)
                paper = paper_cache[corpus_id]

                contribution_obj = None
                if paper is not None:
                    for c in paper.contributions:
                        if c.contribution_id == contribution_id:
                            contribution_obj = c
                            break

                if paper is not None and contribution_obj is not None:
                    result["contribution_name"] = contribution_obj.name
                    result["contribution_description"] = contribution_obj.description
                    result["paper_info"] = {
                        "corpus_id": corpus_id,
                        "paper_title": paper.title,
                        "paper_year": paper.year,
                        "paper_publication_date": paper.publication_date,
                    }
                else:
                    result["contribution_name"] = None
                    result["contribution_description"] = None
                    result["paper_info"] = None

        return ranked_list

    #
    #   Accessors for contributions and papers
    #

    # Load one paper by corpus_id
    def load_paper(self, corpus_id:str):
        try:
            filename = corpus_id + ".json"
            subdir = str(corpus_id[:2]) # The first 2 digits of the corpus_id determine the subdirectory (e.g. "23" for corpus_id "233297051"). This is to avoid having too many files in one directory.
            full_path = os.path.join(self.path_papers, subdir)
            filepath = os.path.join(full_path, filename)

            with open(filepath, "rb") as f:
                data = orjson.loads(f.read())
                paper = Paper.model_validate(data)
                return paper
        except FileNotFoundError:
            print(f"Paper with corpus_id {corpus_id} not found at {filepath}.")
        except Exception as e:
            print(f"Error loading paper with corpus_id {corpus_id} from {filepath}: {e}")

        return None

    # Extract the corpus_id from a contribution_id. This assumes that the contribution_id is in the format "corpus_id.contribution_key". If the format is invalid, return None.
    def get_corpus_id_from_contribution(self, contribution_id: str) -> Optional[str]:
        # Assuming contribution_id is in the format "corpus_id.contribution_key"
        if "." in contribution_id:
            corpus_id = contribution_id.split(".")[0]
            return corpus_id
        else:
            print(f"Invalid contribution_id format: {contribution_id}. Expected format 'corpus_id.contribution_key'.")
            return None

    # Get a contribution object by its contribution_id. This requires loading the corresponding paper and finding the contribution within it.
    def get_contribution_by_id(self, contribution_id: str) -> Optional[Contribution]:
        corpus_id = self.get_corpus_id_from_contribution(contribution_id)
        if corpus_id is None:
            return None

        paper = self.load_paper(corpus_id)
        if paper is None:
            return None

        for contribution in paper.contributions:
            if contribution.contribution_id == contribution_id:
                return contribution

        print(f"Contribution with id {contribution_id} not found in paper with corpus_id {corpus_id}.")
        return None

    # Get the paper metadata (corpus_id, title, year, publication_date) for a given contribution_id
    def get_paper_info_by_contribution_id(self, contribution_id: str) -> Optional[Dict[str, Optional[str]]]:
        corpus_id = self.get_corpus_id_from_contribution(contribution_id)
        if corpus_id is None:
            return None

        paper = self.load_paper(corpus_id)
        if paper is None:
            return None

        return {
            "corpus_id": corpus_id,
            "paper_title": paper.title,
            "paper_year": paper.year,
            "paper_publication_date": paper.publication_date
        }

    def get_paper_info_by_corpus_id(self, corpus_id: str) -> Optional[Dict[str, Optional[str]]]:
        paper = self.load_paper(corpus_id)
        if paper is None:
            return None

        return {
            "corpus_id": corpus_id,
            "paper_title": paper.title,
            "paper_year": paper.year,
            "paper_publication_date": paper.publication_date
        }

    # Get a list of contribution_ids for a given corpus_id
    def get_contribution_ids_for_corpus_id(self, corpus_id: str) -> List[str]:
        paper = self.load_paper(corpus_id)
        if paper is None:
            return []
        contribution_ids = [contribution.contribution_id for contribution in paper.contributions]
        return contribution_ids

    #
    #   Get a corpus ID from a paper title
    #
    def get_corpus_id_from_paper_title(self, paper_title: str, top_k=10) -> List[dict]:
        title_tokens = self.title_normalize_tokenize(paper_title)
        title_normalized = " ".join(title_tokens)
        # See if we get a direct match on the normalized title
        if (title_normalized in self.paper_title_normalized_to_corpus_id):
            match = self.paper_title_normalized_to_corpus_id[title_normalized]
            return [{"score": 1.0, "corpus_id": match["corpus_id"], "paper_title": match["title"]}]

        # If we don't, fall-back to partial matching using intersections
        scored_list = []
        title_tokens_set = set(title_tokens)
        if (len(title_tokens_set) == 0):
            return []

        for candidate_info in self.paper_title_tokenized_to_corpus_id:
            candidate_tokens = candidate_info["tokens"]
            if (not candidate_tokens) or (len(candidate_tokens) == 0):
                continue
            intersection = title_tokens_set.intersection(candidate_tokens)
            score = len(intersection) / len(title_tokens_set)
            scored_list.append((score, candidate_info))

        # Sort by score and return the top matches
        scored_list.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored_list[:top_k]

        # Pack the results
        results = []
        for score, candidate_info in top_matches:
            results.append({
                "score": score,
                "corpus_id": candidate_info["corpus_id"],
                "paper_title": candidate_info["title"]
            })
        return results




    #
    #   Get a list of all corpus files
    #

    def get_all_corpus_files(self) -> List[str]:
        # Get all the JSON files in the directory, and extract the corpus_id from the filename (assuming filename is "corpus_id.json")
        corpus_files = []
        # JSON files are in nested subdirectories.  Have to explore the subdirectory tree 1 level deep.
        for root, dirs, files in os.walk(self.path_papers):
            for filename in files:
                if (filename.endswith(".json")):
                    corpus_id = filename[:-5]  # Remove the ".json" extension
                    corpus_files.append(corpus_id)

        return corpus_files

    # Get a list of all corpus_ids in the graph.
    def get_all_corpus_ids(self) -> List[str]:
        return self.all_corpus_ids




    #
    #   Accessors for forward references
    #
    def get_papers_referencing_contribution(self, contribution_id: str) -> List[str]:
        return self.forward_references_contrib_to_paper.get(contribution_id, [])

    def get_contributions_referencing_contribution(self, contribution_id: str) -> List[str]:
        return self.forward_references_contrib_to_contrib.get(contribution_id, [])

    #
    #   Pre-generate forward references
    #

    def generate_forward_references(self, filename_out:str):
        # Each paper nominally has a list of back references (corpus_id -> list of contributions -> list of prerequisites -> list of references -> list of matches).
        # If we want to go forward, we need the list_of_matches -> corpus_id.

        forward_references_contrib_to_paper = {}
        forward_references_contrib_to_contrib = {}

        # Get all the JSON files
        corpus_files = self.get_all_corpus_files()
        for corpus_id in tqdm(corpus_files):
            # Load the paper
            paper = self.load_paper(corpus_id)
            if (paper is None):
                continue

            # for each contribution, for each prerequisite, for each paper reference, for each match, add to the forward_references
            for contribution in paper.contributions:
                for prereq in contribution.prerequisites:
                    for ref in prereq.references:
                        if isinstance(ref, PaperReference):
                            for match in ref.matches:
                                contribution_id = match.contribution_id

                                # Contribution to paper
                                if contribution_id not in forward_references_contrib_to_paper:
                                    forward_references_contrib_to_paper[contribution_id] = []
                                if (corpus_id not in forward_references_contrib_to_paper[contribution_id]):
                                    forward_references_contrib_to_paper[contribution_id].append(corpus_id)
                                # Contribution to contribution
                                if contribution_id not in forward_references_contrib_to_contrib:
                                    forward_references_contrib_to_contrib[contribution_id] = []
                                if (contribution.contribution_id not in forward_references_contrib_to_contrib[contribution_id]):
                                    forward_references_contrib_to_contrib[contribution_id].append(contribution.contribution_id)



        # Now, save the forward references to a file.
        packed = {
            "forward_references_contribution_to_paper": forward_references_contrib_to_paper,
            "forward_references_contribution_to_contribution": forward_references_contrib_to_contrib
        }
        print("Writing: " + filename_out)
        with open(filename_out, "w") as f:
            json.dump(packed, f, indent=4)


    # Load the set of forward references from a file.
    def load_forward_references(self, filename_in:str):
        print("Loading pre-cached forward references...")
        if (not os.path.exists(filename_in)):
            print(f"Forward references file not found at {filename_in}. Regenerating (this may take a few minutes...)")
            self.generate_forward_references(filename_out=filename_in)

        with open(filename_in, "r") as f:
            data = json.load(f)
            self.forward_references_contrib_to_paper = data.get("forward_references_contribution_to_paper", {})
            self.forward_references_contrib_to_contrib = data.get("forward_references_contribution_to_contribution", {})


    #
    #   Pre-generate a Paper Title -> Corpus ID Look-up-table
    #
    def generate_paper_title_to_corpus_id_lookup(self, filename_out:str):
        paper_title_to_corpus_id = {}

        # Get all the JSON files
        corpus_files = self.get_all_corpus_files()
        for corpus_id in tqdm(corpus_files):
            # Load the paper
            paper = self.load_paper(corpus_id)
            if (paper is None):
                continue

            paper_title_to_corpus_id[paper.title] = corpus_id

        # Now, save the forward references to a file.
        print("Writing: " + filename_out)
        with open(filename_out, "w") as f:
            json.dump(paper_title_to_corpus_id, f, indent=4)


    def title_normalize_tokenize(self, title: str) -> str:
        # Lowercase, split on non-alphanumeric characters.
        tokens = re.split(r'\W+', title.lower())
        # Remove empty tokens
        tokens = [t for t in tokens if t]
        return tokens



    def load_paper_title_to_corpus_id_lookup(self, filename_in:str):
        print("Loading pre-cached paper title to corpus id lookup...")
        if (not os.path.exists(filename_in)):
            print(f"Paper title to corpus id lookup file not found at {filename_in}. Regenerating (this may take a few minutes...)")
            self.generate_paper_title_to_corpus_id_lookup(filename_out=filename_in)
            #return

        with open(filename_in, "r") as f:
            data = json.load(f)
            self.paper_title_to_corpus_id = data

        # Covert the key to be normalized and tokenized, for more robust searching.
        self.paper_title_normalized_to_corpus_id = {}
        for title, corpus_id in self.paper_title_to_corpus_id.items():
            normalized = " ".join(self.title_normalize_tokenize(title))
            self.paper_title_normalized_to_corpus_id[normalized] = {"corpus_id": corpus_id, "title": title}

        # Generate a normalized, tokenized version of the paper title to corpus ID lookup, for more robust searching.
        # First, convert to lowercase, and tokenize.
        self.paper_title_tokenized_to_corpus_id = []
        for title, corpus_id in self.paper_title_to_corpus_id.items():
            tokens = self.title_normalize_tokenize(title)
            packed = {
                "title": title,
                "tokens": set(tokens),
                "corpus_id": corpus_id
            }
            self.paper_title_tokenized_to_corpus_id.append(packed)




    # Corpus ID to paper year lookup
    def generate_corpus_id_to_paper_year_lookup(self, filename_out:str):
        corpus_id_to_paper_year = {}

        # Get all the JSON files
        corpus_files = self.get_all_corpus_files()
        for corpus_id in tqdm(corpus_files):
            # Load the paper
            paper = self.load_paper(corpus_id)
            if (paper is None):
                continue

            corpus_id_to_paper_year[corpus_id] = paper.year

        # Now, save the forward references to a file.
        print("Writing: " + filename_out)
        with open(filename_out, "w") as f:
            json.dump(corpus_id_to_paper_year, f, indent=4)

    def load_corpus_id_to_paper_year_lookup(self, filename_in:str):
        print("Loading pre-cached corpus id to paper year lookup...")
        if (not os.path.exists(filename_in)):
            print(f"Corpus id to paper year lookup file not found at {filename_in}. Regenerating (this may take a few minutes...)")
            self.generate_corpus_id_to_paper_year_lookup(filename_out=filename_in)
            return

        with open(filename_in, "r") as f:
            data = json.load(f)
            self.corpus_id_to_paper_year = data



    #
    #   Impact Metric (forward-crawl derived)
    #
    def _calculate_impact_score(self, edge_depth_lut:Dict[str, int]) -> float:
        impact_score = 0.0
        impact_score_dampened = 0.0
        for depth in edge_depth_lut.values():
            contribution_score = 1.0
            impact_score += contribution_score

            contribution_score_dampened = contribution_score / (depth + 1)
            impact_score_dampened += contribution_score_dampened

        return {"impact_score": impact_score, "impact_score_dampened": impact_score_dampened}


    def calculate_impact_metric_contribution(self, contribution_id:str, max_depth:int=5, include_edge_depth_lut:bool=False):
        # Forward crawl-based impact metric calculation
        forward_crawl_results = self.crawl_forwards_from_contribution(
            contribution_id=contribution_id,
            max_depth=max_depth,
            only_strong_connections=True,
            verbose_progress=True
        )

        edge_depth_lut = {}
        for edge in forward_crawl_results["edges"]:
            used_by_contribution_id = edge["used_by_contribution_id"]
            depth = edge["depth"]
            edge_depth_lut[used_by_contribution_id] = depth

        impact_scores = self._calculate_impact_score(edge_depth_lut)
        if (include_edge_depth_lut == True):
            impact_scores["edge_depth_lut"] = edge_depth_lut

        # Also get the name of the contribution
        contribution_obj = self.get_contribution_by_id(contribution_id)
        contribution_name = contribution_obj.name if contribution_obj is not None else None
        out = {"contribution_name": contribution_name}
        out.update(impact_scores)

        return out

    def calculate_impact_metric_paper(self, corpus_id:str, max_depth:int=5):
        contribution_ids = self.get_contribution_ids_for_corpus_id(corpus_id)
        paper_impact_scores = {
            "corpus_id": corpus_id,
            "paper_info": None,
            "max_depth": max_depth,
            "contribution_impact_scores": {},
            "overall_paper_impact_score": None
        }

        paper_impact_scores["paper_info"] = self.get_paper_info_by_corpus_id(corpus_id)

        edge_depth_lut_overall = {}
        # Calculate the impact scores for each contribution in the paper
        for contribution_id in contribution_ids:
            impact_scores = self.calculate_impact_metric_contribution(contribution_id, max_depth=max_depth, include_edge_depth_lut=True)
            edge_depth_lut_contribution = impact_scores.pop("edge_depth_lut", {})
            for used_by_contribution_id, depth in edge_depth_lut_contribution.items():
                if (used_by_contribution_id not in edge_depth_lut_overall):
                    edge_depth_lut_overall[used_by_contribution_id] = depth
                else:
                    edge_depth_lut_overall[used_by_contribution_id] = min(edge_depth_lut_overall[used_by_contribution_id], depth)

            paper_impact_scores["contribution_impact_scores"][contribution_id] = impact_scores

        # Calculate the overall paper impact score, by combining the edge_depth_lut_overall for all contributions in the paper.
        overall_impact_scores = self._calculate_impact_score(edge_depth_lut_overall)
        paper_impact_scores["overall_paper_impact_score"] = overall_impact_scores

        return paper_impact_scores


    #
    #   Crawling (forward/backward crawling)
    #



    #
    #   Backward Crawling (given a contribution, find all the contributions that it directly or indirectly relies on, up to a certain depth)
    #
    def crawl_backwards_from_contribution(self, contribution_id: str, max_depth: int = 3, current_depth: int = 0, visited_contributions: Optional[set] = None, stats: Optional[dict] = None, only_strong_connections:bool=True, verbose_progress:bool=True) -> Dict[str, List[str]]:
        if visited_contributions is None:
            visited_contributions = set()

        if stats is None:
            stats = {
                "start_time": time.time(),
                "nodes_visited": 0,
                "edges_added": 0,
            }

        if current_depth > max_depth:
            return None

        if contribution_id in visited_contributions:
            return None

        visited_contributions.add(contribution_id)
        stats["nodes_visited"] += 1

        if stats["nodes_visited"] % 100 == 0:
            elapsed = time.time() - stats["start_time"]
            print(f"[backward crawl] depth={current_depth} visited={stats['nodes_visited']} edges={stats['edges_added']} elapsed={elapsed:.1f}s")

        contribution_obj = self.get_contribution_by_id(contribution_id)
        if contribution_obj is None:
            return None

        out = {
            "root_node": contribution_id,
            "nodes": {},
            "edges": []
        }

        num_added = 0
        # Include the tqdm progress bar, if requested
        progress_bar = None
        if (verbose_progress == True) and (current_depth <= 1):
            progress_bar = tqdm(total=len(contribution_obj.prerequisites), desc=f"Crawling backwards from contribution {contribution_id} (depth {max_depth})", unit="prerequisite")

        for prereq in contribution_obj.prerequisites:
            for ref in prereq.references:
                if isinstance(ref, PaperReference):
                    for match in ref.matches:
                        next_contribution_id = match.contribution_id
                        match_type = match.match_type
                        if ((not isinstance(match_type, str)) or ((only_strong_connections == True) and (match_type != "strong"))):
                            # If it's not a strong connection, or if it's an unknown connetion, then do not follow this edge in the crawl.
                            # Update progress bar
                            continue

                        next_contribution_obj = self.get_contribution_by_id(next_contribution_id)
                        next_contribution_obj_json = next_contribution_obj.model_dump() if next_contribution_obj is not None else None

                        if (next_contribution_obj_json is not None):
                            next_contribution_obj_json.pop("prerequisites", None)

                        paper_title = None
                        paper_corpus_id = None
                        if (next_contribution_obj is not None):
                            next_corpus_id = self.get_corpus_id_from_contribution(next_contribution_id)
                            if next_corpus_id is not None:
                                paper = self.load_paper(next_corpus_id)
                                if paper is not None:
                                    paper_title = paper.title
                                    paper_corpus_id = paper.corpus_id

                        new_node = {
                            "contribution_id": next_contribution_id,
                            "paper_title": paper_title,
                            "paper_corpus_id": paper_corpus_id,
                            "contribution_obj": next_contribution_obj_json,
                        }

                        prerequisite_description = prereq.description
                        prerequisite_explanation = prereq.explanation

                        new_edge = {
                            "contribution_id": next_contribution_id,
                            "used_by_contribution_id": contribution_id,
                            "prerequisite_description": prerequisite_description,
                            "prerequisite_explanation": prerequisite_explanation,
                            "strengths": [match_type],
                            "depth": current_depth,
                        }

                        out["nodes"][next_contribution_id] = new_node
                        out["edges"].append(new_edge)
                        stats["edges_added"] += 1

                        results_next_level = self.crawl_backwards_from_contribution(
                            contribution_id=next_contribution_id,
                            max_depth=max_depth,
                            current_depth=current_depth + 1,
                            visited_contributions=visited_contributions,
                            stats=stats,
                            only_strong_connections=only_strong_connections,
                            verbose_progress=verbose_progress
                        )
                        if (results_next_level is not None):
                            out["nodes"].update(results_next_level["nodes"])
                            out["edges"].extend(results_next_level["edges"])

                        num_added += 1

            # Update progress bar
            if (progress_bar is not None):
                progress_bar.update(1)
                progress_bar.set_description(f"Crawling backwards from contribution {contribution_id} (depth {current_depth}/{max_depth}) added {num_added} nodes")


        if (num_added > 0):
            contribution_obj_json = contribution_obj.model_dump() if contribution_obj is not None else None

            paper_title = None
            paper_corpus_id = None
            corpus_id = self.get_corpus_id_from_contribution(contribution_id)
            if corpus_id is not None:
                paper = self.load_paper(corpus_id)
                if paper is not None:
                    paper_title = paper.title
                    paper_corpus_id = paper.corpus_id

            new_node = {
                "contribution_id": contribution_id,
                "paper_title": paper_title,
                "paper_corpus_id": paper_corpus_id,
                "contribution_obj": contribution_obj_json,
            }

            if contribution_id not in out["nodes"]:
                out["nodes"][contribution_id] = new_node

        return out



    #
    #   Forward Crawling (given a contribution, find all the contributions that directly or indirectly use it, up to a certain depth)
    #
    # From a given contribution, crawl forward through the graph to find all the contributions that directly (or indirectly, up to a certain depth) use it.
    # Output will be a graph, comprising a list of {contribution_id:"", used_by_contribution_id:"", depth:int} dictionaries.
    def crawl_forwards_from_contribution(self, contribution_id: str, max_depth: int = 3, current_depth: int = 0, visited_contributions: Optional[set] = None, only_strong_connections:bool=True, verbose_progress:bool=True) -> Dict[str, List[str]]:
        if visited_contributions is None:
            visited_contributions = set()

        if current_depth > max_depth:
            return None

        if contribution_id in visited_contributions:
            return None

        visited_contributions.add(contribution_id)

        forward_references = self.get_contributions_referencing_contribution(contribution_id)

        out = {
            "root_node": contribution_id,
            "nodes": {},
            "edges": []
        }

        if (forward_references is None) or (len(forward_references) == 0):
            return out

        num_added = 0
        # Include the tqdm progress bar, if requested
        progress_bar = None
        if (verbose_progress == True) and (current_depth <= 1):
            progress_bar = tqdm(forward_references, desc=f"Crawling forwards from contribution {contribution_id} (depth {max_depth})", unit="contribution")

        for next_contribution_id in forward_references:

            # get the contribution object for the next_contribution_id
            contribution_obj = self.get_contribution_by_id(next_contribution_id)       # TODO: Needs to be a serialized JSON object
            contribution_obj_json = contribution_obj.model_dump() if contribution_obj is not None else None

            # Here, we'll actually go through the prerequisites, and see the connection strength.
            connection_strengths = set()
            prerequisite_description = None
            prerequisite_explanation = None
            for prereq in contribution_obj.prerequisites:
                for ref in prereq.references:
                    if isinstance(ref, PaperReference):
                        for match in ref.matches:
                            if match.contribution_id == contribution_id:
                                connection_strengths.add(match.match_type)
                                prerequisite_description = prereq.description
                                prerequisite_explanation = prereq.explanation

            # Check to see if there's a strong connection, and skip this one if it's not strong.
            has_strong_connection = "strong" in connection_strengths
            if (only_strong_connections == True) and (not has_strong_connection):
                if (progress_bar is not None):
                    progress_bar.update(1)
                    progress_bar.set_description(f"Crawling forwards from contribution {contribution_id} (depth {current_depth}/{max_depth})")
                continue

            # Strip out the prerequisites from the JSON object, since they can be very large and we don't need them for the forward crawl.
            if (contribution_obj_json is not None):
                contribution_obj_json.pop("prerequisites", None)

            # Get the paper title for the contribution, to display to the user.
            paper_title = None
            paper_corpus_id = None
            if (contribution_obj is not None):
                paper = self.load_paper(self.get_corpus_id_from_contribution(next_contribution_id))
                if paper is not None:
                    paper_title = paper.title
                    paper_corpus_id = paper.corpus_id

            # Let's split into nodes and edges. The node is the contribution, and the edge is the connection from the original contribution to this one.
            new_node = {
                "contribution_id": next_contribution_id,
                "paper_title": paper_title,
                "paper_corpus_id": paper_corpus_id,
                "contribution_obj": contribution_obj_json,
            }

            new_edge = {
                "contribution_id": contribution_id,
                "used_by_contribution_id": next_contribution_id,
                "prerequisite_description": prerequisite_description,
                "prerequisite_explanation": prerequisite_explanation,
                "strengths": list(connection_strengths),
                "depth": current_depth,
            }

            #out["nodes"].append(new_node)
            out["nodes"][next_contribution_id] = new_node
            out["edges"].append(new_edge)

            results_next_level = self.crawl_forwards_from_contribution(
                contribution_id=next_contribution_id,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                visited_contributions=visited_contributions,
                only_strong_connections=only_strong_connections,
                verbose_progress=verbose_progress
            )
            if (results_next_level is not None):
                # for result in results_next_level:
                #     if (result is not None):
                #         out.append(result)
                #out["nodes"].extend(results_next_level["nodes"])
                out["nodes"].update(results_next_level["nodes"])
                out["edges"].extend(results_next_level["edges"])

            num_added += 1

            if (progress_bar is not None):
                progress_bar.update(1)
                progress_bar.set_description(f"Crawling forwards from contribution {contribution_id} (depth {current_depth}/{max_depth}) - processing {next_contribution_id}")


        if (num_added > 0):
            # Add the original contribution as a node as well, if it doesn't already exist in the nodes.
            paper_title = None
            paper_corpus_id = None
            contribution_obj_json = None

            contribution_obj = self.get_contribution_by_id(contribution_id)
            contribution_obj_json = contribution_obj.model_dump() if contribution_obj is not None else None

            paper_title = None
            paper_corpus_id = None
            if (contribution_obj is not None):
                paper = self.load_paper(self.get_corpus_id_from_contribution(contribution_id))
                if paper is not None:
                    paper_title = paper.title
                    paper_corpus_id = paper.corpus_id

            new_node = {
                "contribution_id": contribution_id,
                "paper_title": paper_title,
                "paper_corpus_id": paper_corpus_id,
                "contribution_obj": contribution_obj_json,
            }

            # Add it to the nodes if it's not already there.
            if contribution_id not in out["nodes"]:
                out["nodes"][contribution_id] = new_node

        return out
