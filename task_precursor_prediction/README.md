# Scientific Contribution Graph: Technological Prerequisite Prediction

The technological prerequisite predictiont task takes a target new technology as input (e.g. a cartoon example: `A novel high-recall novelty detection system that operationalizes novelty mechanistically`),
and must predict the existing technologies in the scientific contribution graph that would be most likely to be required to build this new technology. 

The code here frames this as a ranking task:
- The user/scientific discovery agent provides a description of the new technology they'd like to develop
- The description of the new technology is used to perform an embedding-based search over the Scientific Contribution Graph, and return the top-n highest matching technologies (based on cosine similarity)
- A language model is used as a reranker, by being provided with the shortlist of existing technologies, and tasked to selectively rank which ones are most likely to be useful.

The result is a ranked list of existing technologies, each a scientific contribution from a paper in the scientific contribution graph.  They are ranked in decreasing order of relevance (i.e. the highest-ranked technology is nominally the most relevant). 

### **Table of Contents: Code in this repository**
The code in this directory includes: 
1. **Runnable Example:** A simple end-to-end runnable example function, that can be directly used to apply this task for real-world technological requirement prediction tasks.
2. **Benchmarking Code:** The code and associated data for the benchmarking experiments in the paper (Table 3), to replicate the results, or run new models. 

### 1. Runnable Example

The `precursor_prediction_example.py` file includes a simple, end-to-end, runnable example.  Simply replace the example technology with the description of the technology you'd like to develop:
```
    # Load the graph (must enable search)    
    graph = ScientificContributionGraph(path=path_to_graph_data, search_enabled=True, search_device="cpu")

    # Example: predict prerequisites for a hypothetical technology
    technology_name = "Mechanistic Novelty Detection System for Scientific Research"
    technology_description = "A novelty detection system for scientific research, that achieves high-recall detection by operationalizing novelty detection mechanistically.\n"

    #model_str = "gpt-5-mini"
    model_str = "claude-haiku-4-5"

    result = technology_requirement_prediction(
        technology_description=technology_description,
        model_str=model_str,
        graph=graph,
        technology_name=technology_name,
        top_k_search=30,
        use_reflection=False,
    )
```

Example output (for the hypothetical novelty example above) is in `precursor_prediction_example_output.txt` and `precursor_prediction_example_output.json`.  The top handful of results are shown here: 
```
================================================================================

Technology requirement prediction for: Mechanistic Novelty Detection System for Scientific Research
Description: A novelty detection system for scientific research, that achieves high-recall detection by operationalizing novelty detection mechanistically.

================================================================================
Ranked predicted prerequisites:
================================================================================

[Rank 1] Idea Novelty Checker: Retrieval‑augmented LLM system for literature‑grounded novelty assessment
    Contribution ID: 280012372.c0
    Description:     The paper presents Idea Novelty Checker, an end‑to‑end pipeline that automatically judges the novelty of a scientific idea. It first retrieves a broad set of potentially relevant papers using keyword expansion, snippet search, and seed‑paper recommendation via the Semantic Scholar API. The candidate set is filtered with SPECTER‑2 embedding similarity and then re‑ranked by a facet‑aware LLM re‑ranker (RankGPT) that scores papers on purpose, mechanism, evaluation, and application facets. Finally, GPT‑4o is prompted with expert‑annotated in‑context examples to produce a binary novelty decision together with a literature‑grounded rationale. The implementation, prompts, and a curated expert‑labeled dataset are released as open‑source resources.
    Paper:           Literature-Grounded Novelty Assessment of Scientific Ideas (2025)
    Search cosine:   0.599
    Explanation:     End-to-end retrieval-augmented system specifically designed for assessing novelty of scientific ideas with facet-aware re-ranking and LLM-based judgment.

[Rank 2] Facet‑based definition of scientific idea novelty
    Contribution ID: 280012372.c1
    Description:     The authors define novelty of a scientific idea as a deviation in at least one of three core facets—purpose (the problem addressed), mechanism (the technical approach), or evaluation (the validation method)—or as a novel combination or application of these facets. This definition is derived from a formative expert study and builds on prior work that decomposes research contributions into purpose, mechanism, and evaluation facets. The facet‑based definition serves both as a conceptual framework for assessing novelty and as an operational metric that guides retrieval, re‑ranking, and in‑context prompting within the Idea Novelty Checker system.
    Paper:           Literature-Grounded Novelty Assessment of Scientific Ideas (2025)
    Search cosine:   0.640
    Explanation:     Mechanistic definition of scientific novelty via purpose, mechanism, and evaluation facets; directly operationalizes how novelty should be assessed.

[Rank 3] Method for detecting scientific novelty via conceptual link recombination
    Contribution ID: 210718657.c3
    Description:     The paper introduces a scalable text‑based pipeline that (i) extracts salient scientific concepts from millions of dissertation abstracts using structural topic modeling (STM) and FREX scoring, (ii) identifies novel conceptual co‑occurrences (new links) by comparing each thesis to the accumulated concept network and applying a log‑odds significance filter, and (iii) quantifies a thesis's novelty as the count of such new links. This operationalization treats scientific innovation as the introduction of previously unseen concept pairings in the scholarly corpus and enables population‑scale measurement of novelty across three decades of US PhD dissertations.
    Paper:           The Diversity–Innovation Paradox in Science (2019)
    Search cosine:   0.607
    Explanation:     Scalable mechanistic pipeline that operationalizes scientific novelty as novel concept co-occurrences in scholarly corpora using topic modeling.

[Rank 4] Formative expert study on novelty evaluation challenges
    Contribution ID: 280012372.c4
    Description:     The authors performed a two‑phase expert study on 51 scientific ideas (46 generated by the Scideator system and 5 from real conference submissions) to assess novelty judgments. They measured inter‑annotator agreement with Cohen's Kappa (0.64 → 0.68) and identified two primary challenges: (i) the inherent subjectivity of novelty judgments and (ii) the difficulty of retrieving all relevant prior work when relying on keyword‑only retrieval. These findings motivated the design of a high‑quality retrieval pipeline and the facet‑based novelty definition used in the Idea Novelty Checker.
    Paper:           Literature-Grounded Novelty Assessment of Scientific Ideas (2025)
    Search cosine:   0.591
    Explanation:     Expert study identifying key challenges in scientific novelty assessment (subjectivity, retrieval difficulty) that motivate system design.

[Rank 5] Analysis of mismatch between normal‑text and research‑paper novelty detection
    Contribution ID: 207909527.c4
    Description:     The paper empirically demonstrates that traditional normal‑text novelty detection methods (e.g., TF‑IDF weighting and co‑occurrence graph features) perform poorly when applied to scholarly articles, whereas features derived from entity‑based citation graphs (keyword and topic graphs) capture both the static and dynamic aspects of research‑paper novelty as defined in prior literature. Quantitative experiments on a large arXiv corpus and a small human annotation study support the claim that novelty in scholarly documents is fundamentally different from novelty in ordinary text. The analysis explains why entity‑based citation features are more effective for detecting novel research contributions.
    Paper:           Evaluating Research Novelty Detection: Counterfactual Approaches (2019)
    Search cosine:   0.637
    Explanation:     Demonstrates that research-paper novelty requires entity-citation features rather than standard text-novelty methods; essential for scientific specificity.
```


### 2. Benchmarking Code

The benchmarking code helps replicate the results in the paper, or allows adding new results. 

**Data:** The 2000 example problems used to generate Table 3 in the paper are provided in: `prerequisite_prediction_problems.final_format.20260508-113303.zip`. 

**Code:** The benchmarking code is provided in `task_precursor_prediction_model.py`. The following notes apply: 
- If you are adding a model not already described, don't forget to add its knowledge cutoff.
- The `DEBUG_LIMIT` controls the number of problems to run.  It's recommended to initially run this on a small number of problems (e.g. 10) to verify the model is functioning correctly, before running it on a large number. (`DEBUG_LIMIT = None` will run all problems).
- The `NUM_WORKERS` paramater controls the number of parallel threads, and will depend upon your rate limits. 
- The output will be placed in `precursor_prediction_outputs/`, and include separate files for each problem and its associated output, as well as a summary file describing overall performance across the run.

**Generating your own data:** If you need to regenerate a larger set of problems (i.e. for models with newer knowledge cutoff dates), the generation script is provided in `task_precursor_prediction_generation.py` for reference.
