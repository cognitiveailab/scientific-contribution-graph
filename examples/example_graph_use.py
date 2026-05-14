# example_graph_use.py

import json

from scicontgraph import ScientificContributionGraph
from scicontgraph.ScientificContributionGraphVisualization import *



# Papers are indexed by their Semantic Scholar (S2) corpus ID.
# Using the Semantic Scholar API to look-up papers is best, but here we also provide a quick title search functionality. This is not intended to be particularly robust, but more a rough-and-ready method for testing.
def example1_find_paper_by_title(graph):
    # Helper
    def search_and_print(paper_title):
        corpus_id_list = graph.get_corpus_id_from_paper_title(paper_title=paper_title, top_k=10)
        print("Search results for paper title: `" + str(paper_title) + "`")
        for search_result in corpus_id_list:
            print(search_result)
        print("")


    # Example 1A: You have the correct paper title and there is one exact match
    search_and_print("Explaining Answers with Entailment Trees")

    # Example output (1A):
    # Search results for paper title: `Explaining Answers with Entailment Trees`
    # {'score': 1.0, 'corpus_id': '233297051', 'paper_title': 'Explaining Answers with Entailment Trees'}


    # Example 1B: You have only a partial paper title
    search_and_print("Explaining Answers entailment Trees")

    # Example output (1B):
    # {'score': 1.0, 'corpus_id': '233297051', 'paper_title': 'Explaining Answers with Entailment Trees'}
    # {'score': 0.5, 'corpus_id': '14805503', 'paper_title': 'DLSITE-2: Semantic Similarity Based on Syntactic Dependency Trees Applied to Textual Entailment'}
    # {'score': 0.5, 'corpus_id': '250390963', 'paper_title': 'Explaining Neural NLP Models for the Joint Analysis of Open- and Closed-Ended Survey Answers'}
    # {'score': 0.5, 'corpus_id': '275821195', 'paper_title': 'It is not a piece of cake for GPT: Explaining Textual Entailment Recognition in the presence of Figurative Language'}
    # {'score': 0.5, 'corpus_id': '195791777', 'paper_title': 'Pentagon at MEDIQA 2019: Multi-task Learning for Filtering and Re-ranking Answers using Language Inference and Question Entailment'}
    # {'score': 0.5, 'corpus_id': '253237103', 'paper_title': 'RLET: A Reinforcement Learning Based Approach for Explainable QA with Entailment Trees'}
    # {'score': 0.5, 'corpus_id': '3129209', 'paper_title': 'Similarity between Pairs of Co-indexed Trees for Textual Entailment Recognition'}
    # {'score': 0.5, 'corpus_id': '268091324', 'paper_title': 'TV-TREES: Multimodal Entailment Trees for Neuro-Symbolic Video Reasoning'}
    # {'score': 0.5, 'corpus_id': '4052735', 'paper_title': 'VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions'}
    # {'score': 0.25, 'corpus_id': '264146913', 'paper_title': '"Are Your Explanations Reliable?" Investigating the Stability of LIME in Explaining Text Classifiers by Marrying XAI and Adversarial Attack'}


def example2_display_paper_contributions(graph, corpus_id):
    paper = graph.load_paper(corpus_id)

    if paper is None:
        print(f"Could not load paper with corpus_id={corpus_id}")
        return

    print("=" * 80)
    print("Paper object")
    print("=" * 80)
    print(f"Corpus ID: {paper.corpus_id}")
    print(f"Title:     {paper.title}")
    print(f"Year:      {paper.year}")
    print(f"Date:      {paper.publication_date}")
    print(f"Number of contributions: {len(paper.contributions)}")
    print("")

    # print("=" * 80)
    # print("Paper JSON structure")
    # print("=" * 80)
    # print(json.dumps(paper.model_dump(mode="json"), indent=4))
    # print("")

    print("=" * 80)
    print("Contributions and Prerequisites")
    print("=" * 80)

    for i, contribution in enumerate(paper.contributions, start=1):
        print(f"\n[Contribution {i}] {contribution.name}")
        print(f"    Contribution ID: {contribution.contribution_id}")
        print(f"    Description: {contribution.description}")

        if contribution.types:
            print("    Types:")
            for contribution_type in contribution.types:
                print(f"      - {contribution_type.type}: {contribution_type.explanation}")

        if contribution.sections:
            print("    Sections:")
            for section in contribution.sections:
                print(f"      - {section}")

        if contribution.prerequisites:
            print("    Prerequisites:")
            for j, prereq in enumerate(contribution.prerequisites, start=1):
                print(f"      [Prerequisite {i}.{j}] {prereq.name}")
                print(f"          Description: {prereq.description}")
                print(f"          Explanation: {prereq.explanation}")
                print(f"          Core/peripheral: {prereq.core_or_peripheral}")

                if prereq.references:
                    print("          References:")
                    for ref in prereq.references:
                        if ref.type == "paper":
                            print(f"            - Paper: {ref.paper_title} ({ref.paper_year})")
                            print(f"              First author: {ref.paper_first_author}")
                            print(f"              Corpus ID: {ref.corpus_id}")

                            if ref.matches:
                                print("              Matched contributions:")
                                for match in ref.matches:
                                    #print(f"                - {match.contribution_id} ({match.match_type}) via {match.match_method}")
                                    print(f"                - {match.contribution_id} ({match.match_type}). Explanation: {match.explanation}")

                        elif ref.type == "internal":
                            print(f"            - Internal: {ref.contribution_name}")
                            print(f"              Contribution ID: {ref.contribution_id}")

                        elif ref.type == "other":
                            print(f"            - Other: {ref.name}")
                            print(f"              URL: {ref.url}")


    # Example output (truncated):
    # ================================================================================
    # Paper object
    # ================================================================================
    # Corpus ID: 233297051
    # Title:     Explaining Answers with Entailment Trees
    # Year:      2021
    # Date:      {'year': 2021, 'month': 4, 'day': 17}
    # Number of contributions: 11
    #
    # ================================================================================
    # Contributions and Prerequisites
    # ================================================================================
    #
    # [Contribution 1] Formulation of explanation as multistep entailment trees
    #     Contribution ID: 233297051.c0
    #     Description: The paper reconceives open-domain textual question-answering explanation as the construction of a directed entailment tree, where each node is a multi-premise textual entailment step that incrementally derives the hypothesis (question + answer) from known facts. This formulation separates the correctness of the derivation from its utility for users, enabling systematic evaluation of explanations independent of answer selection. By defining explanations as trees rather than isolated rationales, the work provides a high-level conceptual framework that can guide future research on explainable QA.
    #     Types:
    #       - problem_formulation: It proposes a novel way to define the explanation task for QA.
    #       - conceptual_framework: It introduces the entailment-tree framework that structures reasoning steps.
    #     Sections:
    #       - Introduction
    #       - Task Definitions
    #     Prerequisites:
    #       [Prerequisite 1.1] Multi-premise textual entailment
    #           Description: Multi-premise textual entailment extends standard natural-language inference to infer a hypothesis from two or more premises, requiring models to combine information across sentences.
    #           Explanation: The entailment-tree formulation relies on each node being a valid multi-premise entailment step.
    #           Core/peripheral: core
    #           References:
    #             - Paper: Recognizing Textual Entailment: Models and Applications (2013)
    #               First author: first_name='Ido' last_name='Dagan'
    #               Corpus ID: None
    #             - Paper: Natural language inference from multiple premises (2017)
    #               First author: first_name='Alice' last_name='Lai'
    #               Corpus ID: 29033327
    #               Matched contributions:
    #                 - 29033327.c0 (strong). Explanation: The paper formulates the Multiple Premise Entailment (MPE) task, explicitly defining a textual entailment problem that requires inference from several premises, which is exactly the concept of multi‑premise textual entailment.
    #                 - 29033327.c1 (weak). Explanation: The creation of the MPE dataset provides a concrete resource for training and evaluating models on multi‑premise entailment, supporting the prerequisite but not defining the concept itself.
    #       [Prerequisite 1.2] Prior explanation approaches using rationales
    #           Description: Existing QA explanation methods provide short textual rationales or supporting sentences rather than a full chain of reasoning.
    #           Explanation: These works motivate the need for a richer, structured explanation format and are contrasted with the entailment-tree approach.
    #           Core/peripheral: peripheral
    #           References:
    #             - Paper: ERASER: A benchmark to evaluate rationalized NLP models (2019)
    #               First author: first_name='Jay' last_name='DeYoung'
    #               Corpus ID: 207847663
    #               Matched contributions:
    #                 - 207847663.c2-1 (strong). Explanation: Implements hard-selection rationalizer models that generate short textual rationales for predictions, exemplifying prior explanation approaches using rationales.
    #                 - 207847663.c2-2 (strong). Explanation: Implements soft-selection rationalizer models that produce continuous importance scores as short rationales, representing another prior rationale‑based explanation method.
    #             - Paper: Explain yourself! Leveraging language models for commonsense reasoning (2019)
    #               First author: first_name='Nazneen' last_name='Rajani'
    #               Corpus ID: 174803111
    # ...



# Example 2A: As above, but converting from the internal storage classes to JSON.
# Note: This is essentially the same as just loading the raw JSON files stored in the `/data/papers/` directory (but does this through the API, instead of manually)
def example2a_display_paper_contributions_as_json(graph, corpus_id, filename_out:str="examples/example2a_paper_contributions.json"):
    paper = graph.load_paper(corpus_id)

    if paper is None:
        print(f"Could not load paper with corpus_id={corpus_id}")
        return

    paper_json = paper.model_dump(mode="json")

    print("=" * 80)
    print("Paper JSON structure")
    print("=" * 80)
    print(json.dumps(paper_json, indent=4))
    print("")

    print("Writing JSON to: " + filename_out)
    with open(filename_out, "w") as f:
        json.dump(paper_json, f, indent=4)

    # Example output (truncated):
    # ================================================================================
    # Paper JSON structure
    # ================================================================================
    # {
    #     "corpus_id": "233297051",
    #     "title": "Explaining Answers with Entailment Trees",
    #     "year": 2021,
    #     "publication_date": {
    #         "year": 2021,
    #         "month": 4,
    #         "day": 17
    #     },
    #     "contributions": [
    #         {
    #             "contribution_id": "233297051.c0",
    #             "name": "Formulation of explanation as multistep entailment trees",
    #             "description": "The paper reconceives open-domain textual question-answering explanation as the construction of a directed entailment tree, where each node is a multi-premise textual entailment step that incrementally derives the hypothesis (question + answer) from known facts. This formulation separates the correctness of the derivation from its utility for users, enabling systematic evaluation of explanations independent of answer selection. By defining explanations as trees rather than isolated rationales, the work provides a high-level conceptual framework that can guide future research on explainable QA.",
    #             "types": [
    #                 {
    #                     "type": "problem_formulation",
    #                     "explanation": "It proposes a novel way to define the explanation task for QA."
    #                 },
    #                 {
    #                     "type": "conceptual_framework",
    #                     "explanation": "It introduces the entailment-tree framework that structures reasoning steps."
    #                 }
    #             ],
    #             "sections": [
    #                 "Introduction",
    #                 "Task Definitions"
    #             ],
    #             "prerequisites": [
    #                 {
    #                     "name": "Multi-premise textual entailment",
    #                     "description": "Multi-premise textual entailment extends standard natural-language inference to infer a hypothesis from two or more premises, requiring models to combine information across sentences.",
    #                     "explanation": "The entailment-tree formulation relies on each node being a valid multi-premise entailment step.",
    #                     "core_or_peripheral": "core",
    #                     "references": [
    #                         {
    #                             "type": "paper",
    #                             "paper_title": "Recognizing Textual Entailment: Models and Applications",
    #                             "paper_year": 2013,
    #                             "paper_first_author": {
    #                                 "first_name": "Ido",
    #                                 "last_name": "Dagan"
    #                             },
    #                             "paper_venue": "Morgan and Claypool",
    #                             "corpus_id": null,
    #                             "corpus_id_match_confidence": 0.0,
    #                             "corpus_id_match_method": null,
    #                             "matches": []
    #                         },
    #                         {
    #                             "type": "paper",
    #                             "paper_title": "Natural language inference from multiple premises",
    #                             "paper_year": 2017,
    #                             "paper_first_author": {
    #                                 "first_name": "Alice",
    #                                 "last_name": "Lai"
    #                             },
    #                             "paper_venue": "IJCNLP",
    #                             "corpus_id": "29033327",
    #                             "corpus_id_match_confidence": 1.0,
    #                             "corpus_id_match_method": "title_exact_sanitized",
    #                             "matches": [
    #                                 {
    #                                     "contribution_id": "29033327.c0",
    #                                     "explanation": "The paper formulates the Multiple Premise Entailment (MPE) task, explicitly defining a textual entailment problem that requires inference from several premises, which is exactly the concept of multi\u2011premise textual entailment.",
    #                                     "match_type": "strong",
    #                                     "match_method": "alignment_v1_feb14"
    #                                 },
    #                                 {
    #                                     "contribution_id": "29033327.c1",
    #                                     "explanation": "The creation of the MPE dataset provides a concrete resource for training and evaluating models on multi\u2011premise entailment, supporting the prerequisite but not defining the concept itself.",
    #                                     "match_type": "weak",
    #                                     "match_method": "alignment_v1_feb14"
    #                                 }
    #                             ]
    #                         }
    #                     ]
    #                 },
    #                 {
    #                     "name": "Prior explanation approaches using rationales",
    #                     "description": "Existing QA explanation methods provide short textual rationales or supporting sentences rather than a full chain of reasoning.",
    #                     "explanation": "These works motivate the need for a richer, structured explanation format and are contrasted with the entailment-tree approach.",
    #                     "core_or_peripheral": "peripheral",
    #                     "references": [
    #                         {
    #                             "type": "paper",
    #                             "paper_title": "ERASER: A benchmark to evaluate rationalized NLP models",
    #                             "paper_year": 2019,
    #                             "paper_first_author": {
    #                                 "first_name": "Jay",
    #                                 "last_name": "DeYoung"
    #                             },
    #                             "paper_venue": "ACL",
    #                             "corpus_id": "207847663",
    #                             "corpus_id_match_confidence": 1.0,
    #                             "corpus_id_match_method": "title_exact_sanitized",
    #                             "matches": [
    #                                 {
    #                                     "contribution_id": "207847663.c2-1",
    #                                     "explanation": "Implements hard-selection rationalizer models that generate short textual rationales for predictions, exemplifying prior explanation approaches using rationales.",
    #                                     "match_type": "strong",
    #                                     "match_method": "alignment_v1_feb14"
    #                                 },
    #                                 {
    #                                     "contribution_id": "207847663.c2-2",
    #                                     "explanation": "Implements soft-selection rationalizer models that produce continuous importance scores as short rationales, representing another prior rationale\u2011based explanation method.",
    #                                     "match_type": "strong",
    #                                     "match_method": "alignment_v1_feb14"
    #                                 }
    #                             ]
    #                         },
    #                         {
    #                             "type": "paper",
    #                             "paper_title": "Explain yourself! Leveraging language models for commonsense reasoning",
    #                             "paper_year": 2019,
    #                             "paper_first_author": {
    #                                 "first_name": "Nazneen",
    #                                 "last_name": "Rajani"
    #                             },
    #                             "paper_venue": "ACL",
    #                             "corpus_id": "174803111",
    #                             "corpus_id_match_confidence": 1.0,
    #                             "corpus_id_match_method": "title_exact_sanitized",
    #                             "matches": []
    #                         }
    #                     ]
    #                 },
    #                 ...



# Example: Retrieve a specific contribution node by its contribution ID, and display it.
def example3_get_contribution_by_id(graph, contribution_id):
    # Get the contribution node
    contribution = graph.get_contribution_by_id(contribution_id)
    # Also get the paper information for this contribution (e.g., to display the paper title alongside the contribution information)
    paper_info = graph.get_paper_info_by_contribution_id(contribution_id)

    if contribution is None:
        print(f"Could not find contribution with ID {contribution_id}")
        return

    print("=" * 80)
    print("Paper information for contribution ID " + contribution_id)
    print("=" * 80)
    if paper_info is not None:
        print(f"Corpus ID: {paper_info['corpus_id']}")
        print(f"Paper title: {paper_info['paper_title']}")
        print(f"Paper year: {paper_info['paper_year']}")
        print(f"Paper publication date: {paper_info['paper_publication_date']}")
    else:
        print("No paper information found for this contribution.")

    print("")

    print("=" * 80)
    print(f"Contribution {contribution_id}")
    print("=" * 80)
    print(f"Contribution Name: {contribution.name}")
    print(f"Contribution Description: {contribution.description}")
    print(f"Contribution Types: {[t.type for t in contribution.types]}")
    print(f"Sections (from paper): {contribution.sections}")
    print(f"Number of prerequisites: {len(contribution.prerequisites)}")

    # Show the prerequisites for this contribution
    for i, prereq in enumerate(contribution.prerequisites):
        print(f"\n[Prerequisite {i+1}] {prereq.name}")
        print(f"    Description: {prereq.description}")
        print(f"    Explanation: {prereq.explanation}")
        print(f"    Core/peripheral: {prereq.core_or_peripheral}")

        if prereq.references:
            print("    References:")
            for ref in prereq.references:
                if ref.type == "paper":
                    print(f"      - Paper: {ref.paper_title} ({ref.paper_year})")
                    print(f"        First author: {ref.paper_first_author}")
                    print(f"        Corpus ID: {ref.corpus_id}")

                    if ref.matches:
                        print("        Matched contributions:")
                        for match in ref.matches:
                            print(f"          - {match.contribution_id} ({match.match_type}). Explanation: {match.explanation}")

                elif ref.type == "internal":
                    print(f"      - Internal: {ref.contribution_name}")
                    print(f"        Contribution ID: {ref.contribution_id}")

                elif ref.type == "other":
                    print(f"      - Other: {ref.name}")
                    print(f"        URL: {ref.url}")


    # Example output:
    # ================================================================================
    # Paper information for contribution ID 233297051.c0
    # ================================================================================
    # Corpus ID: 233297051
    # Paper title: Explaining Answers with Entailment Trees
    # Paper year: 2021
    # Paper publication date: {'year': 2021, 'month': 4, 'day': 17}
    #
    # ================================================================================
    # Contribution 233297051.c0
    # ================================================================================
    # Contribution Name: Formulation of explanation as multistep entailment trees
    # Contribution Description: The paper reconceives open-domain textual question-answering explanation as the construction of a directed entailment tree, where each node is a multi-premise textual entailment step that incrementally derives the hypothesis (question + answer) from known facts. This formulation separates the correctness of the derivation from its utility for users, enabling systematic evaluation of explanations independent of answer selection. By defining explanations as trees rather than isolated rationales, the work provides a high-level conceptual framework that can guide future research on explainable QA.
    # Contribution Types: ['problem_formulation', 'conceptual_framework']
    # Sections (from paper): ['Introduction', 'Task Definitions']
    # Number of prerequisites: 4
    #
    # [Prerequisite 1] Multi-premise textual entailment
    #     Description: Multi-premise textual entailment extends standard natural-language inference to infer a hypothesis from two or more premises, requiring models to combine information across sentences.
    #     Explanation: The entailment-tree formulation relies on each node being a valid multi-premise entailment step.
    #     Core/peripheral: core
    #     References:
    #       - Paper: Recognizing Textual Entailment: Models and Applications (2013)
    #         First author: first_name='Ido' last_name='Dagan'
    #         Corpus ID: None
    #       - Paper: Natural language inference from multiple premises (2017)
    #         First author: first_name='Alice' last_name='Lai'
    #         Corpus ID: 29033327
    #         Matched contributions:
    #           - 29033327.c0 (strong). Explanation: The paper formulates the Multiple Premise Entailment (MPE) task, explicitly defining a textual entailment problem that requires inference from several premises, which is exactly the concept of multi‑premise textual entailment.
    #           - 29033327.c1 (weak). Explanation: The creation of the MPE dataset provides a concrete resource for training and evaluating models on multi‑premise entailment, supporting the prerequisite but not defining the concept itself.
    #
    # [Prerequisite 2] Prior explanation approaches using rationales
    #     Description: Existing QA explanation methods provide short textual rationales or supporting sentences rather than a full chain of reasoning.
    #     Explanation: These works motivate the need for a richer, structured explanation format and are contrasted with the entailment-tree approach.
    #     Core/peripheral: peripheral
    #     References:
    #       - Paper: ERASER: A benchmark to evaluate rationalized NLP models (2019)
    #         First author: first_name='Jay' last_name='DeYoung'
    #         Corpus ID: 207847663
    #         Matched contributions:
    #           - 207847663.c2-1 (strong). Explanation: Implements hard-selection rationalizer models that generate short textual rationales for predictions, exemplifying prior explanation approaches using rationales.
    #           - 207847663.c2-2 (strong). Explanation: Implements soft-selection rationalizer models that produce continuous importance scores as short rationales, representing another prior rationale‑based explanation method.
    #       - Paper: Explain yourself! Leveraging language models for commonsense reasoning (2019)
    #         First author: first_name='Nazneen' last_name='Rajani'
    #         Corpus ID: 174803111
    #
    # [Prerequisite 3] Proof generation with language models
    #     Description: Transformer-based language models can generate formal or natural-language proofs by sequentially applying inference rules, demonstrating the feasibility of end-to-end generation of structured reasoning.
    #     Explanation: The EntailmentWriter model and the linearisation scheme are directly inspired by proof-generation systems such as ProofWriter.
    #     Core/peripheral: core
    #     References:
    #       - Paper: ProofWriter: Generating implications, proofs, and abductive statements over natural language (2021)
    #         First author: first_name='Oyvind' last_name='Tafjord'
    #         Corpus ID: 229371222
    #         Matched contributions:
    #           - 229371222.c0 (strong). Explanation: Introduces an iterative generative ProofWriter model that uses a T5 language model to sequentially generate proof steps, directly exemplifying proof generation with language models.
    #           - 229371222.c1 (strong). Explanation: Presents an All‑At‑Once generative baseline that also uses a T5 language model to produce an entire natural‑language proof in one pass, further demonstrating proof generation with language models.
    #       - Paper: Generative language modeling for automated theorem proving (2020)
    #         First author: first_name='Stanislas' last_name='Polu'
    #         Corpus ID: 221535103
    #         Matched contributions:
    #           - 221535103.c0 (strong). Explanation: GPT‑f is a transformer‑based system that directly generates Metamath proof steps, demonstrating end‑to‑end language‑model proof generation.
    #           - 221535103.c1 (strong). Explanation: The GOAL→PROOFSTEP objective formalises proof‑step generation as a conditional language‑modeling task, the exact paradigm cited as the prerequisite.
    #       - Paper: Learning to prove theorems by learning to generate theorems (2020)
    #         First author: first_name='Ming-Zhe' last_name='Wang'
    #         Corpus ID: 211132980
    #         Matched contributions:
    #           - 211132980.c0 (strong). Explanation: MetaGen is a neural forward‑generation system that synthesizes full Metamath theorems together with proof trees, directly demonstrating end‑to‑end proof generation with a transformer‑style language model.
    #           - 211132980.c2-2 (weak). Explanation: MetaGen‑RL‑LM trains the theorem generator with a language‑model‑based reward, showing a connection to language‑model‑driven proof generation, but the language model is used only for reward, not for the proof itself.
    #
    # [Prerequisite 4] Multi-hop QA datasets (ARC, WorldTree)
    #     Description: Datasets such as ARC and WorldTree provide science questions together with a corpus of domain-specific facts, establishing a setting where multi-step reasoning is required.
    #     Explanation: The formulation is applied to open-domain QA and uses these datasets as the source of facts and as motivation for needing structured explanations.
    #     Core/peripheral: peripheral
    #     References:
    #       - Paper: Think you have solved question answering? Try ARC, the AI2 reasoning challenge (2018)
    #         First author: first_name='Peter' last_name='Clark'
    #         Corpus ID: 3922816
    #         Matched contributions:
    #           - 3922816.c0 (strong). Explanation: The contribution introduces the ARC dataset, a multi-hop science QA benchmark that is explicitly cited as one of the prerequisite datasets.
    #       - Paper: WorldTree V2: A corpus of science-domain structured explanations and inference patterns supporting multi-hop inference (2020)
    #         First author: first_name='Zhengnan' last_name='Xie'
    #         Corpus ID: 218974301
    #         Matched contributions:
    #           - 218974301.c0 (strong). Explanation: The contribution releases WorldTree V2, a multi-fact explanation dataset that is a direct instance of the "WorldTree" multi-hop QA dataset referenced in the prerequisite.


# Example: Retrieve a specific contribution node by its contribution ID (as above), and display it as JSON.
def example3a_get_contribution_by_id_as_json(graph, contribution_id, filename_out:str="examples/example3a_specific_contribution.json"):
    contribution = graph.get_contribution_by_id(contribution_id)

    if contribution is None:
        print(f"Could not find contribution with ID {contribution_id}")
        return

    contribution_json = contribution.model_dump(mode="json")

    packed = contribution_json
    packed["paper_info"] = graph.get_paper_info_by_contribution_id(contribution_id)

    print("=" * 80)
    print(f"Contribution {contribution_id} (as JSON)")
    print("=" * 80)
    print(json.dumps(packed, indent=4))

    print("Writing JSON to: " + filename_out)
    with open(filename_out, "w") as f:
        json.dump(packed, f, indent=4)

    # Example output (truncated):
    # ================================================================================
    # Contribution 233297051.c0 (as JSON)
    # ================================================================================
    # {
    #     "contribution_id": "233297051.c0",
    #     "name": "Formulation of explanation as multistep entailment trees",
    #     "description": "The paper reconceives open-domain textual question-answering explanation as the construction of a directed entailment tree, where each node is a multi-premise textual entailment step that incrementally derives the hypothesis (question + answer) from known facts. This formulation separates the correctness of the derivation from its utility for users, enabling systematic evaluation of explanations independent of answer selection. By defining explanations as trees rather than isolated rationales, the work provides a high-level conceptual framework that can guide future research on explainable QA.",
    #     "types": [
    #         {
    #             "type": "problem_formulation",
    #             "explanation": "It proposes a novel way to define the explanation task for QA."
    #         },
    #         {
    #             "type": "conceptual_framework",
    #             "explanation": "It introduces the entailment-tree framework that structures reasoning steps."
    #         }
    #     ],
    #     "sections": [
    #         "Introduction",
    #         "Task Definitions"
    #     ],
    #     "prerequisites": [
    #         {
    #             "name": "Multi-premise textual entailment",
    #             "description": "Multi-premise textual entailment extends standard natural-language inference to infer a hypothesis from two or more premises, requiring models to combine information across sentences.",
    #             "explanation": "The entailment-tree formulation relies on each node being a valid multi-premise entailment step.",
    #             "core_or_peripheral": "core",
    #             "references": [
    #                 {
    #                     "type": "paper",
    #                     "paper_title": "Recognizing Textual Entailment: Models and Applications",
    #                     "paper_year": 2013,
    #                     "paper_first_author": {
    #                         "first_name": "Ido",
    #                         "last_name": "Dagan"
    #                     },
    #                     "paper_venue": "Morgan and Claypool",
    #                     "corpus_id": null,
    #                     "corpus_id_match_confidence": 0.0,
    #                     "corpus_id_match_method": null,
    #                     "matches": []
    #                 },
    #                 {
    #                     "type": "paper",
    #                     "paper_title": "Natural language inference from multiple premises",
    #                     "paper_year": 2017,
    #                     "paper_first_author": {
    #                         "first_name": "Alice",
    #                         "last_name": "Lai"
    #                     },
    #                     "paper_venue": "IJCNLP",
    #                     "corpus_id": "29033327",
    #                     "corpus_id_match_confidence": 1.0,
    #                     "corpus_id_match_method": "title_exact_sanitized",
    #                     "matches": [
    #                         {
    #                             "contribution_id": "29033327.c0",
    #                             "explanation": "The paper formulates the Multiple Premise Entailment (MPE) task, explicitly defining a textual entailment problem that requires inference from several premises, which is exactly the concept of multi\u2011premise textual entailment.",
    #                             "match_type": "strong",
    #                             "match_method": "alignment_v1_feb14"
    #                         },
    #                         {
    #                             "contribution_id": "29033327.c1",
    #                             "explanation": "The creation of the MPE dataset provides a concrete resource for training and evaluating models on multi\u2011premise entailment, supporting the prerequisite but not defining the concept itself.",
    #                             "match_type": "weak",
    #                             "match_method": "alignment_v1_feb14"
    #                         }
    #                     ]
    #                 }
    #             ]
    #         },
    #         {
    #             "name": "Prior explanation approaches using rationales",
    #             "description": "Existing QA explanation methods provide short textual rationales or supporting sentences rather than a full chain of reasoning.",
    #             "explanation": "These works motivate the need for a richer, structured explanation format and are contrasted with the entailment-tree approach.",
    #             "core_or_peripheral": "peripheral",
    #             "references": [
    #                 {
    #                     "type": "paper",
    #                     "paper_title": "ERASER: A benchmark to evaluate rationalized NLP models",
    #                     "paper_year": 2019,
    #                     "paper_first_author": {
    #                         "first_name": "Jay",
    #                         "last_name": "DeYoung"
    #                     },
    #                     "paper_venue": "ACL",
    #                     "corpus_id": "207847663",
    #                     "corpus_id_match_confidence": 1.0,
    #                     "corpus_id_match_method": "title_exact_sanitized",
    #                     "matches": [
    #                         {
    #                             "contribution_id": "207847663.c2-1",
    #                             "explanation": "Implements hard-selection rationalizer models that generate short textual rationales for predictions, exemplifying prior explanation approaches using rationales.",
    #                             "match_type": "strong",
    #                             "match_method": "alignment_v1_feb14"
    #                         },
    #                         {
    #                             "contribution_id": "207847663.c2-2",
    #                             "explanation": "Implements soft-selection rationalizer models that produce continuous importance scores as short rationales, representing another prior rationale\u2011based explanation method.",
    #                             "match_type": "strong",
    #                             "match_method": "alignment_v1_feb14"
    #                         }
    #                     ]
    #                 },
    #                 {
    #                     "type": "paper",
    #                     "paper_title": "Explain yourself! Leveraging language models for commonsense reasoning",
    #                     "paper_year": 2019,
    #                     "paper_first_author": {
    #                         "first_name": "Nazneen",
    #                         "last_name": "Rajani"
    #                     },
    #                     "paper_venue": "ACL",
    #                     "corpus_id": "174803111",
    #                     "corpus_id_match_confidence": 1.0,
    #                     "corpus_id_match_method": "title_exact_sanitized",
    #                     "matches": []
    #                 }
    #             ]
    #         },
    # ... (truncated) ...
    #     ],
    #     "paper_info": {
    #         "corpus_id": "233297051",
    #         "paper_title": "Explaining Answers with Entailment Trees",
    #         "paper_year": 2021,
    #         "paper_publication_date": {
    #             "year": 2021,
    #             "month": 4,
    #             "day": 17
    #         }
    #     }
    # }


# Backward crawling: Show all the scientific contributions that were needed (i.e. are prerequisites) for a given contribution, directly or indirectly.
def example4_backward_crawl(graph, contribution_id, filename_out:str="examples/backward_crawl_example_results.json"):
    backward_crawl_results = graph.crawl_backwards_from_contribution(contribution_id=contribution_id, max_depth=2, only_strong_connections=True)
    print("\n" + "=" * 80)
    print(f"Backward crawl results starting from contribution {contribution_id} (only strong connections, max depth 2)")
    print("Exporting Writing: " + filename_out)
    with open(filename_out, "w") as f:
        json.dump(backward_crawl_results, f, indent=4)

    # Example output (JSON):
    # {
    #     "root_node": "233297051.c0",
    #     "nodes": { # Dictionary of nodes (key = contribution_id, value = contribution information and metadata)
    #         "233297051.c0": {
    #             "contribution_id": "233297051.c0",
    #             "paper_title": "Explaining Answers with Entailment Trees",
    #             "paper_corpus_id": "233297051",
    #             "contribution_obj": {
    #                 "contribution_id": "233297051.c0",
    #                 "name": "Formulation of explanation as multistep entailment trees",
    #                 "description": "The paper reconceives open-domain textual question-answering explanation as the construction of a...",
    #                 "types": [...],
    #                 "sections": [...],
    #                 "prerequisites": [...]
    #         },
    #         "29033327.c0": { ... },
    #         # ... additional nodes ...
    #     },
    #     "edges": [ # List of edges (each edge is a dictionary with source and target contribution IDs, and information about the prerequisite relationship)
    #         {
    #             "contribution_id": "29033327.c0",
    #             "prerequisite_for_contribution_id": "233297051.c0",
    #             "prerequisite_description": "...",
    #             "prerequisite_explanation": "...",
    #             "strengths": ["strong"],
    #             "depth": 1
    #         },
    #         {
    #             "contribution_id": "...",
    #             "prerequisite_for_contribution_id": "...",
    #             ...
    #         },
    #         ...
    #     ]
    # }


# Backward crawling: Show all the scientific contributions that were needed (i.e. are prerequisites) for a given contribution, directly or indirectly.
def example4_backward_crawl(graph, contribution_id, filename_out:str="examples/backward_crawl_example_results.json", export_visualization:bool=True):
    print("\n" + "=" * 80)
    print(f"Backward crawl results starting from contribution {contribution_id} (only strong connections, max depth 2)")
    print("")
    backward_crawl_results = graph.crawl_backwards_from_contribution(contribution_id=contribution_id, max_depth=2, only_strong_connections=True)
    print("")
    print("Exporting Writing: " + filename_out)
    print("\n" + "=" * 80)
    with open(filename_out, "w") as f:
        json.dump(backward_crawl_results, f, indent=4)

    # Example output (see forward crawling example; same format)

    # Visualization
    if (export_visualization == True):
        # DOT/Graphviz visualization
        try:
            filename_out_prefix = filename_out.replace(".json", ".graphviz")
            convert_crawl_results_to_dot(backward_crawl_results, filename_out_prefix, crawl_direction="backward")
        except Exception as e:
            print("Error exporting DOT/Graphviz visualization: " + str(e))

        # DOT/Graphviz visualization (with edge labels as intermediate nodes)
        try:
            filename_out_prefix = filename_out.replace(".json", ".graphviz.edgelabels")
            convert_crawl_results_to_dot_with_edge_nodes(backward_crawl_results, filename_out_prefix, crawl_direction="backward", edge_label_width=54, edge_label_max_chars=900, group_parallel_edges=True, semantic_arrowheads=True)
        except Exception as e:
            print("Error exporting DOT/Graphviz visualization with edge labels: " + str(e))


        # Radial
        try:
            filename_out_radial_svg = filename_out.replace(".json", ".radial.svg")
            #export_forward_crawl_results_to_radial_tree_svg_v3c(backward_crawl_results, filename_out_radial_svg,
            export_crawl_results_to_radial_tree_svg(backward_crawl_results, filename_out_radial_svg,
                                                                node_diameter_px=120, min_gap_px=30, radial_step_px=150, margin_px=40,
                                                                contraction_iterations=1500, gravity_k=4*0.055, spring_k=2*0.020, node_repulsion_k=1.5*18.0, edge_repulsion_k=1.5*10.0, max_move_px=16.0, node_edge_gap_px=30.0, node_center_min_distance_px=150.0,
                                                                metadata_line_mode="paper_title",
                                                                min_children=0, min_thresh=0,       # Trim nodes that have fewer than `min_children`.  If `min_thresh` is set, it will add a square box to indicate the graph was reduced.
                                                                root_fill_color="#CCCCCC",
                                                                trim_root_leaf_summary_nodes=False)  # If enabled, makes the visualization cleaner by removing summary nodes that expand less.
        except Exception as e:
            print("Error exporting radial visualization: " + str(e))


# Forward crawling: Show all the contributions that build upon a given contribution (i.e., all the contributions that have this contribution as a prerequisite, directly or indirectly).
def example5_forward_crawl(graph, contribution_id, filename_out:str="examples/forward_crawl_example_results.json", export_visualization:bool=True):
    print("\n" + "=" * 80)
    print(f"Forward crawl results starting from contribution {contribution_id} (only strong connections, max depth 2)")
    print("")
    forward_crawl_results = graph.crawl_forwards_from_contribution(contribution_id=contribution_id, max_depth=2, only_strong_connections=True)
    print("")
    print("Exporting Writing: " + filename_out)
    print("\n" + "=" * 80)
    with open(filename_out, "w") as f:
        json.dump(forward_crawl_results, f, indent=4)


    # Visualization
    if (export_visualization == True):
        # DOT/Graphviz visualization
        try:
            filename_out_prefix = filename_out.replace(".json", ".graphviz")
            convert_crawl_results_to_dot(forward_crawl_results, filename_out_prefix, crawl_direction="forward")
        except Exception as e:
            print("Error exporting DOT/Graphviz visualization: " + str(e))

        # DOT/Graphviz visualization (with edge labels as intermediate nodes)
        try:
            filename_out_prefix = filename_out.replace(".json", ".graphviz.edgelabels")
            convert_crawl_results_to_dot_with_edge_nodes(forward_crawl_results, filename_out_prefix, crawl_direction="backward", edge_label_width=54, edge_label_max_chars=900, group_parallel_edges=True, semantic_arrowheads=True)
        except Exception as e:
            print("Error exporting DOT/Graphviz visualization with edge labels: " + str(e))

        # Radial
        try:
            filename_out_radial_svg = filename_out.replace(".json", ".radial.svg")
            export_crawl_results_to_radial_tree_svg(forward_crawl_results, filename_out_radial_svg,
                                                                node_diameter_px=120, min_gap_px=30, radial_step_px=150, margin_px=40,
                                                                contraction_iterations=1500, gravity_k=4*0.055, spring_k=2*0.020, node_repulsion_k=1.5*18.0, edge_repulsion_k=1.5*10.0, max_move_px=16.0, node_edge_gap_px=30.0, node_center_min_distance_px=150.0,
                                                                metadata_line_mode="paper_title",
                                                                min_children=0, min_thresh=0,       # Trim nodes that have fewer than `min_children`.  If `min_thresh` is set, it will add a square box to indicate the graph was reduced.
                                                                root_fill_color="#CCCCCC",
                                                                trim_root_leaf_summary_nodes=False)  # If enabled, makes the visualization cleaner by removing summary nodes that expand less.
        except Exception as e:
            print("Error exporting radial visualization: " + str(e))




    # Example output (JSON):
    # {
    #    "root_node": "233297051.c0",
    #    "nodes": { # Dictionary of nodes (key = contribution_id, value = contribution information and metadata)
    #        "233297051.c0": {
    #            "contribution_id": "233297051.c0",
    #            "paper_title": "Explaining Answers with Entailment Trees",
    #            "paper_corpus_id": "233297051",
    #            "contribution_obj": {
    #                "contribution_id": "233297051.c0",
    #                "name": "Formulation of explanation as multistep entailment trees",
    #                "description": "The paper reconceives open-domain textual question-answering explanation as the construction of a...",
    #                "types": [...],
    #                "sections": [...],
    #                "prerequisites": [...]
    #        },
    #        "29033327.c0": { ... },
    #        # ... additional nodes ...
    #    },
    #    "edges": [ # List of edges (each edge is a dictionary with source
    #        {
    #            "contribution_id": "233297051.c0",
    #            "used_by_contribution_id": "258479954.c0",
    #            "prerequisite_description": "Defines the task of answering a question while providing a valid entailment tree whose intermediate conclusions logically support the answer, as introduced in prior work on faithful QA and entailment\u2011tree explanations.",
    #            "prerequisite_explanation": "FAME\u2019s problem formulation builds directly on the definition of FQA and the entailment\u2011tree representation; without this prior task definition the new decision\u2011making framing would have no target.",
    #            "strengths": ["strong"],
    #            "depth": 0
    #        },
    #        {
    #            "contribution_id": "233297051.c0",
    #            "used_by_contribution_id": "273641238.c0",
    #            "prerequisite_description": "The concept of constructing a tree of premises, intermediate conclusions, and a hypothesis to explain reasoning, originally introduced in the EntailmentBank dataset and subsequent entailment\u2011tree generation work.",
    #            "prerequisite_explanation": "The joint formulation directly adopts this paradigm to structure multimodal evidence, so the underlying theory and prior methods for entailment\u2011tree generation are required.",
    #            "strengths": ["strong"],
    #            "depth": 0
    #        },
    #        # ... additional edges ...
    #    ]
    # }



def example6_impact_metric(graph, paper_id, filename_out:str="examples/example6_impact_metric.json"):
    impact_metric = graph.calculate_impact_metric_paper(corpus_id=paper_id)
    print("\n" + "=" * 80)
    print("Impact metric for paper ID " + paper_id + ":")
    print("Writing impact metric results to: " + filename_out)
    print(json.dumps(impact_metric, indent=4))
    print("\n" + "=" * 80)

    with open(filename_out, "w") as f:
        json.dump(impact_metric, f, indent=4)

    # Example output:
    # Read as:
    # - (Contribution-level): 619 unique scientific contributions directly or indirectly build upon `ENTAILMENTBANK: a large-scale dataset of multistep entailment trees for elementary-science QA`
    # - (Paper-level): 631 unique scientific contributions directly or indirectly build upon any/all contributions from the paper `Explaining Answers with Entailment Trees`
    # The dampened metric applies a reciprocal rank of depth weighting.  Direct citations count as 1, contributions at depth 2 count as 0.5, contributions at depth 3 count as 0.33, etc.

    # ================================================================================
    # Impact metric for paper ID 233297051:
    # {
    #     "corpus_id": "233297051",
    #     "paper_info": {
    #         "corpus_id": "233297051",
    #         "paper_title": "Explaining Answers with Entailment Trees",
    #         "paper_year": 2021,
    #         "paper_publication_date": {
    #             "year": 2021,
    #             "month": 4,
    #             "day": 17
    #         }
    #     },
    #     "max_depth": 5,
    #     "contribution_impact_scores": {
    #         "233297051.c0": {
    #             "contribution_name": "Formulation of explanation as multistep entailment trees",
    #             "impact_score": 69.0,
    #             "impact_score_dampened": 48.33333333333333
    #         },
    #         "233297051.c1": {
    #             "contribution_name": "ENTAILMENTBANK: a large-scale dataset of multistep entailment trees for elementary-science QA",
    #             "impact_score": 619.0,
    #             "impact_score_dampened": 335.2333333333316
    #         },
    #         "233297051.c2": {
    #             "contribution_name": "Web\u2011based drag\u2011and\u2011drop authoring tool for constructing entailment trees",
    #             "impact_score": 3.0,
    #             "impact_score_dampened": 3.0
    #         },
    #         "233297051.c3": {
    #             "contribution_name": "Multi\u2011stage relevant\u2011fact retrieval pipeline for the full\u2011corpus explanation task",
    #             "impact_score": 19.0,
    #             "impact_score_dampened": 8.833333333333332
    #         },
    #         "233297051.c4": {
    #             "contribution_name": "EntailmentWriter: T5\u2011based generative model for entailment\u2011tree generation",
    #             "impact_score": 85.0,
    #             "impact_score_dampened": 54.666666666666664
    #         },
    #         "233297051.c5": {
    #             "contribution_name": "Tree Alignment Algorithm for Evaluating Entailment Trees",
    #             "impact_score": 26.0,
    #             "impact_score_dampened": 24.0
    #         },
    #         "233297051.c6-1": {
    #             "contribution_name": "Empirical evaluation of EntailmentWriter on three explanation tasks",
    #             "impact_score": 0.0,
    #             "impact_score_dampened": 0.0
    #         },
    #         "233297051.c6-2": {
    #             "contribution_name": "Error analysis of EntailmentWriter revealing common failure modes",
    #             "impact_score": 3.0,
    #             "impact_score_dampened": 3.0
    #         },
    #         "233297051.c7": {
    #             "contribution_name": "Analysis of reasoning types required for multistep entailments",
    #             "impact_score": 16.0,
    #             "impact_score_dampened": 11.5
    #         },
    #         "233297051.c8-1": {
    #             "contribution_name": "Zero-shot out-of-domain generalization of EntailmentWriter",
    #             "impact_score": 0.0,
    #             "impact_score_dampened": 0.0
    #         },
    #         "233297051.c8-2": {
    #             "contribution_name": "Shredded\u2011tree interactive explanation generation",
    #             "impact_score": 14.0,
    #             "impact_score_dampened": 9.0
    #         }
    #     },
    #     "overall_paper_impact_score": {
    #         "impact_score": 631.0,
    #         "impact_score_dampened": 356.89999999999793
    #     }
    # }


# Search example: Find contributions related to a specific query string.
def example7_search_contributions(graph, query: str, top_k: int = 10, filename_out: str = "examples/example7_search_results.json"):
    import json
    import os

    print("=" * 80)
    print("Search Contributions")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Top K: {top_k}")
    print("")

    # This is the raw API return value.
    search_results = graph.search(query=query, top_n=top_k, populate_names=True)

    print("=" * 80)
    print(f"Search results for query: `{query}`")
    print("=" * 80)

    if len(search_results) == 0:
        print("No search results found.")
    else:
        for i, result in enumerate(search_results, start=1):
            contribution_id = result.get("contribution_id")
            corpus_id = result.get("corpus_id")
            cosine = result.get("cosine")

            contribution_name = result.get("contribution_name") or "No contribution name available."
            contribution_description = result.get("contribution_description") or "No contribution description available."

            paper_info = result.get("paper_info") or {}
            paper_title = paper_info.get("paper_title") or "No paper title available."
            paper_year = paper_info.get("paper_year")

            print("")
            print(f"[Result {i}]")
            print(f"    Contribution ID: {contribution_id}")
            print(f"    Corpus ID:       {corpus_id}")
            print(f"    Cosine score:    {cosine:.4f}" if isinstance(cosine, float) else f"    Cosine score:    {cosine}")
            print(f"    Paper title:     {paper_title}")
            print(f"    Paper year:      {paper_year}")
            print(f"    Contribution:    {contribution_name}")
            print(f"    Description:     {contribution_description}")

    if filename_out is not None:
        out_dir = os.path.dirname(filename_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        print("")
        print("Writing raw search API results to: " + filename_out)
        with open(filename_out, "w", encoding="utf-8") as f:
            json.dump(search_results, f, indent=4, ensure_ascii=False)

    return search_results




#
#  Examples
#
if __name__ == "__main__":
    # Toggle search functionality (this increases load time by several minutes, and uses approximately 12GB of memory)
    search_enabled = True
    #search_enabled = False

    path_to_graph_data = "/data-ssd2/scientific-contribution-graph/download/"

    # Load the graph
    graph = ScientificContributionGraph(path=path_to_graph_data, search_enabled=search_enabled, search_device="cpu")


    # Example 1: Find a corpus ID based on a paper title
    example1_find_paper_by_title(graph)


    # Example 2: Display all the contributions (and their prerequisites) for a given paper.
    example_corpus_id = "233297051"  # EntailmentBank paper
    example2_display_paper_contributions(graph, example_corpus_id)

    # Example 2A: As above, but converting from the internal storage classes to JSON.
    example2a_display_paper_contributions_as_json(graph, example_corpus_id, filename_out="examples/example2a_paper_contributions.json")


    # Example 3: Get a specific contribution node.
    contribution_id = "233297051.c0"
    example3_get_contribution_by_id(graph, contribution_id)

    # Example 3A: Get a specific contribution node, and display it as JSON.
    example3a_get_contribution_by_id_as_json(graph, contribution_id, filename_out="examples/example3a_specific_contribution.json")


    # Example 4: Backward crawling: Show all the scientific contributions that were needed (i.e. are prerequisites) for a given contribution, directly or indirectly.
    contribution_id = "233297051.c0"
    example4_backward_crawl(graph, contribution_id, filename_out="examples/backward_crawl_example_results.json")


    # Example 5: Forward crawling: Show all the contributions that build upon a given contribution (i.e., all the contributions that have this contribution as a prerequisite, directly or indirectly).
    contribution_id = "233297051.c0"
    example5_forward_crawl(graph, contribution_id, filename_out="examples/forward_crawl_example_results.json")


    # Example 6: A citation metric based on number of downstream contributions a given paper (or contribution) makes.
    paper_id = "233297051"
    example6_impact_metric(graph, paper_id, filename_out="examples/example6_impact_metric.json")

    # Search example: Find contributions related to a specific query string.
    if (search_enabled == True):
        query = "Task: This is a search task, to find highly related content similar to the query.\nQuery: entailment tree generation with language models"
        example7_search_contributions(graph, query, top_k=5, filename_out="examples/example7_search_results.json")