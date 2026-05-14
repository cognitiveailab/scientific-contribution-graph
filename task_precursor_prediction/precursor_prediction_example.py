# precusor_prediction_example.py
# This is a free-form example of the precursor prediction task.
#
# Given a free-form description of a desired technology, this function:
#   (1) Searches the scientific contribution graph for candidate prerequisites
#   (2) Asks an LLM to rank those candidates by how likely they are to be direct prerequisites
#   (3) Returns the ranked list (with explanations).

import json
import random
import time

from ExtractionUtils import *  # for getLLMResponseJSON, loadAPIKeys
from scicontgraph import ScientificContributionGraph


#
#   Prompt
#
def llm_technology_requirement_prompt(technology:dict, candidates:list, model_str:str, max_tokens:int=16000, temperature:float=0.0, use_reflection:bool=False, max_generation_time_seconds:int=600):
    def mkPrompt(technology:dict, candidates:list, reflection=None):
        prompt = "You are ScientistGPT, an expert AI scientist. You can answer any scientific problem correctly, faithfully, and accurately, using the highest scientific integrity.\n"
        prompt += "\n"
        prompt += "You must solve this task using the information provided, and your own knowledge. Do NOT call external tools, such as a search engine, paper database, or other tools to answer, as it may contaminate your knowledge.\n"
        prompt += "\n"
        prompt += "# Task\n"
        prompt += "This is a technological-requirement prediction task.\n"
        prompt += "You will be provided with a description of a technology that a user would like to build (the technology may be hypothetical, partially-developed, or fully real).\n"
        prompt += "You will also be provided with a list of candidate scientific contributions (the `possible prerequisites`), presented in randomized order, that may or may not be required to build the technology.\n"
        prompt += "Your task is to determine which of the `possible prerequisites` are most likely to be actual *direct* prerequisites to building the described technology.\n"
        prompt += "\n"
        prompt += "## Additional Instructions\n"
        prompt += "- This is a ranking task. You will generate a ranked list of the possible prerequisites (by their `id`), from most useful/direct to least useful/irrelevant for enabling the described technology.\n"
        prompt += "- Your task is to determine *direct* prerequisites. For example, as a cartoon example: a complicated quantum mechanics technology might technically have `multiplication` as a prerequisite, but it likely has many more direct prerequisites closer than this -- list the direct prerequisites first.\n"
        prompt += "- Consider methodological, theoretical, dataset, software, and engineering prerequisites where appropriate.\n"
        prompt += "\n"

        prompt += "# Desired Technology\n"
        prompt += "Here is the description of the technology the user would like to build:\n"
        prompt += "```\n"
        prompt += json.dumps(technology, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Possible Prerequisites\n"
        prompt += "Here is the list of possible prerequisites (in random order):\n"
        prompt += "```\n"
        prompt += json.dumps(candidates, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        if (reflection is not None):
            prompt += "# Reflection\n"
            prompt += "This is a reflection step. Previously, you generated the output below. Your task is to reflect on that output, and correct any errors, omissions, inaccuracies, or any other issues.\n"
            prompt += "```\n"
            prompt += json.dumps(reflection, indent=4) + "\n"
            prompt += "```\n"
            prompt += "\n"

        prompt += "# Output Instructions\n"
        prompt += "You are welcome to think as much as you would like before answering. Your answer must be in JSON format, between codeblocks (```). Your JSON must be valid JSON.\n"
        prompt += "Your JSON should be a dictionary with the following keys:\n"
        prompt += "- `ranked_prerequisites`: A list of dictionaries representing the possible prerequisites, ranked from most likely to be an actual direct prerequisite to least likely. The first element is the most likely. Each dictionary will have the following keys:\n"
        prompt += "  - `id`: The id of the possible prerequisite (corresponding to the id in the list of possible prerequisites above)\n"
        prompt += "  - `explanation`: A 1-sentence, concise, information-dense explanation of why this possible prerequisite is ranked at this position.\n"
        prompt += "\n"
        prompt += "## Output Guidelines\n"
        prompt += "- Your output list should include all the prerequisites in the list above (which is " + str(len(candidates)) + " possible prerequisites), but in ranked order. You may omit items only if they are completely unrelated.\n"
        prompt += "- To prevent the output from being too long, your explanations for items that seem highly unlikely/irrelevant can be 1-2 words (e.g. 'not relevant').\n"
        prompt += "- Each ID must appear only once in the ranked list. You should not repeat IDs.\n"
        prompt += "\n"
        prompt += "## Example Output\n"
        prompt += "Here is an example of a valid JSON output (this is just a toy example, and not meant to be scientifically accurate):\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += '    "ranked_prerequisites": [\n'
        prompt += '        {\n'
        prompt += '            "id": "abcd",\n'
        prompt += '            "explanation": "..."\n'
        prompt += '        },\n'
        prompt += '        {\n'
        prompt += '            "id": "wxyz",\n'
        prompt += '            "explanation": "..."\n'
        prompt += '        }\n'
        prompt += '    ]\n'
        prompt += '}\n'
        prompt += "```\n"
        prompt += "\n"
        prompt += "# Important Instructions\n"
        prompt += "- You must be accurate, rigorous, truthful, and faithful to the ask. You always act with the highest scientific integrity, and never make up information.\n"
        prompt += "- Your output should be ASCII formatted, unless Unicode is absolutely necessary (e.g. hyphens should be -, commas should be ', etc.)\n"
        prompt += "- The last thing you output must be valid JSON, between codeblocks (```).\n"
        prompt += "- You must solve this task using the information provided, and your own knowledge. Do NOT call external tools, such as a search engine, paper database, or other tools to answer, as it may contaminate your knowledge.\n"
        prompt += "- Do not hallucinate.\n"

        return prompt

    # Step 1: First LLM call
    prompt = mkPrompt(technology=technology, candidates=candidates, reflection=None)
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=max_generation_time_seconds)

    responses = []
    if (responseJSON is not None):
        responses.append(responseJSON)

    no_first_response = False
    if (responseJSON is None):
        no_first_response = True
        print(f"WARNING: LLM did not return a valid JSON response for the first response.")

    # Step 2: Reflection pass (optional, or forced if first response failed)
    if (use_reflection) or (no_first_response):
        temp = temperature
        if (responseJSON is None):
            temp = max(temperature + 0.2, 1.0)

        reflection_prompt = mkPrompt(technology=technology, candidates=candidates, reflection=responseJSON)
        responseJSON_reflection, responseText_reflection, cost_reflection = getLLMResponseJSON(reflection_prompt, model_str, temp, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=max_generation_time_seconds)

        if (responseJSON_reflection is not None):
            responses.append(responseJSON_reflection)
            cost += cost_reflection

    # Step 3: Pick the latest valid response
    best_response = None
    for response in reversed(responses):
        if (response is not None):
            best_response = response
            break

    return best_response, responses, cost


#
#   Main function: technology requirement prediction
#
def technology_requirement_prediction(technology_description:str, model_str:str, graph:ScientificContributionGraph, technology_name:str=None, top_k_search:int=30, max_tokens:int=16000, temperature:float=0.0, use_reflection:bool=False, max_generation_time_seconds:int=600):
    if (graph is None):
        raise ValueError("A loaded ScientificContributionGraph instance must be provided.")

    # Step 1: Search the graph for candidate prerequisites
    query = "Task: This is a search task, to find scientific contributions that are likely prerequisites for building the following technology.\n"
    query += "Technology: " + technology_description
    print(f"[technology_requirement_prediction] Searching graph (top_k={top_k_search})...")
    search_results = graph.search(query=query, top_n=top_k_search, populate_names=True)
    print(f"[technology_requirement_prediction] Got {len(search_results)} candidates from search.")

    # Step 2: Build compact candidate list for the LLM prompt, and a lookup for enrichment
    candidates = []
    candidate_lookup = {}
    for r in search_results:
        cid = r.get("contribution_id")
        if (cid is None):
            continue
        paper_info = r.get("paper_info") or {}
        candidate_obj = {
            "id": cid,
            "name": r.get("contribution_name") or "",
            "description": r.get("contribution_description") or "",
            "paper_title": paper_info.get("paper_title"),
            "paper_year": paper_info.get("paper_year"),
        }
        candidates.append(candidate_obj)
        candidate_lookup[cid] = {
            "search_result": r,
            "candidate_obj": candidate_obj,
        }

    # Shuffle so the prompt order is randomized (mirrors the precursor task)
    random.shuffle(candidates)

    # Step 3: Package the technology description
    technology = {
        "name": technology_name or "User-described technology",
        "description": technology_description,
    }

    # Step 4: Call the LLM
    print(f"[technology_requirement_prediction] Calling LLM ({model_str})...")
    best_response, all_responses, cost = llm_technology_requirement_prompt(technology=technology, candidates=candidates, model_str=model_str, max_tokens=max_tokens, temperature=temperature, use_reflection=use_reflection, max_generation_time_seconds=max_generation_time_seconds)

    # Step 5: Enrich the ranked items with contribution metadata (where possible)
    ranked_prerequisites_raw = []
    if (best_response is not None):
        ranked_prerequisites_raw = best_response.get("ranked_prerequisites", []) or []

    ranked_prerequisites = []
    for rank_idx, item in enumerate(ranked_prerequisites_raw, start=1):
        cid = item.get("id")
        meta = candidate_lookup.get(cid, {}) if (cid is not None) else {}
        cand_obj = meta.get("candidate_obj", {})
        search_obj = meta.get("search_result", {})
        ranked_prerequisites.append({
            "rank": rank_idx,
            "contribution_id": cid,
            "contribution_name": cand_obj.get("name"),
            "contribution_description": cand_obj.get("description"),
            "paper_title": cand_obj.get("paper_title"),
            "paper_year": cand_obj.get("paper_year"),
            "corpus_id": search_obj.get("corpus_id"),
            "search_cosine": search_obj.get("cosine"),
            "explanation": item.get("explanation", ""),
        })

    # Step 6: Pack the response
    packed = {
        "technology": technology,
        "model_str": model_str,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k_search": top_k_search,
        "num_candidates": len(candidates),
        #"candidates": candidates,
        "ranked_prerequisites": ranked_prerequisites,
        "first_response": all_responses[0] if (len(all_responses) > 0) else None,
        "second_response": all_responses[1] if (len(all_responses) > 1) else None,
        "cost": cost,
    }

    return packed


#
#   Main
#
if __name__ == "__main__":
    loadAPIKeys()
    random.seed(42)

    # Load the graph (must enable search)
    path_to_graph_data = "/data-ssd2/scientific-contribution-graph/download/"
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

    print("\n" + "=" * 80)
    print(f"Technology requirement prediction for: {technology_name}")
    print(f"Description: {technology_description}")


    # Print the ranked list
    print("\n" + "=" * 80)
    print("Ranked predicted prerequisites:")
    print("=" * 80)
    for item in result["ranked_prerequisites"]:
        print(f"\n[Rank {item['rank']}] {item['contribution_name']}")
        print(f"    Contribution ID: {item['contribution_id']}")
        print(f"    Description:     {item['contribution_description']}")
        print(f"    Paper:           {item['paper_title']} ({item['paper_year']})")
        if (item['search_cosine'] is not None) and (isinstance(item['search_cosine'], (int, float))):
            print(f"    Search cosine:   {item['search_cosine']:.3f}")
        print(f"    Explanation:     {item['explanation']}")


    if (isinstance(result["cost"], (int, float))):
        print(f"\nTotal LLM cost for this query: ${result['cost']:.4f}")

    # Save the full output to disk
    filename_out = "technology_requirement_prediction_output." + time.strftime("%Y%m%d-%H%M%S") + ".json"
    with open(filename_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"\nFull output saved to: {filename_out}")