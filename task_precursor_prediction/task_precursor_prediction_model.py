# task_precursor_prediction.py
# This script runs the precursor prediction task, for various base LLM models, on the pre-defined dataset.

import os
import json
import random
from tqdm import tqdm

from ExtractionUtils import *

import concurrent.futures

#
#   Prompt
#

def llm_precursor_ranking_prompt(problem_:dict, model_str:str, max_tokens:int=8000, temperature:float=0.0, use_reflection:bool=True, max_generation_time_seconds:int=300):
    def mkPrompt(contribution:dict, prerequisites:list, reflection:None):
        prompt = "You are ScientistGPT, an expert AI scientist. You can answer any scientific problem correctly, faithfully, and accurately, using the highest scientific integrity.\n"
        prompt += "\n"
        prompt += "You must solve this task using the information provided, and your own knowledge. Do NOT call external tools, such as a search engine, paper database, or other tools to answer, as it may contaminate your knowledge.\n"
        prompt += "\n"
        prompt += "# Task\n"
        prompt += "This is a prerequiste matching task.\n"
        prompt += "You will be provided with a scientific contribution, that may be hypothetical or real.\n"
        prompt += "You will also be provided with a list of other scientific contributions, that are under the list of 'possible prerequisites', which are presented in randomied order.\n"
        prompt += "Your task is to determine which of the `possible prerequisites` are most likely to be actual prerequisties to make the main scientific contribution.\n"
        prompt += "\n"
        prompt += "## Additional Instructions\n"
        prompt += "- This will be framed as a ranking task.  You will generate a ranked list of the possible prerequisites (by their ids), in order of most useful to least useful for enabling the main scientific contribution.\n"
        prompt += "- Your task is to determine *direct* precursors.  For example, as a cartoon example: a complicated quantum mechanics contribution might technically have `multiplication` as a precursor, but it likely has a great many more direct precursors closer than this -- you should list the direct precursors first.\n"
        prompt += "\n"

        prompt += "# Main Scientific Contribution\n"
        prompt += "Here is the main scientific contribution (which may be hypothetical or real):\n"
        prompt += "```\n"
        prompt += json.dumps(contribution, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Possible Prerequisites\n"
        prompt += "Here is the list of possible prerequisites (in random order):\n"
        prompt += "```\n"
        prompt += json.dumps(prerequisites, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        if (reflection is not None):
            prompt += "# Reflection\n"
            prompt += "This is a reflection step. Previously, you generated the output below.  Your task is to reflect on that output, and correct any errors, omissions, inaccuracies, or any other issues.\n"
            prompt += "```\n"
            prompt += json.dumps(reflection, indent=4) + "\n"
            prompt += "```\n"
            prompt += "\n"

        prompt += "# Output Instructions\n"
        prompt += "You are welcome to think as much as you would like before answering. Your answer must be in JSON format, between codeblocks (```). Your JSON must be valid JSON.\n"
        prompt + "Your JSON should be a dictionary with the following keys:\n"
        prompt += "- `ranked_prerequisites`: A list of dictionaries representing the possible prerequisites, ranked from most likely to be an actual prerequisite to least likely. The first element is the most likely. Each dictionary will have the following keys:\n"
        prompt += "  - `id`: The id of the possible prerequisite (corresponding to the id in the list of possible prerequisites above)\n"
        prompt += "  - `explanation`: A 1-sentence, concise, information dense explanation of why this possible prerequisite is ranked at this position.\n"
        prompt += "\n"
        prompt += ""
        prompt += "## Output Guidelines\n"
        prompt += "- Your output list must include all the prerequisites in the list above (which is " + str(len(prerequisites)) + " possible prerequisites), but in ranked order.  You should not omit any of the possible prerequisites, but you should rank them in the correct order.\n"
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
        prompt += '            "explanation": "..." \n'
        prompt += '        },\n'
        prompt += '        # Add more ranked prerequisites as needed, for all ' + str(len(prerequisites)) + ' possible prerequisites\n'
        prompt += '    ]\n'
        prompt += '}\n'
        prompt += "```\n"
        prompt += "\n"
        prompt += "# Important Instructions\n"
        prompt += "- You must be accurate, rigorous, truthful, and faithful to the ask. You always act with the highest scientific integrity, and never make up information.\n"
        prompt += "- Please pay particular attention to the guidelines above, to ensure the output is useful for technological roadmapping at a fine granularity.\n"
        prompt += "- Your output should be ASCII formatted, unless Unicode is absolutely necessary (e.g. hyphens should be -, commas should be ', etc.)\n"
        prompt += "- The last thing you output must be valid JSON, between codeblocks (```).\n"
        prompt += "- You must solve this task using the information provided, and your own knowledge. Do NOT call external tools, such as a search engine, paper database, or other tools to answer, as it may contaminate your knowledge.\n"
        prompt += "- Do not hallucinate.\n"

        return prompt

    total_tokens_prompt = 0
    total_tokens_response = 0

    # Unpack the problem
    problem = problem_.get("problem", {})
    contribution_name = problem.get("contribution_name", None)
    contribution_description = problem.get("contribution_description", None)
    year = problem.get("year", None)
    publication_date = problem.get("publication_date", None)
    choices = problem.get("choices", [])

    # Shuffle the choices (they're already shuffled, but we'll shuffle them again)
    random.shuffle(choices)

    # Get the indices of the gold ones
    oracle_information = problem_.get("oracle_information", {})
    gold_prerequisite_ids = set(oracle_information.get("gold_prerequisite_ids", []))
    corpus_id = oracle_information.get("corpus_id", None)
    contribution_id = oracle_information.get("contribution_id", None)

    # Step 2: Make the prompt
    contribution = {
        "name": contribution_name,
        "description": contribution_description
    }
    prompt = mkPrompt(contribution=contribution, prerequisites=choices, reflection=None)

    # Step 3: Get the LLM response
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=max_generation_time_seconds)

    responses = []
    if (responseJSON is not None):
        responses.append(responseJSON)

    no_first_response = False
    if (responseJSON is None):
        no_first_response = True
        print(f"WARNING: LLM did not return a valid JSON response for the first response.")

    # Now, do reflection
    if (use_reflection) or (no_first_response):
        temp = temperature
        if (responseJSON == None):
            temp = max(temperature+0.2, 1.0)

        reflection_prompt = mkPrompt(contribution=contribution, prerequisites=choices, reflection=responseJSON)
        responseJSON_reflection, responseText_reflection, cost_reflection = getLLMResponseJSON(reflection_prompt, model_str, temp, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=max_generation_time_seconds)

        if (responseJSON_reflection is not None):
            responses.append(responseJSON_reflection)

        # For simplicity, we'll just use the reflection response as the final response if it's valid JSON. In practice, you might want to do something more sophisticated here.
        if (responseJSON_reflection is not None):
            responseJSON = responseJSON_reflection
            cost += cost_reflection

    best_response = None
    # Get the last valid JSON response (either the original or the reflection)
    for response in reversed(responses):
        if (response is not None):
            best_response = response
            break

    # Validate the best response -- check that it has (a) all the ids, (b) no duplicate ids, and (c) add any missing IDs to the end of the ranked list in the same random order they appear in the original `choices` list.
    def validate_responses(responses:list):
        validated_ranked_prerequisites = []
        validation_errors = []
        if (responses is not None):

           # Step 1A: Check for any IDs that are not in the original `choices` list, and remove them.
            valid_ids = set([item.get("id", None) for item in choices if item.get("id", None) is not None])
            invalid_removed = []
            for item in responses:
                item_id = item.get("id", None)
                if (item_id is not None):
                    if (item_id not in valid_ids):
                        validation_errors.append(f"WARNING: [INVALID_ID] ID {item_id} in ranked_prerequisites is not in the original choices list. It has been removed from the ranked list by the data validation step.")
                        continue
                    invalid_removed.append(item)

            # Step 1: Check for any duplicate IDs.
            found_ids = set()
            deduplicated_ranked_prerequisites = []
            for item in invalid_removed:
                item_id = item.get("id", None)
                if (item_id is not None):
                    if (item_id in found_ids):
                        validation_errors.append(f"WARNING: [DUPLICATE] Duplicate ID {item_id} found in ranked_prerequisites. Only the first occurrence is kept.")
                        continue
                    found_ids.add(item_id)
                    deduplicated_ranked_prerequisites.append(item)

            # Step 2: Check for any missing IDs, and add them to the end in the same random order they appear in the original `choices` list.
            for item in choices:
                item_id = item.get("id", None)
                if (item_id is not None):
                    if (item_id not in found_ids):
                        deduplicated_ranked_prerequisites.append({
                            "id": item_id,
                            "explanation": "WARNING: Not ranked by model, and added automatically in random order to the end of the list by the data validation step.",
                        })
                        found_ids.add(item_id)
                        validation_errors.append(f"WARNING: [UNRANKED] ID {item_id} is missing from ranked_prerequisites. It has been added to the end of the list by the data validation step.")

            # Step 3: Replace the ranked_prerequisites with the deduplicated and completed list.
            validated_ranked_prerequisites = deduplicated_ranked_prerequisites
        return validated_ranked_prerequisites, validation_errors

    # Calculate the average precision (before and after reflection)
    from sklearn.metrics import average_precision_score
    def sklearn_ap(ranked_prerequisites, gold_indices):
        # Check the input is valid
        if (ranked_prerequisites is None) or (gold_indices is None):
            return 0.0
        if (isinstance(ranked_prerequisites, list) and len(ranked_prerequisites) == 0) or (isinstance(gold_indices, set) and len(gold_indices) == 0):
            return 0.0

        ranked_indices = [x["id"] for x in ranked_prerequisites]
        gold_set = set(gold_indices)

        y_true = [1 if idx in gold_set else 0 for idx in ranked_indices]
        y_score = list(range(len(ranked_indices), 0, -1))

        return average_precision_score(y_true, y_score)

    ap_first_response = None
    ap_second_response = None
    validated_first_response = None
    validated_second_response = None
    validation_errors_1 = []
    validation_errors_2 = []
    try:
        first_response = responses[0] if len(responses) > 0 else None
        second_response = responses[1] if len(responses) > 1 else None
        #validated_first_response, validation_errors_1 = validate_responses(first_response.get("ranked_prerequisites", [])) if first_response is not None else None
        #validated_second_response, validation_errors_2 = validate_responses(second_response.get("ranked_prerequisites", [])) if second_response is not None else None
        validated_first_response, validation_errors_1 = validate_responses(first_response.get("ranked_prerequisites", [])) if first_response is not None else (None, [])
        validated_second_response, validation_errors_2 = validate_responses(second_response.get("ranked_prerequisites", [])) if second_response is not None else (None, [])
        ap_first_response = sklearn_ap(validated_first_response, gold_prerequisite_ids)
        ap_second_response = sklearn_ap(validated_second_response, gold_prerequisite_ids)
        print(f"Average Precision of first response: {ap_first_response}")
        print(f"Average Precision of second response (reflection): {ap_second_response}")
    except Exception as e:
        import traceback
        print(f"Error calculating average precision: {e}")
        traceback.print_exc()

    # Pack the response
    packed = {
        "contribution_id": contribution_id,
        "corpus_id": corpus_id,
        "year": year,
        "publication_date": publication_date,
        "problem": problem_,
        "model_str": model_str,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "average_precision_first_response": ap_first_response,
        "average_precision_second_response": ap_second_response,
        "first_response": first_response,
        "second_response": second_response,
        "validated_first_response": validated_first_response,
        "validated_second_response": validated_second_response,
        "validation_errors_first_response": validation_errors_1,
        "validation_errors_second_response": validation_errors_2,
        "cost": cost,
        "gold_prerequisite_ids": list(gold_prerequisite_ids),
    }

    return packed


#
#   Random Baseline
#
def random_baseline(problem_:dict):
    # Unpack the problem
    problem = problem_.get("problem", {})
    contribution_name = problem.get("contribution_name", None)
    contribution_description = problem.get("contribution_description", None)
    year = problem.get("year", None)
    publication_date = problem.get("publication_date", None)
    choices = problem.get("choices", [])

    # Shuffle the choices (they're already shuffled, but we'll shuffle them again)
    random.shuffle(choices)

    # Get the indices of the gold ones
    oracle_information = problem_.get("oracle_information", {})
    gold_prerequisite_ids = set(oracle_information.get("gold_prerequisite_ids", []))
    corpus_id = oracle_information.get("corpus_id", None)
    contribution_id = oracle_information.get("contribution_id", None)

    # Validate the best response -- check that it has (a) all the ids, (b) no duplicate ids, and (c) add any missing IDs to the end of the ranked list in the same random order they appear in the original `choices` list.
    def validate_responses(responses:list):
        validated_ranked_prerequisites = []
        validation_errors = []
        if (responses is not None):

           # Step 1A: Check for any IDs that are not in the original `choices` list, and remove them.
            valid_ids = set([item.get("id", None) for item in choices if item.get("id", None) is not None])
            invalid_removed = []
            for item in responses:
                item_id = item.get("id", None)
                if (item_id is not None):
                    if (item_id not in valid_ids):
                        validation_errors.append(f"WARNING: [INVALID_ID] ID {item_id} in ranked_prerequisites is not in the original choices list. It has been removed from the ranked list by the data validation step.")
                        continue
                    invalid_removed.append(item)

            # Step 1: Check for any duplicate IDs.
            found_ids = set()
            deduplicated_ranked_prerequisites = []
            for item in invalid_removed:
                item_id = item.get("id", None)
                if (item_id is not None):
                    if (item_id in found_ids):
                        validation_errors.append(f"WARNING: [DUPLICATE] Duplicate ID {item_id} found in ranked_prerequisites. Only the first occurrence is kept.")
                        continue
                    found_ids.add(item_id)
                    deduplicated_ranked_prerequisites.append(item)

            # Step 2: Check for any missing IDs, and add them to the end in the same random order they appear in the original `choices` list.
            for item in choices:
                item_id = item.get("id", None)
                if (item_id is not None):
                    if (item_id not in found_ids):
                        deduplicated_ranked_prerequisites.append({
                            "id": item_id,
                            "explanation": "WARNING: Not ranked by model, and added automatically in random order to the end of the list by the data validation step.",
                        })
                        found_ids.add(item_id)
                        validation_errors.append(f"WARNING: [UNRANKED] ID {item_id} is missing from ranked_prerequisites. It has been added to the end of the list by the data validation step.")

            # Step 3: Replace the ranked_prerequisites with the deduplicated and completed list.
            validated_ranked_prerequisites = deduplicated_ranked_prerequisites
        return validated_ranked_prerequisites, validation_errors

    # Calculate the average precision (before and after reflection)
    from sklearn.metrics import average_precision_score
    def sklearn_ap(ranked_prerequisites, gold_indices):
        # Check the input is valid
        if (ranked_prerequisites is None) or (gold_indices is None):
            return 0.0
        if (isinstance(ranked_prerequisites, list) and len(ranked_prerequisites) == 0) or (isinstance(gold_indices, set) and len(gold_indices) == 0):
            return 0.0

        ranked_indices = [x["id"] for x in ranked_prerequisites]
        gold_set = set(gold_indices)

        y_true = [1 if idx in gold_set else 0 for idx in ranked_indices]
        y_score = list(range(len(ranked_indices), 0, -1))

        return average_precision_score(y_true, y_score)

    ap_first_response = None
    ap_second_response = None
    validated_first_response = None
    validated_second_response = None
    validation_errors_1 = []
    validation_errors_2 = []
    try:
        first_response = [] # Empty list should auto-populate with all the choices in random order, with blank explanations, after validation step
        second_response = []
        validated_first_response, validation_errors_1 = validate_responses([])  # Empty list should auto-populate with all the choices in random order, with blank explanations, after validation step
        validated_second_response, validation_errors_2 = validate_responses([])  # Empty list should auto-populate with all the choices in random order, with blank explanations, after validation step
        ap_first_response = sklearn_ap(validated_first_response, gold_prerequisite_ids)
        ap_second_response = sklearn_ap(validated_second_response, gold_prerequisite_ids)
        print(f"Average Precision of first response: {ap_first_response}")
        print(f"Average Precision of second response (reflection): {ap_second_response}")
    except Exception as e:
        import traceback
        print(f"Error calculating average precision: {e}")
        traceback.print_exc()

    # Pack the response
    packed = {
        "contribution_id": contribution_id,
        "corpus_id": corpus_id,
        "year": year,
        "publication_date": publication_date,
        "problem": problem_,
        "model_str": "random_baseline",
        "average_precision_first_response": ap_first_response,
        "average_precision_second_response": ap_second_response,
        "first_response": first_response,
        "second_response": second_response,
        "validated_first_response": validated_first_response,
        "validated_second_response": validated_second_response,
        "validation_errors_first_response": validation_errors_1,
        "validation_errors_second_response": validation_errors_2,
        "cost": 0.0,
        "gold_prerequisite_ids": list(gold_prerequisite_ids),
    }

    return packed



#
#   Main
#
if __name__ == "__main__":
    loadAPIKeys()

    # Set random seed
    random.seed(42)
    # Set of 2000, with ~half of samples from 2025 (when many common knowledge cutoff dates are)
    filename_benchmark_in = "task_precursor_prediction/prerequisite_prediction_problems.final_format.20260508-113303.json"
    # Check that the file exists.
    if (not os.path.isfile(filename_benchmark_in)):
        print(f"Error: file {filename_benchmark_in} does not exist.")
        print("You may need to unzip it from the corresponding .zip file, which is too large to upload directly to GitHub.")
        exit(1)

    # Load the precursor prediction problems from the JSON file
    print("Loading precursor prediction problems from: " + filename_benchmark_in)
    with open(filename_benchmark_in, "r", encoding="utf-8") as f:
        problems = json.loads(f.read())
    print("Loaded " + str(len(problems)) + " precursor prediction problems.")

    # Run a handful
    #DEBUG_LIMIT = 5
    DEBUG_LIMIT = 25
    #DEBUG_LIMIT = 100
    #DEBUG_LIMIT = 200
    #DEBUG_LIMIT = None


    #model_str = "gpt-5-mini"       # Submitted
    model_str = "gpt-5-nano"       # Submitted
    #model_str = "gpt-5.4-mini"     # Submitted
    #model_str = "gpt-5.4"           # Submitted
    #model_str = "gpt-4o-mini"       # Submitted
    #model_str = "openrouter/openai/gpt-oss-120b"    # Submitted
    #model_str = "openrouter/meta-llama/llama-3.1-8b-instruct"
    #model_str = "anthropic/claude-haiku-4-5"        # Submitted
    #model_str = "anthropic/claude-sonnet-4-5"       # Submitted
    #model_str = "anthropic/claude-sonnet-4-6"       # Omit (after knowledge cutoff)
    #model_str = "anthropic/claude-opus-4-6"          # Submitted

    #model_str = "random"           # Submitted

    knowledge_cutoff_year = None
    knowledge_cutoff_month = None

    knowledge_cutoffs = {
        "gpt-4o-mini": (2023, 10),
        "gpt-4.1": (2024, 6),
        "gpt-5-mini": (2024, 5),
        "gpt-5-nano": (2024, 5),
        "gpt-5.4-mini": (2025, 8),
        "gpt-5.4": (2025, 8),
        "openrouter/openai/gpt-oss-120b": (2024, 6),
        "openrouter/meta-llama/llama-3.1-8b-instruct": (2023, 12),
        "anthropic/claude-haiku-4-5": (2025, 7),
        "anthropic/claude-sonnet-4-5": (2025, 7),
        "anthropic/claude-sonnet-4-6": (2025, 1),
        "anthropic/claude-opus-4-6": (2025, 8),
    }

    if (model_str in knowledge_cutoffs):
        knowledge_cutoff_year, knowledge_cutoff_month = knowledge_cutoffs[model_str]
        print(f"Using knowledge cutoff of {knowledge_cutoff_year}-{knowledge_cutoff_month:02d} for model {model_str}")
    elif (model_str == "random"):
        print(f"Random baseline does not require a knowledge cutoff, so proceeding without filtering problems by knowledge cutoff.")
        knowledge_cutoff_year = 2050
        knowledge_cutoff_month = 1
    else:
        print(f"Warning: no knowledge cutoff specified for model {model_str}.  Proceeding without filtering problems by knowledge cutoff.")
        exit(1)


    model_str_sanitized = model_str.replace("/", "_").replace(" ", "_").replace("@", "_").replace(".", "_")



    path_out = "precursor_prediction_outputs/" + model_str_sanitized + "/"
    os.makedirs(path_out, exist_ok=True)

    #problems_to_process = problems[:DEBUG_LIMIT] if DEBUG_LIMIT is not None else problems
    problems_to_process = problems
    if (DEBUG_LIMIT is not None):
        # Don't shuffle or take the first N -- subsample every Nth, since they'll be roughly sorted in temporal order.
        take_every_nth = max(1, len(problems) // DEBUG_LIMIT)
        problems_to_process = problems[::take_every_nth]
        print(f"DEBUG_LIMIT is set to {DEBUG_LIMIT}, so subsampling: taking every {take_every_nth}th problem for a total of {len(problems_to_process)} problems to process.")


    # Serial version
    # for i, problem in enumerate(problems_to_process):
    #     print(f"Running precursor prediction for problem {i+1}/{len(problems)}")
    #     output = llm_precursor_ranking_prompt(problem, model_str=model_str, max_tokens=8000, temperature=0.0, use_reflection=True, max_generation_time_seconds=300)
    #     output_filename = os.path.join(path_out, f"precursor_prediction_output_{i+1}.json")
    #     with open(output_filename, "w", encoding="utf-8") as f:
    #         json.dump(output, f, indent=4)
    #     print(f"Output saved to: {output_filename}")

    #exit(1)

    # Parallel version
    NUM_WORKERS = 10
    #NUM_WORKERS = 25
    #NUM_WORKERS = 40
    #NUM_WORKERS = 100
    #NUM_WORKERS = 200
    all_results = []
    total_cost = 0.0
    temperature = 0.0
    use_reflection = False  # Doesn't really seem to help on this task

    filename_out_summary = "precursor_prediction_summary." + model_str_sanitized + ".run" + time.strftime("%Y%m%d-%H%M%S") + ".json"
    filename_out_summary = os.path.join(path_out, filename_out_summary)

    results_before_knowledge_cutoff = []
    results_after_knowledge_cutoff = []
    binning_histogram = {
        "before_knowledge_cutoff": 0,
        "after_knowledge_cutoff": 0,
        "at_knowledge_cutoff": 0,
        "unknown": 0,
        "None": 0,
    }

    #max_generation_time = 300
    max_generation_time = 60*15  # 10 minutes
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for i, problem in enumerate(problems_to_process):
            print(f"Submitting precursor prediction for problem {i+1}/{len(problems)}")
            future = None
            if (model_str == "random"):
                future = executor.submit(random_baseline, problem)
            else:
                future = executor.submit(llm_precursor_ranking_prompt, problem, model_str=model_str, max_tokens=16000, temperature=temperature, use_reflection=use_reflection, max_generation_time_seconds=max_generation_time)
            futures.append((future, i))

        for future, i in futures:
            try:
                output = future.result()
                all_results.append(output)

                output_filename = os.path.join(path_out, f"precursor_prediction_output_{i+1}.json")
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=4)
                print(f"Output saved to: {output_filename}")

                total_cost += output.get("cost", 0.0)

                # Calculate mean average precision across all problems processed so far
                all_aps_first_response = [result.get("average_precision_first_response", 0.0) for result in all_results if result.get("average_precision_first_response") is not None]
                all_aps_second_response = [result.get("average_precision_second_response", 0.0) for result in all_results if result.get("average_precision_second_response") is not None]
                # Keep track of number of `Nones`
                num_nones_first_response = sum([1 for result in all_results if result.get("average_precision_first_response") is None])
                num_nones_second_response = sum([1 for result in all_results if result.get("average_precision_second_response") is None])

                mean_ap_first_response = sum(all_aps_first_response) / len(all_aps_first_response) if len(all_aps_first_response) > 0 else None
                mean_ap_second_response = sum(all_aps_second_response) / len(all_aps_second_response) if len(all_aps_second_response) > 0 else None

                print("Current mean average precision:")
                print(f"\tAfter first response: {mean_ap_first_response} (based on {len(all_aps_first_response)} samples, with {num_nones_first_response} Nones)")
                print(f"\tAfter second response (reflection): {mean_ap_second_response} (based on {len(all_aps_second_response)} samples, with {num_nones_second_response} Nones)")
                cost_per_problem = total_cost / len(all_results) if len(all_results) > 0 else 0.0
                print(f"Total cost so far: ${total_cost:.4f}  (average cost per problem: ${cost_per_problem:.4f})")


                # NOTE: We're also going to bin the results by those 'before knowledge cutoff' vs 'after knowledge cutoff'. The `problem` has a `publication_date` field, with `year` and `month` keys.
                publication_date = output.get("publication_date", None)
                bin_type = None
                pub_year = None
                pub_month = None
                coarse_year = output.get("year", None)
                if (knowledge_cutoff_year is not None) and (knowledge_cutoff_month is not None):
                    # Primary method: A publication date in the problem metadata.  If it's present, we'll use it to determine whether this problem is before or after the knowledge cutoff.
                    if (publication_date is not None) and isinstance(publication_date, dict):
                        pub_year = publication_date.get("year", None)
                        pub_month = publication_date.get("month", None)
                        if (pub_year is not None) and (pub_month is not None):
                            # If it's after the knowledge cutoff, put it in the after.  If it's 1 month or more before, put it in the before. If it's on the same month, then don't include it in either bin.
                            if (pub_year < knowledge_cutoff_year) or (pub_year == knowledge_cutoff_year and pub_month < knowledge_cutoff_month):
                                bin_type = "before_knowledge_cutoff"
                            elif (pub_year > knowledge_cutoff_year) or (pub_year == knowledge_cutoff_year and pub_month > knowledge_cutoff_month):
                                bin_type = "after_knowledge_cutoff"
                            elif (pub_year == knowledge_cutoff_year and pub_month == knowledge_cutoff_month):
                                bin_type = "at_knowledge_cutoff"
                            else:
                                bin_type = "unknown"
                        else:
                            bin_type = "unknown"

                    # Secondary method: If there's no publication date, we can also check the year field as a fallback, though this is less precise since it doesn't have the month.
                    if (bin_type == "unknown") and (coarse_year is not None):
                        if (coarse_year < knowledge_cutoff_year):
                            bin_type = "before_knowledge_cutoff"
                        elif (coarse_year > knowledge_cutoff_year):
                            bin_type = "after_knowledge_cutoff"
                        elif (coarse_year == knowledge_cutoff_year):
                            bin_type = "at_knowledge_cutoff"
                        else:
                            bin_type = "unknown"

                else:
                    bin_type = "unknown"

                # DEBUG: Put the date and bin type in the output for debugging purposes
                #print(f"Knowledge Cutoff: {knowledge_cutoff_year}-{knowledge_cutoff_month:02d}, Publication Date: {pub_year}-{pub_month:02d}, Bin Type: {bin_type}")
                # Handle Nones
                print("Knowledge cutoff: " + str(knowledge_cutoff_year) + "-" + str(knowledge_cutoff_month) + "   Publication Date: " + str(pub_year) + "-" + str(pub_month) + "   Coarse Year: " + str(coarse_year) + "   Bin Type: " + str(bin_type))
                if (bin_type is None):
                    bin_type = "None"
                binning_histogram[bin_type] = binning_histogram.get(bin_type, 0) + 1
                print("Current binning histogram: " + json.dumps(binning_histogram, indent=4))


                # Now, put it in the right bins.
                if (bin_type == "before_knowledge_cutoff"):
                    results_before_knowledge_cutoff.append(output)
                elif (bin_type == "after_knowledge_cutoff"):
                    results_after_knowledge_cutoff.append(output)
                else:
                    pass

                # Calculate the mean average precision for the before vs after knowledge cutoff bins as well. (as well as # of samples and Nones in each bin)
                def calculate_bin_stats(results):
                    aps_first_response = [result.get("average_precision_first_response", 0.0) for result in results if result.get("average_precision_first_response") is not None]
                    aps_second_response = [result.get("average_precision_second_response", 0.0) for result in results if result.get("average_precision_second_response") is not None]
                    num_nones_first_response = sum([1 for result in results if result.get("average_precision_first_response") is None])
                    num_nones_second_response = sum([1 for result in results if result.get("average_precision_second_response") is None])
                    mean_ap_first_response = sum(aps_first_response) / len(aps_first_response) if len(aps_first_response) > 0 else None
                    mean_ap_second_response = sum(aps_second_response) / len(aps_second_response) if len(aps_second_response) > 0 else None
                    return mean_ap_first_response, mean_ap_second_response, num_nones_first_response, num_nones_second_response, aps_first_response, aps_second_response

                mean_ap_first_response_before, mean_ap_second_response_before, num_nones_first_response_before, num_nones_second_response_before, all_aps_first_response_before, all_aps_second_response_before = calculate_bin_stats(results_before_knowledge_cutoff)
                mean_ap_first_response_after, mean_ap_second_response_after, num_nones_first_response_after, num_nones_second_response_after, all_aps_first_response_after, all_aps_second_response_after = calculate_bin_stats(results_after_knowledge_cutoff)


                # Pack summary
                summary = {
                    "model_str": model_str,

                    "num_problems_processed": len(all_results),
                    "mean_ap_first_response": mean_ap_first_response,
                    "mean_ap_second_response": mean_ap_second_response,
                    "total_cost": total_cost,
                    "average_cost_per_problem": cost_per_problem,
                    "num_nones_first_response": num_nones_first_response,
                    "num_nones_second_response": num_nones_second_response,
                    # Raw APs
                    "all_aps_first_response": all_aps_first_response,
                    "all_aps_second_response": all_aps_second_response,

                    "binning_histogram": binning_histogram,

                    # Now by knowledge cutoff
                    "knowledge_cutoff_analysis": {
                        "knowledge_cutoff_year": knowledge_cutoff_year,
                        "knowledge_cutoff_month": knowledge_cutoff_month,
                        "before_knowledge_cutoff": {
                            "num_problems": len(results_before_knowledge_cutoff),
                            "mean_ap_first_response": mean_ap_first_response_before,
                            "mean_ap_second_response": mean_ap_second_response_before,
                            "num_nones_first_response": num_nones_first_response_before,
                            "num_nones_second_response": num_nones_second_response_before,
                            "all_aps_first_response": all_aps_first_response_before,
                            "all_aps_second_response": all_aps_second_response_before,
                        },
                        "after_knowledge_cutoff": {
                            "num_problems": len(results_after_knowledge_cutoff),
                            "mean_ap_first_response": mean_ap_first_response_after,
                            "mean_ap_second_response": mean_ap_second_response_after,
                            "num_nones_first_response": num_nones_first_response_after,
                            "num_nones_second_response": num_nones_second_response_after,
                            "all_aps_first_response": all_aps_first_response_after,
                            "all_aps_second_response": all_aps_second_response_after,
                        },
                    }
                }

                # put in path_out
                with open(filename_out_summary, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4)
                print(f"Summary saved to: {filename_out_summary}")


            except Exception as e:
                import traceback
                print(f"Error processing problem {i+1}: {e}")
                traceback.print_exc()