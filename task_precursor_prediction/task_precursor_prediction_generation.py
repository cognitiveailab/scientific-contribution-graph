# task_prerequisite_prediction.py

import os
import json
import time
import random

from scicontgraph import ScientificContributionGraph, PaperReference


def generate_dataset_for_prerequisite_prediction(graph:ScientificContributionGraph, filename_out:str):
    # hyperparameters
    years = [2021, 2022, 2023, 2024, 2025]
    num_per_year_by_year = {
        2021: 200,
        2022: 200,
        2023: 200,
        2024: 200,
        2025: 1200,
    }

    # Step 1: Get a set of corpus_ids that are in specific years.
    lut_corpus_id_to_year = graph.corpus_id_to_paper_year
    corpus_ids_by_year = {year: set() for year in years}
    for corpus_id, year in lut_corpus_id_to_year.items():
        if year in corpus_ids_by_year:
            corpus_ids_by_year[year].add(corpus_id)

    # Step 2: For each year, randomly sample some corpus_ids.
    sampled_corpus_ids_by_year = {}
    for year in years:
        corpus_ids = list(corpus_ids_by_year[year])
        num_per_year = num_per_year_by_year[year]
        if (len(corpus_ids) > num_per_year):
            sampled_corpus_ids_by_year[year] = random.sample(corpus_ids, num_per_year*3)  # over-sample in the hope that at least `num_per_year` will meet requirements.
        else:
            sampled_corpus_ids_by_year[year] = corpus_ids

    # Step 3: For each sampled corpus_id, randomly pick either it's first or second contribution, and get the contribution_id.
    contributions_by_year = {year: [] for year in years}
    # Progress bar: Count total number of corpus_ids we're sampling from across all years, and keep a running count of how many we've processed so we can print progress.
    total_corpus_ids = sum(len(corpus_ids) for corpus_ids in sampled_corpus_ids_by_year.values())
    processed_corpus_ids = 0

    # Break out below into a function, so we can parallelize it
    def process_corpus_id(corpus_id, year):
        contribution_ids = graph.get_contribution_ids_for_corpus_id(corpus_id)
        if (len(contribution_ids) == 0):
            return None
        contribution_id = random.choice(contribution_ids[:2])  # pick either the first or second contribution

        # Get the contribution
        contribution_obj = graph.get_contribution_by_id(contribution_id)
        if (contribution_obj is None):
            return None

        # Pack the problem.
        problem = {
            "contribution_id": contribution_id,
            "corpus_id": corpus_id,
            "year": year,
            "contribution_name": contribution_obj.name,
            "contribution_description": contribution_obj.description,
            "prerequisites": [],
            "omit_corpus_ids_for_distractors": [],
            "distractor_candidates": []
        }

        #### TODO: Should add 'exclude_contribution_ids' field, which lists any papers (that happen to be listed in any of the other contributions prereqs) that shouldn't be included in the distractors list, for potentially being false negatives.

        # Populate the prerequisites. For each prerequisite, we want to include the name and description.
        # We only want to include them if they are labeled as 'core', and if they come from an external paper, with a populated match (since we'll be getting the prerequisite text from that paper's contribution).

        prerequisites = []
        for prereq in contribution_obj.prerequisites:
            if (prereq.core_or_peripheral == "core"):
                references = prereq.references
                for ref in references:
                    if (isinstance(ref, PaperReference)):
                        for match in ref.matches:
                            # Check if the 'contribution_id' field is populated
                            contribution_id_in_match = match.contribution_id
                            match_type = match.match_type
                            if (contribution_id_in_match is not None) and (match_type == "strong"):
                                # We likely have a valid prerequisite -- it comes from a paper, and is listed as a strong match. We can include it in the problem.

                                # Load the contribution object for this prerequisite contribution_id
                                prereq_contribution_obj = graph.get_contribution_by_id(contribution_id_in_match)
                                if (prereq_contribution_obj is not None):
                                    prerequisites.append({
                                        "contribution_id": contribution_id_in_match,
                                        "name": prereq_contribution_obj.name,
                                        "description": prereq_contribution_obj.description
                                    })

        problem["prerequisites"] = prerequisites

        # Make sure the problem has at least (say) 3 prerequisites, otherwise it might be too easy and not useful for evaluation.
        #if (len(problem["prerequisites"]) >= 3):
            #contributions_by_year[year].append(problem)
        if (len(problem["prerequisites"]) < 3):
            return None

        # The omit list for distractors: A list of corpus_ids that are prerequisites for any contribution in this paper.
        omit_list= set()
        # Add the paper's own corpus_id to the omit list, since we don't want to include other contributions from the same paper as distractors.
        omit_list.add(corpus_id)
        # Add any corpus_id that is a prerequisite for any contribution in this paper to the omit list, since those would be too closely related to the problem contribution to be good distractors.
        for contribution in graph.load_paper(corpus_id).contributions:
            for prereq in contribution.prerequisites:
                for ref in prereq.references:
                    if (isinstance(ref, PaperReference)):
                        omit_list.add(ref.corpus_id)
        problem["omit_corpus_ids_for_distractors"] = list(omit_list)

        # Now, we'll search for potential distractors.
        # Distractors must:
        # - Be from a year that's earlier than the problem contribution's year.
        # - Not be from a corpus_id that's in the omit list.
        search_query_str = f"Instruction: Search for related contributions. Name: {contribution_obj.name}. Description: {contribution_obj.description}"
        search_results = graph.search(search_query_str, top_n=1000, populate_names=True)
        distractor_candidates = []
        for result in search_results:
            # First, check if the corpus_id is on the omit list
            #print("Result:\n" + json.dumps(result, indent=4))
            if (result.get("corpus_id", None) is None):
                #print("\tSkipping result with missing corpus_id.")
                continue
            if (result["corpus_id"] in omit_list):
                #print(f"\tSkipping result with corpus_id {result['corpus_id']} since it's on the omit list.")
                continue
            # Second, check the year
            paper_year = result.get("paper_info", {}).get("paper_year", None)
            if (paper_year is None):
                #print("\tSkipping result with missing paper_year.")
                continue
            if (not isinstance(paper_year, int)):
                #print(f"\tSkipping result with invalid paper_year {paper_year}.")
                continue
            if (paper_year >= year):
                #print(f"\tSkipping result with paper_year {paper_year} since it's not earlier than problem year {year}.")
                continue

            print(f"\tKeeping result with paper_year {paper_year} since it's earlier than problem year {year}.")
            distractor_candidates.append(result)


        # Only keep problems that have a total of at least (say) 100 distractors.
        print("Found {} distractor candidates for contribution_id {}.".format(len(distractor_candidates), contribution_id))
        if (len(distractor_candidates) >= 50):
            problem["distractor_candidates"] = distractor_candidates
            #contributions_by_year[year].append(problem)
            return problem

        return None


    # Serial version
    # for year, corpus_ids in sampled_corpus_ids_by_year.items():
    #     for corpus_id in corpus_ids:
    #         processed_corpus_ids += 1
    #         # Update progress bar
    #         print("Processing corpus_id {}/{} ({}%)".format(processed_corpus_ids, total_corpus_ids, round(processed_corpus_ids/total_corpus_ids*100, 2)), end="\r")

    #         problem = process_corpus_id(corpus_id, year)
    #         if (problem is not None):
    #             contributions_by_year[year].append(problem)

    # PARALLEL VERSION
    #NUM_WORKERS = 1
    NUM_WORKERS = 10
    start_time = time.time()
    total_time = 0
    avg_time_per_corpus_id = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for year, corpus_ids in sampled_corpus_ids_by_year.items():
            for corpus_id in corpus_ids:
                futures.append(executor.submit(process_corpus_id, corpus_id, year))

        for future in as_completed(futures):
            processed_corpus_ids += 1
            # Update progress bar
            print("Processing corpus_id {}/{} ({}%)".format(processed_corpus_ids, total_corpus_ids, round(processed_corpus_ids/total_corpus_ids*100, 2)), end="\r")

            problem = future.result()
            if (problem is not None):
                contributions_by_year[problem["year"]].append(problem)

            total_time = time.time() - start_time
            avg_time_per_corpus_id = total_time / processed_corpus_ids if processed_corpus_ids > 0 else 0
            print(f"Total time elapsed: {total_time:.2f} seconds")
            print(f"\tAverage time per corpus_id so far: {avg_time_per_corpus_id:.2f} seconds")
            estimated_total_time = avg_time_per_corpus_id * total_corpus_ids
            print(f"\tEstimated total time for all corpus_ids: {estimated_total_time:.2f} seconds")
            estimated_remaining_time = estimated_total_time - total_time
            print(f"\tEstimated remaining time: {estimated_remaining_time:.2f} seconds")

            # Partial save: Save every 50 processed corpus_ids, so that if the process is interrupted we don't lose everything. We can overwrite the same file each time since we're only keeping the final results in memory, and not appending to the file.
            if (processed_corpus_ids % 500 == 0) or (processed_corpus_ids < 50):
                print(f"\nProcessed {processed_corpus_ids} corpus_ids, saving intermediate results to {filename_out}...")
                with open(filename_out, "w", encoding="utf-8") as f:
                    json.dump(contributions_by_year, f, indent=4, ensure_ascii=False)

                # Also export the contributions by year
                num_contributions_by_year = {year: len(probs) for year, probs in contributions_by_year.items()}
                repacked = {
                    "processed_corpus_ids": processed_corpus_ids,
                    "total_corpus_ids": total_corpus_ids,
                    "total_time_elapsed_seconds": total_time,
                    "average_time_per_corpus_id_seconds": avg_time_per_corpus_id,
                    "estimated_total_time_seconds": estimated_total_time,
                    "estimated_remaining_time_seconds": estimated_remaining_time,
                    "num_contributions_by_year": num_contributions_by_year,
                }
                with open("num_contributions_by_year_intermediate.json", "w", encoding="utf-8") as f:
                    json.dump(repacked, f, indent=4, ensure_ascii=False)

            # Show a histogram of how many problems we got per year.
            for year, problems in contributions_by_year.items():
                print(f"Year {year}: {len(problems)} problems")


    # Show a histogram of how many problems we got per year.
    for year, problems in contributions_by_year.items():
        print(f"Year {year}: {len(problems)} problems")

    # Export them to a JSON file.
    print(f"Exporting {sum(len(probs) for probs in contributions_by_year.values())} problems to {filename_out}...")
    with open(filename_out, "w", encoding="utf-8") as f:
        json.dump(contributions_by_year, f, indent=4, ensure_ascii=False)

    # Also export the contributions by year
    repacked = {
        "processed_corpus_ids": processed_corpus_ids,
        "total_corpus_ids": total_corpus_ids,
        "total_time_elapsed_seconds": total_time,
        "average_time_per_corpus_id_seconds": avg_time_per_corpus_id,
        "num_contributions_by_year": num_contributions_by_year,
    }
    with open("num_contributions_by_year_intermediate.json", "w", encoding="utf-8") as f:
        json.dump(repacked, f, indent=4, ensure_ascii=False)



            #contributions_by_year[year].append((corpus_id, contribution_id))




def convert_raw_problem_to_final_format(problem_raw:dict, total_problem_length:int=50):
    contribution_id_main = problem_raw["contribution_id"]
    corpus_id_main = problem_raw["corpus_id"]
    year_main = problem_raw["year"]
    contribution_name_main = problem_raw["contribution_name"]
    contribution_description_main = problem_raw["contribution_description"]
    publication_date_main = problem_raw.get("publication_date", None)
    gold_prerequisites = problem_raw["prerequisites"]
    distractor_candidates = problem_raw["distractor_candidates"]

    # We'll need the total gold prerequisites + distractors to be `total_problem_length` (e.g. 50).
    num_distractors_to_include = total_problem_length - len(gold_prerequisites)
    if (num_distractors_to_include < 0):
        return None

    # Get the top distractors, since they're sorted by cosine similarity.
    distractors_to_include = distractor_candidates[:num_distractors_to_include]

    distractors_filtered = []
    # For each distractor, re-pack it to only include the fields we want (corpus_id, contribution_id, name, description).
    for distractor in distractors_to_include:
        #contribution_id = distractor.get("contribution_id", None)
        contribution_name = distractor.get("contribution_name", None)
        contribution_description = distractor.get("contribution_description", None)
        #cosine = distractor.get("cosine", None)
        #year = distractor.get("paper_info", {}).get("paper_year", None)

        distractors_filtered.append({
            "id": "",
            "name": contribution_name,
            "description": contribution_description,
            "gold": False,
            "cosine": distractor.get("cosine", None),
        })

    # Also convert the prerequisites into just the fields we want (name, description).
    prerequisites_filtered = []
    for prereq in gold_prerequisites:
        name = prereq.get("name", None)
        description = prereq.get("description", None)
        prerequisites_filtered.append({
            "id": "",
            "name": name,
            "description": description,
            "gold": True,
            "cosine": None,
        })

    # Generate random IDs for the prerequisties and distractors. Random 4-letter strings (a-z).
    random_ids = set()
    while (len(random_ids) < (len(prerequisites_filtered) + len(distractors_filtered))):
        random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
        random_ids.add(random_id)
    random_ids = list(random_ids)

    # Add the random IDs to the prerequisites and distractors.
    all_data = prerequisites_filtered + distractors_filtered
    gold_ids = set()
    for i in range(len(all_data)):
        all_data[i]["id"] = random_ids[i]
        if (all_data[i]["gold"]):
            gold_ids.add(random_ids[i])

    # If the total problem length is less than `total_problem_length`, then return None
    if (len(all_data) < total_problem_length):
        print ("Problem with contribution_id {} has only {} total choices ({} gold, {} distractors), which is less than the required total_problem_length of {}. Skipping this problem.".format(contribution_id_main, len(all_data), len(prerequisites_filtered), len(distractors_filtered), total_problem_length))
        return None

    # Remove the "gold" field since we don't want to include that in the final problem format.
    cosine_lut = {}
    for item in all_data:
        del item["gold"]

        # Don't include `None` cosines
        cosine = item.get("cosine", None)
        if (cosine is not None):
            cosine_lut[item["id"]] = cosine
        # Delete the cosine field since we don't want to include that in the final problem format.
        del item["cosine"]

    # Shuffle the order of the prerequisites and distractors, so that the gold ones aren't always at the front.
    random.shuffle(all_data)

    # Pack the final problem format.
    problem_final = {
        "problem": {
            "contribution_name": contribution_name_main,
            "contribution_description": contribution_description_main,
            "year": year_main,
            "publication_date": publication_date_main,
            "choices": all_data,
        },
        "oracle_information": {
            "contribution_id": contribution_id_main,
            "corpus_id": corpus_id_main,
            "gold_prerequisite_ids": list(gold_ids),
            "cosine_lut": cosine_lut,
        }
    }

    return problem_final



def convert_raw_data_to_problems(filename_in:str, filename_out:str, total_problem_length:int=100): #, max_per_year:int=250):
    print("* Loading raw data: " + str(filename_in))
    with open(filename_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_per_year_by_year = {
        2021: 250,
        2022: 250,
        2023: 250,
        2024: 250,
        2025: 1000,
    }

    print("* Converting raw problems...")
    problems = []
    for year, problems_raw in data.items():
        for problem_raw in problems_raw:
            problem_final = convert_raw_problem_to_final_format(problem_raw, total_problem_length=total_problem_length)
            if (problem_final is not None):
                problems.append(problem_final)

    # Count number of problems by year
    problems_by_year = {}
    for problem in problems:
        year = problem["problem"]["year"]
        if year not in problems_by_year:
            problems_by_year[year] = []
        problems_by_year[year].append(problem)

    # Print number of problems by year
    for year, problems_in_year in problems_by_year.items():
        print(f"Year {year}: {len(problems_in_year)} problems")

    # Trim the number of problems per year to `max_per_year`, since some years have a lot more problems than others and we want to have a more balanced dataset.
    problems_trimmed = []
    for year, problems_in_year in problems_by_year.items():
        max_per_year = num_per_year_by_year.get(year, 250)
        if (len(problems_in_year) > max_per_year):
            # Prefer those with a non-None `publication_date`, since that might be useful information for the model and for analysis. So first, we'll separate the problems into those with a non-None `publication_date` and those with a None `publication_date`, and then we'll sample from the ones with a non-None `publication_date` first, and if we still have room to reach `max_per_year`, then we'll sample from the ones with a None `publication_date`.
            problems_with_pub_date = [p for p in problems_in_year if p["problem"]["publication_date"] is not None]
            problems_without_pub_date = [p for p in problems_in_year if p["problem"]["publication_date"] is None]
            #problems_trimmed.extend(random.sample(problems_in_year, max_per_year))

            # If this is 2025, find all the problems with dates past August 2025, and include them in the pool.
            if (year == 2025):
                problems_with_pub_date_2025_09 = []
                problems_with_pub_date_2025_not_09 = []

                for p in problems_with_pub_date:
                    pub_date = p["problem"]["publication_date"]
                    if (pub_date is not None):
                        pub_year = pub_date.get("year", None)
                        pub_month = pub_date.get("month", None)
                        if (pub_year is not None) and (pub_month is not None):
                            #if (pub_year > 2025) or (pub_year == 2025 and pub_month > 8):
                            if (pub_year > 2025) or (pub_year == 2025 and pub_month > 7):       # Include July, since some models have this as a cutoff as well
                                problems_with_pub_date_2025_09.append(p)
                            else:
                                problems_with_pub_date_2025_not_09.append(p)
                        else:
                            problems_with_pub_date_2025_not_09.append(p)
                    else:
                        problems_with_pub_date_2025_not_09.append(p)

                # Add these to the pool of problems with publication dates, since we want to make sure to include them in the dataset.
                problems_trimmed.extend(problems_with_pub_date_2025_09)
                # Swap out the original list with the remaining ones, since we've already added the 2025-09 ones to the main list.
                problems_with_pub_date = problems_with_pub_date_2025_not_09
                # Reduce the max_per_year by the number of 2025-09 problems we've added, since those will be taking up some of the slots for 2025.
                max_per_year -= len(problems_with_pub_date_2025_09)
                print("Found {} problems with publication dates past August 2025, which will be included in the dataset. This reduces the max_per_year for 2025 from {} to {}.".format(len(problems_with_pub_date_2025_09), num_per_year_by_year[2025], max_per_year))

            if (len(problems_with_pub_date) >= max_per_year):
                problems_trimmed.extend(random.sample(problems_with_pub_date, max_per_year))
            else:
                problems_trimmed.extend(problems_with_pub_date)
                remaining_slots = max_per_year - len(problems_with_pub_date)
                if (len(problems_without_pub_date) >= remaining_slots):
                    problems_trimmed.extend(random.sample(problems_without_pub_date, remaining_slots))
                else:
                    problems_trimmed.extend(problems_without_pub_date)

            #print("Trimmed year {} from {} problems to {} problems.".format(year, len(problems_in_year), max_per_year))
            print("Trimmed year {} from {} problems to {} problems ({} with publication date, {} without publication date).".format(year, len(problems_in_year), max_per_year, len(problems_with_pub_date), len(problems_without_pub_date)))
        else:
            problems_trimmed.extend(problems_in_year)

    # Make a month-year histogram of publication dates of the problems. (YYYY-MM, where MM is UNK if publication date is None or if month is not specified in the publication date).
    pub_date_histogram = {}
    for problem in problems_trimmed:
        pub_date = problem["problem"]["publication_date"]
            #         "publication_date": {
            #     "year": 2021,
            #     "month": 4,
            #     "day": 30
            # }
        if (pub_date is None) or (pub_date.get("year", None) is None):
            year = problem["problem"]["year"]
            pub_date_str = "{}-UNK".format(year)
        else:
            year = pub_date.get("year", None)
            month = pub_date.get("month", None)
            if (year is not None) and (month is not None):
                pub_date_str = "{}-{:02d}".format(year, month)
            elif (year is not None):
                pub_date_str = "{}-UNK".format(year)
            else:
                pub_date_str = "UNK-UNK"

        if pub_date_str not in pub_date_histogram:
            pub_date_histogram[pub_date_str] = 0
        pub_date_histogram[pub_date_str] += 1

    # Show the histogram (sorted by month-year)
    print("\nPublication date histogram (month-year):")
    for pub_date_str in sorted(pub_date_histogram.keys()):
        count = pub_date_histogram[pub_date_str]
        print(f"{pub_date_str}: {count} problems")

    print("")

    print(f"Converted {len(problems_trimmed)} problems. Exporting to {filename_out}...")
    with open(filename_out, "w", encoding="utf-8") as f:
        json.dump(problems_trimmed, f, indent=4, ensure_ascii=False)



# def add_publication_dates_to_raw_data(filename_in:str, filename_out:str):
#     print("* Loading raw data: " + str(filename_in))
#     with open(filename_in, "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     # Load the scientific contribution graph
#     path_to_graph_data = "/data-ssd2/scientific-contribution-graph/download/"
#     graph = ScientificContributionGraph(path=path_to_graph_data, search_enabled=True, search_device="cpu")
#
#     print("* Adding publication dates...")
#     for year, problems in data.items():
#         for problem in problems:
#             contribution_id = problem["contribution_id"]
#             corpus_id = problem["corpus_id"]
#             paper_info = graph.get_paper_info_by_contribution_id(contribution_id)
#             publication_date = None
#             if (paper_info is not None):
#                 publication_date = paper_info.get("paper_publication_date", None)
#             problem["publication_date"] = publication_date

#     print(f"Exporting updated data with publication dates to {filename_out}...")
#     with open(filename_out, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)


# Main
if __name__ == "__main__":

    # # Step 1: Generation
    path_to_graph_data = "/data-ssd2/scientific-contribution-graph/download/"
    graph = ScientificContributionGraph(path=path_to_graph_data, search_enabled=True, search_device="cpu")
    filename_out = "prerequisite_prediction_dataset.debug.large-may7a.json"
    generate_dataset_for_prerequisite_prediction(graph, filename_out)

    # Step 2: Convert the data from the raw form into a set of problems.

    # # First, add the publication dates to the raw data, since we'll want to include those in the final problem format and they weren't included in the raw data.
    filename_in = "prerequisite_prediction_dataset.debug.large-may7a.json"
    filename_out = "prerequisite_prediction_dataset.debug.large-may7a.with_publication_dates.json"
    #add_publication_dates_to_raw_data(filename_in, filename_out)

    total_length = 100
    #filename_in = "prerequisite_prediction_dataset.debug.json"
    filename_in = filename_out
    filename_out = "prerequisite_prediction_problems.final_format." + time.strftime("%Y%m%d-%H%M%S") + ".json"
    convert_raw_data_to_problems(filename_in, filename_out, total_problem_length=total_length)