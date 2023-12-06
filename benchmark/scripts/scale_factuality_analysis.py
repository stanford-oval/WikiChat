# for each system, what is the number of accepted/rejected/NEI claims and the average factual accuracy?
# for each system, and for each turn number, what is the number of accepted/rejected/NEI claims and the average factual accuracy?

from collections import defaultdict
import json
import pandas as pd
import os
from tqdm import tqdm
import html
import argparse


def get_dialog_turn_map(simulation_file):
    dialog_turn_map = {}  # map from chatbot utterance to dialog id turn number
    turn_id = 1
    dialog_id = 1
    user_utterance = None
    with open(simulation_file, "r") as f:
        for line in tqdm(f.readlines()):
            line_split = line.split("): ")
            if line_split[0].startswith("User"):
                user_utterance = line_split[1].strip()
            if line_split[0].startswith("Chatbot"):
                agent_utterance = line_split[1].strip()
                if f"{turn_id}_{agent_utterance}_{user_utterance}" in dialog_turn_map:
                    print(
                        f"agent utterance {agent_utterance} in dialog_turn_map already"
                    )
                dialog_turn_map[f"{turn_id}_{agent_utterance}_{user_utterance}"] = (
                    str(dialog_id) + "_" + str(turn_id)
                )
                turn_id += 1
            elif line.startswith("====="):
                turn_id = 1
                dialog_id += 1
    return dialog_turn_map


def task_stats(tasks_file):
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    stats = {}
    for selected_pipeline in ["early_combine", "generate", "atlas"]:
        stats[selected_pipeline] = {}
        for selected_subset in ["head", "tail", "recent"]:
            pipeline_subset_task_count = 0
            for task in tasks:
                metadata = task["metadata"]
                pipeline = metadata["pipeline"]
                subset = metadata["subset"]
                if pipeline != selected_pipeline or subset != selected_subset:
                    continue
                pipeline_subset_task_count += 1
            stats[selected_pipeline][selected_subset] = pipeline_subset_task_count
    for pipeline in ["early_combine", "generate", "atlas"]:
        stats[pipeline]["total"] = sum(stats[pipeline].values())
    print(stats)
    return stats


def analyze_tasks(tasks_file, output_file):
    with open(tasks_file, "r") as f:
        tasks = json.load(f)
    df_rows = []
    with open(output_file, "w") as f:
        for selected_engine in ["llama", "text-davinci-003", "gpt-4", "atlas"]:
            for selected_pipeline in ["generate", "atlas", "early_combine"]:
                for selected_subset in ["head", "tail", "recent", "recentuserstudy"]:
                    num_correct_per_turn = defaultdict(
                        int
                    )  # map from dialog_id and turn number to number of correct claims
                    num_incorrect_per_turn = defaultdict(
                        int
                    )  # map from dialog_id and turn number to number of incorrect claims
                    num_nei_per_turn = defaultdict(
                        int
                    )  # map from dialog_id and turn number to number of NEI claims
                    unique_turns = set()

                    simulation_file = f"benchmark/simulated_dialogs/{selected_pipeline}_{selected_subset}_{selected_engine}.txt"
                    if not os.path.isfile(simulation_file):
                        continue

                    dialog_turn_map = get_dialog_turn_map(simulation_file)
                    pipeline_subset_task_count = 0
                    for task in tasks:
                        metadata = task["metadata"]
                        pipeline = metadata["pipeline"]
                        engine = metadata["engine"]
                        subset = metadata["subset"]
                        turn_num = metadata["turn_num"]
                        if (
                            pipeline != selected_pipeline
                            or subset != selected_subset
                            or engine != selected_engine
                        ):
                            continue
                        chatbot_utterance = metadata["agent_utterance"]
                        user_utterance = task["params"]["attachments"][0][
                            "content"
                        ].split("\n")[0][
                            len('<strong style="color:green">User</strong>:') + 1 :
                        ]
                        user_utterance = html.unescape(user_utterance)

                        key = f"{turn_num}_{chatbot_utterance}_{user_utterance}"
                        if key not in dialog_turn_map:
                            print(
                                "WARNING: chatbot utterance not found in dialog_turn_map: {}".format(
                                    key
                                )
                            )
                            continue
                        id = dialog_turn_map[key]
                        unique_turns.add(id)

                        if "response" not in task:
                            print(
                                "WARNING: task response not found in task: {}".format(
                                    chatbot_utterance
                                )
                            )
                            continue

                        pipeline_subset_task_count += 1
                        if (
                            "Fact-check the claim"
                            not in task["response"]["annotations"]
                            or len(
                                task["response"]["annotations"]["Fact-check the claim"]
                            )
                            == 0
                        ):
                            print("WARNING: Empty annotation")
                            continue
                        task_response = task["response"]["annotations"][
                            "Fact-check the claim"
                        ][0]
                        if (
                            task_response
                            == "This claim is CORRECT according to these passages."
                        ):
                            num_correct_per_turn[id] += 1
                        elif (
                            task_response
                            == "This claim is NOT CORRECT according to these passages."
                        ):
                            num_incorrect_per_turn[id] += 1
                        elif (
                            task_response
                            == "There is NOT ENOUGH INFORMATION in these passages to verify this claim."
                        ):
                            num_nei_per_turn[id] += 1
                        else:
                            raise ValueError(
                                "Unknown task response: {}".format(task_response)
                            )

                    # output total number of correct/incorrect/NEI claims to file
                    total_num_claims = pipeline_subset_task_count
                    if total_num_claims == 0:
                        print("Skipping empty pipeline-subset-task")
                        continue
                    # factual accuracy: for each system and subset, how many claims got CORRECT, INCORRECT and NEI?
                    factual_accuracy = (
                        sum(num_correct_per_turn.values()) / total_num_claims
                    )
                    f.write(
                        f"Factual accuracy for {selected_engine} {selected_pipeline} {selected_subset}: {factual_accuracy}\n"
                    )
                    f.write("Total number of claims: {}\n".format(total_num_claims))
                    f.write(
                        "Total number of correct claims: {}. Percentage: {}\n".format(
                            sum(num_correct_per_turn.values()),
                            sum(num_correct_per_turn.values()) / total_num_claims,
                        )
                    )
                    f.write(
                        "Total number of incorrect claims: {}. Percentage: {}\n".format(
                            sum(num_incorrect_per_turn.values()),
                            sum(num_incorrect_per_turn.values()) / total_num_claims,
                        )
                    )
                    f.write(
                        "Total number of NEI claims: {}. Percentage: {}\n".format(
                            sum(num_nei_per_turn.values()),
                            sum(num_nei_per_turn.values()) / total_num_claims,
                        )
                    )

                    # joint factual accuracy: for each system and subset, how many turns got all claims marked as CORRECT?
                    num_turns_all_correct = 0
                    num_turns_with_claims = 0
                    for id in unique_turns:
                        total_num_claim_per_turn = (
                            num_correct_per_turn[id]
                            + num_incorrect_per_turn[id]
                            + num_nei_per_turn[id]
                        )
                        # exclude turns with no claims
                        if total_num_claim_per_turn == 0:
                            continue
                        num_turns_with_claims += total_num_claim_per_turn != 0
                        if num_correct_per_turn[id] == total_num_claim_per_turn:
                            num_turns_all_correct += 1
                    joint_factual_accuracy = (
                        num_turns_all_correct / num_turns_with_claims
                    )
                    f.write(
                        f"Joint factual accuracy for {selected_engine} {selected_pipeline} {selected_subset}: {joint_factual_accuracy}\n"
                    )
                    f.write(
                        "Total number of turns with all correct claims: {}. Total number of turns with claims: {} Percentage: {}\n".format(
                            num_turns_all_correct,
                            num_turns_with_claims,
                            joint_factual_accuracy,
                        )
                    )
                    f.write("dialog id _ turn number,correct,incorrect,NEI\n")
                    for id in unique_turns:
                        f.write(
                            "{},{},{},{}\n".format(
                                id,
                                num_correct_per_turn[id],
                                num_incorrect_per_turn[id],
                                num_nei_per_turn[id],
                            )
                        )

                    f.write("\n\n")
                    df_rows.append(
                        {
                            "engine": selected_engine,
                            "piepline": selected_pipeline,
                            "subset": selected_subset,
                            "factual accuracy": round(factual_accuracy * 100, 2),
                            "joint factual accuracy": round(
                                joint_factual_accuracy * 100, 2
                            ),
                        }
                    )
    # print(df_rows)
    df = pd.DataFrame(df_rows)
    df.to_csv(output_file.replace(".txt", ".csv"))
    # # make a dataframe with "correct", "incorrect", "NEI" columns and "turn number" rows
    # df = pd.DataFrame({"correct": num_correct_per_turn, "incorrect": num_incorrect_per_turn, "NEI": num_nei_per_turn})
    # df.to_csv("scale_factuality_analysis.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    analyze_tasks(args.input_file, args.output_file)
