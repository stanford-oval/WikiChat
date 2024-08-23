"""
Evaluates a .json file that contains the outputs of the distilled model in the following format:
[
    {
        "template_name": "...",
        "instruction": "...",
        "input": "...",
        "output": "...",
        "prediction: "..."
    },
    ...
]
"""

import argparse
import json
import re

import evaluate
from tqdm import tqdm

rouge = evaluate.load("rouge")


def RougeL(pred, label):
    result = rouge.compute(predictions=[pred], references=[label], use_stemmer=True)
    # print(pred)
    # print(label)
    # print(result["rougeL"])
    # print("=====")
    return round(result["rougeL"] * 100, 4)


def metric_for_initial_search(prediction, gold):
    search_pattern = 'Yes\. You.*"([^"]*)".* The year of the results is "([^=]*)"\.]?'

    def extract_parts(o):
        search_prompt_output = o.strip()
        search_match = re.match(search_pattern, search_prompt_output)
        if search_match:
            classification = "Yes"
            search_query = search_match.group(1)
            search_query_time = search_match.group(2)
        else:
            if o.startswith("No"):
                classification = "No"
                search_query = ""
                search_query_time = ""
            else:
                print("Malformed search_prompt_output: ", search_prompt_output)
                classification = ""
                search_query = ""
                search_query_time = ""
        return classification, search_query, search_query_time

    p_c, p_q, p_t = extract_parts(prediction)
    g_c, g_q, g_t = extract_parts(gold)

    classification = (p_c == g_c) * 100
    query = RougeL(p_q, g_q)
    time = (p_t == g_t) * 100

    return (classification, query, time)


def just_rougeL(prediction, gold):
    return (RougeL(prediction, gold),)


def metrics_for_verification(prediction, gold):
    def extract_label(o):
        verification_label = "REFUTES"  # default set to be "REFUTES"
        if (
            'is "supports"' in o.lower()
            or "no fact-checking is needed for this claim" in o.lower()
            or "the fact-checking result is not applicable to this response"
            in o.lower()
        ):
            verification_label = "SUPPORTS"
        elif 'the fact-checking result is "not enough info"' in o.lower():
            verification_label = "NOT ENOUGH INFO"
        return verification_label

    p = extract_label(prediction)
    g = extract_label(gold)
    return ((p == g) * 100,)


def metrics_for_refine(prediction, gold):
    def extract_scores(feedback):
        if "User:" in feedback:
            feedback = feedback.split("User:")[0]
        feedback_lines = feedback.strip().split("\n")

        if len(feedback_lines) < 4 or len(feedback_lines) > 5:
            print("Feedback malformatted: ", feedback_lines)
            return []

        scores = (
            []
        )  # Relevant, Informative, Conversational, Non-Redundant, Temporally Correct scores
        for line in feedback_lines:
            score = line.strip().split(" ")[-1].strip()
            if (
                score == "N/A" or "this criterion is not applicable" in line
            ):  # some models say "not applicable" instead of N/A
                score = 100
            else:
                try:
                    score = int(score.split("/")[0])
                except:
                    print(f"Feedback line malformatted: {line}")
                    score = 100
            scores.append(score)
        return scores

    def extract_refined_reponse(o):
        refine_identifiers = [
            "Revised response after applying this feedback:",
            "Response after applying this feedback:",
        ]
        for identifier in refine_identifiers:
            if identifier in o:
                feedback, refined_utterance = o.split(identifier)
                scores = extract_scores(feedback)
                refined_utterance = refined_utterance.strip()
                return scores, refined_utterance
        print("Refined response malformatted: %s" % refined_utterance)

    p_scores, p_refine = extract_refined_reponse(prediction)
    g_scores, g_refined = extract_refined_reponse(gold)

    score_accuracy = (
        sum([p_scores[i] == g_scores[i] for i in range(len(p_scores))])
        / len(p_scores)
        * 100
    )
    refined_rougL = RougeL(p_refine, g_refined)

    return (score_accuracy, refined_rougL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        help="The json file that contains the distilled model's predictions.",
    )

    args = parser.parse_args()

    with open(args.input_file) as input_file:
        all_examples = json.load(input_file)

    prompt_matrics = {
        "generate.prompt": {
            "function": just_rougeL,
            "metric_names": ["baseline_rougeL"],
        },
        "query.prompt": {
            "function": metric_for_initial_search,
            "metric_names": [
                "initial_search_classification_accuracy",
                "initial_search_rougeL",
                "initial_search_time_accuracy",
            ],
        },
        "summarize_and_filter.prompt": {
            "function": just_rougeL,
            "metric_names": ["summarize_and_filter_rougeL"],
        },
        "split_claims.prompt": {
            "function": just_rougeL,
            "metric_names": ["split_claims_rougeL"],
        },
        "verify.prompt": {
            "function": metrics_for_verification,
            "metric_names": ["verification_accuracy"],
        },
        "draft.prompt": {
            "function": just_rougeL,
            "metric_names": ["combiner_rougeL"],
        },
        "refine_w_feedback.prompt": {
            "function": metrics_for_refine,
            "metric_names": ["feedback_score_accuracy", "refined_response_rougeL"],
        },
        "refine.prompt": {
            "function": just_rougeL,
            "metric_names": ["refined_response_rougeL"],
        },
    }

    for prompt in prompt_matrics.keys():
        prompt_matrics[prompt]["values"] = [0] * len(
            prompt_matrics[prompt]["metric_names"]
        )
        prompt_matrics[prompt]["count"] = 0

    for example in tqdm(all_examples, desc="Calculating metrics"):
        found = False
        for prompt in prompt_matrics.keys():
            if example["template_name"].endswith(prompt):
                metrics = prompt_matrics[prompt]["function"](
                    example["prediction"], example["output"]
                )
                # each metric function outputs an iterable
                for i in range(len(prompt_matrics[prompt]["values"])):
                    prompt_matrics[prompt]["values"][i] += metrics[i]
                prompt_matrics[prompt]["count"] += 1
                found = True
                break
        if not found:
            print(
                "Did not find the metric function for template ",
                example["template_name"],
            )

    for prompt in prompt_matrics.keys():
        if prompt_matrics[prompt]["count"] == 0:
            print("Did not encounter any examples with template_name=" + prompt)
        else:
            for i, metric_name in enumerate(prompt_matrics[prompt]["metric_names"]):
                print(
                    f"%-40s=%.2f"
                    % (
                        # prompt,
                        metric_name,
                        prompt_matrics[prompt]["values"][i]
                        / prompt_matrics[prompt]["count"],
                    )
                )
