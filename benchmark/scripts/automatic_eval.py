from concurrent.futures import ThreadPoolExecutor
import json
import argparse
from typing import List
from tqdm import tqdm
import numpy as np
import logging
from scipy.stats import ttest_ind
import sys

sys.path.insert(0, "./")

from pipelines.dialog_turn import DialogueTurn
from llm.llm_generate import llm_generate
from llm.global_variables import get_total_cost

logger = logging.getLogger(__name__)


def get_feedback(object_dlg_history: List[DialogueTurn], new_dlg_turn: DialogueTurn):
    feedback = llm_generate(
        template_file="benchmark/prompts/feedback_gpt4.prompt",
        prompt_parameter_values={
            "dlg": object_dlg_history,
            "new_dlg_turn": new_dlg_turn,
        },
        engine="gpt-4",
        max_tokens=256,
        temperature=0,
        stop_tokens=None,
        top_p=0.5,
        postprocess=False,
    )
    # print(new_dlg_turn.agent_utterance)
    # print(feedback)
    feedback_lines = feedback.strip().split("\n")

    if len(feedback_lines) != 5:
        logger.error("Feedback malformatted: %s", feedback)
        return [], []
    scores = (
        []
    )  # Relevant, Informative, Conversational, Non-Redundant, Temporally Correct scores
    for line in feedback_lines:
        score = line.strip().split(" ")[-1].strip()
        if (
            score == "N/A" or "this criterion is not applicable" in line
        ):  # some models say "not applicable" instead of N/A
            score = 5
        else:
            try:
                score = int(score.split("/")[0])
            except:
                if "4/5" in line:
                    score = 4
                elif "3/5" in line:
                    score = 3
                elif "2/5" in line:
                    score = 2
                elif "1/5" in line:
                    score = 1
                elif "5/5" in line:
                    score = 5
                else:
                    logger.error("Feedback line malformatted: %s", line)
                    score = 5

        scores.append(score)

    return feedback_lines, scores


def evaluate_file(input_file, output_file, before_refine):
    # print("before_refine = ", before_refine)
    dlg_history = []
    feedback_out = []

    with open(input_file, "r") as f:
        if before_refine:
            for line in tqdm(f.readlines()):
                data = json.loads(line)
                user_utterance = data["user_utterance"]
                combined_utterance = data["combined_utterance"]
                if not combined_utterance:
                    combined_utterance = data["llm_utterance"]

                cur_dlg_turn = DialogueTurn(
                    user_utterance=user_utterance, agent_utterance=combined_utterance
                )
                feedback_lines, scores = get_feedback(dlg_history, cur_dlg_turn)
                feedback_out.append(
                    {
                        "user_utterance": cur_dlg_turn.user_utterance,
                        "agent_utterance": cur_dlg_turn.agent_utterance,
                        "feedback_lines": feedback_lines,
                        "scores": scores,
                    }
                )
                dlg_history.append(cur_dlg_turn)
                if len(dlg_history) == 5:  # each dialog has 5 turns
                    dlg_history = []
        else:
            for line in tqdm(f.readlines()):
                line_split = line.split("): ")
                if line_split[0].startswith("User"):
                    cur_dlg_turn = DialogueTurn(user_utterance=line_split[1].strip())
                elif line_split[0].startswith("Chatbot"):
                    cur_dlg_turn.agent_utterance = line_split[1].strip()
                    feedback_lines, scores = get_feedback(dlg_history, cur_dlg_turn)
                    feedback_out.append(
                        {
                            "user_utterance": cur_dlg_turn.user_utterance,
                            "agent_utterance": cur_dlg_turn.agent_utterance,
                            "feedback_lines": feedback_lines,
                            "scores": scores,
                        }
                    )
                    dlg_history.append(cur_dlg_turn)
                elif line.startswith("====="):
                    dlg_history = []
                elif line.startswith("Topic:"):
                    pass
                else:
                    logger.error("Unknown line type: %s", line)
    with open(output_file, "w") as f:
        json.dump(feedback_out, f, indent=4)

    print("Total LLM cost: $%.2f" % get_total_cost())


def gather_stats(input_file, idx_to_ignore=[], ignore_im_not_sures=False):
    """
    Compute automatic evaluation scores given GPT-4 feedback output
    ignore_im_not_sures: if True, will skip turns where the agent utterance before refinement is "Sorry, I'm not sure."
    """
    with open(input_file, "r") as f:
        feedback_out = json.load(f)
    idx_to_ignore = set(idx_to_ignore)
    scores = []
    im_not_sure_idx = []
    for idx, feedback in enumerate(feedback_out):
        if ignore_im_not_sures and idx in idx_to_ignore:
            continue
        if len(feedback["scores"]) != 5:
            print("ERROR: Feedback malformatted", feedback["scores"])
            continue
        if (
            ignore_im_not_sures
            and feedback["agent_utterance"] == "Sorry, I'm not sure."
        ):
            im_not_sure_idx.append(idx)
            continue
        scores.append(feedback["scores"])
    scores = np.array(scores)
    scores[:, -1] = (scores[:, -1] == 5) * 100  # convert temporal to accuracy
    # print("temporal = ", scores)
    # print("Average scores:")
    mean_scores = np.mean(scores, axis=0)
    # print with 1 decimal place
    # print(",".join([f"{score:.1f}" for score in mean_scores]))
    # print('Median scores:')
    # print(np.median(scores, axis=0))
    # print("Std scores:")
    std_scores = np.std(scores, axis=0)
    # print(",".join([f"{score:.1f}" for score in std_scores]))
    return mean_scores, std_scores, scores, im_not_sure_idx


def gather_stats_all_experiments(aggregate_file: str, pipeline: str, engine: str):
    # compute the average automatic evaluation metrics for all subsets
    row_template = "{:15} {:18} {:12} {:15} {:15} {:15} {:15} {:15}\n"
    score_headers = [
        "Relevant",
        "Informational",
        "Natural",
        "Non-Repetitive",
        "Temporal",
    ]
    with open(aggregate_file, "a") as f:
        # f.write("\n=====\n")
        # f.write("Input file: " + input_file + "\n")
        f.write(
            row_template.format(
                *[
                    "pipeline",
                    "engine",
                    "subset",
                ]
                + score_headers
            )
        )
        for engine in ["llama", "text-davinci-003", "gpt-4"]:
            generate_scores = None
            early_combine_scores = None
            for pipeline in ["generate", "early_combine"]:
                all_scores = None
                for subset in ["recent", "head", "tail"]:
                    input_file = f"benchmark/evaluation_results/{pipeline}_{subset}_{engine}-feedback.json"
                    # print(input_file)
                    mean_scores, std_scores, scores, _ = gather_stats(input_file)
                    if all_scores is None:
                        all_scores = scores
                    else:
                        all_scores = np.concatenate([all_scores, scores], axis=0)
                    # sum_scores += scores
                    # print("scores = ", scores)
                    # print("sum_scores = ", all_scores.shape)
                    # format mean and std scores with 1 decimal place
                    row = row_template.format(
                        *(
                            [pipeline, engine, subset]
                            + [
                                f"{mean:.1f} ± {std:.1f}"
                                for mean, std in zip(mean_scores, std_scores)
                            ]
                        )
                    )
                    f.write(row)

                # write the 'all' subset
                subset = "all"
                mean_scores, std_scores = np.mean(all_scores, axis=0), np.std(
                    all_scores, axis=0
                )
                row = row_template.format(
                    *(
                        [pipeline, engine, subset]
                        + [
                            f"{mean:.1f} ± {std:.1f}"
                            for mean, std in zip(mean_scores, std_scores)
                        ]
                    )
                )
                f.write(row)
                f.write("\n")
                if pipeline == "generate":
                    generate_scores = all_scores
                elif pipeline == "early_combine":
                    early_combine_scores = all_scores
            assert generate_scores.shape == early_combine_scores.shape
            print(
                "T-test results between generate and early_combine for engine="
                + engine
                + ":"
            )
            for idx in range(generate_scores.shape[1]):
                print(score_headers[idx])
                print(
                    ttest_ind(generate_scores[:, idx], early_combine_scores[:, idx]),
                )
            print("\n=====")


def gather_stats_before_after_all_subsets(input_after_file, ignore_im_not_sures=False):
    # compute the improvement in automatic evaluation metrics before and after refinement
    diffs = []
    for subset in ["head", "tail", "recent"]:
        after_file = (
            input_after_file.replace("recent", subset)
            .replace("head", subset)
            .replace("tail", subset)
        )
        before_file = after_file.replace(".json", "-before.json")
        # print(before_file, after_file)
        mean_before, _, _, im_not_sure_idx = gather_stats(
            before_file, [], ignore_im_not_sures=ignore_im_not_sures
        )
        mean_after, _, _, _ = gather_stats(
            after_file, im_not_sure_idx, ignore_im_not_sures=ignore_im_not_sures
        )
        diff = mean_after - mean_before
        print(f"{subset} improvement:")
        print(",".join([f"{score:.1f}" for score in diff]))
        diffs.append(diff)
    print("average improvement:")
    print(",".join([f"{score:.1f}" for score in np.mean(diffs, axis=0)]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_to_eval", type=str, required=True)
    parser.add_argument("--pipeline", type=str, required=True)
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--feedback_output_file", type=str, required=True)
    parser.add_argument(
        "--score_output_file",
        type=str,
        default="benchmark/evaluation_results/automatic_eval_scores.csv",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["evaluate_file", "compare_before_after", "get_stats"],
    )
    args = parser.parse_args()

    if args.mode == "evaluate_file":
        if args.pipeline == "early_combine":
            with ThreadPoolExecutor(2) as executor:
                after_refine = executor.submit(
                    evaluate_file,
                    args.file_to_eval,
                    args.feedback_output_file,
                    before_refine=False,
                )
                before_refine = executor.submit(
                    evaluate_file,
                    args.file_to_eval.replace(".txt", ".log"),
                    args.feedback_output_file.replace(".json", "-before.json"),
                    before_refine=True,
                )

            before_refine = before_refine.result()
            after_refine = after_refine.result()
        else:
            evaluate_file(
                args.file_to_eval, args.feedback_output_file, before_refine=False
            )
    elif args.mode == "compare_before_after":
        assert args.pipeline == "early_combine"
        print("If we include all turns:")
        gather_stats_before_after_all_subsets(
            args.feedback_output_file, ignore_im_not_sures=False
        )

        print(
            "\nIf we skip turns when the agent says I'm not sure (due to the combined evidence being empty):"
        )
        gather_stats_before_after_all_subsets(
            args.feedback_output_file, ignore_im_not_sures=True
        )
    elif args.mode == "get_stats":
        gather_stats_all_experiments(
            args.score_output_file,
            args.pipeline,
            args.engine,
        )
    else:
        raise ValueError("Unknown mode")
