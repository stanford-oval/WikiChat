import argparse
import json
from typing import List
import pandas as pd
from tqdm import tqdm
import sys

sys.path.insert(0, "./")

from pipelines.dialog_turn import DialogueTurn
from pipelines.chatbot import Chatbot
from pipelines.utils import make_parent_directories
from pipelines.pipeline_arguments import (
    add_pipeline_arguments,
    check_pipeline_arguments,
)
from llm.load_prompt import _fill_template
from llm.global_variables import get_total_cost

stopwords = [
    "the",
    "a",
    "and",
    "or",
    "then",
    "he",
    "she",
    "it",
    "they",
    "you",
    "to",
    "me",
    "on",
    "was",
    "at",
    "in",
    "was",
    "of",
    "for",
    "is",
    "are",
    "were",
    "not",
    "be",
    "had",
    "I",
    "would",
    "will",
]
stopwords += [s.capitalize() for s in stopwords]


def highlight_keywords_from_claim(
    claim: str, evidence_texts: List[str], evidence_titles: List[str]
):
    claim_keywords = [
        w
        for w in claim.replace(".", " ")
        .replace(",", " ")
        .replace("?", " ")
        .replace('"', " ")
        .replace("'", " ")
        .split(" ")
        if w not in stopwords and len(w) > 0
    ]

    evidence_texts = [
        e.replace("$", "\$").replace("–", "-") for e in evidence_texts
    ]  # escape $ to work with Scale's UI
    for prefix in [" ", "\n", "(", '"']:
        for suffix in [" ", ".", ",", ";", "?", ")", "\n", '"']:
            for i in range(len(evidence_texts)):
                for k in claim_keywords:
                    evidence_texts[i] = evidence_texts[i].replace(
                        prefix + k + suffix,
                        prefix
                        + '<strong style="background-color:beige;"">'
                        + k
                        + "</strong>"
                        + suffix,
                    )
                    evidence_titles[i] = evidence_titles.replace(
                        prefix + k + suffix,
                        prefix
                        + '<strong style="background-color:beige;"">'
                        + k
                        + "</strong>"
                        + suffix,
                    )

    return evidence_texts, evidence_titles


# TODO parallelize this function
def format_simulated_data(args):
    simulation_pipeline = args.pipeline
    args.pipeline = "generate_and_correct"  # needed to use claim_splitter from chatbot
    chatbot = Chatbot(args)
    dlg_history = []
    dlg_claims = set()
    make_parent_directories(args.output_file)
    content_list = []
    metadata_list = []
    example_id = 0
    turn_num = 0
    dlg_topic = ""
    with open(args.input_file, "r") as f:
        for line in tqdm(f.readlines(), desc="Lines"):
            line_split = line.split("): ")
            if line_split[0].startswith("User"):
                cur_dlg_turn = DialogueTurn(user_utterance=line_split[1].strip())
            elif line_split[0].startswith("Chatbot"):
                turn_num += 1
                cur_dlg_turn.agent_utterance = line_split[1].strip()
                claims = chatbot.claim_splitter.split_claim(
                    dialog_history=dlg_history,
                    new_user_utterance=cur_dlg_turn.user_utterance,
                    current_agent_utterance=cur_dlg_turn.agent_utterance,
                    system_parameters={"engine": "gpt-4"},
                    dialog_topic=dlg_topic,
                )
                # print(claims)
                dlg_history.append(cur_dlg_turn)

                ret_output = chatbot._retrieve_evidences(claims, top_p=0.7)

                for claim_idx, evidences in ret_output.items():
                    claim_idx = int(claim_idx)
                    if claims[claim_idx][0] in dlg_claims:
                        # print("Skipping duplicate claim")
                        continue

                    claim = claims[claim_idx][0]
                    evidence_texts = [e[2] for e in evidences]
                    evidence_titles = [e[0] for e in evidences]
                    evidence_texts, evidence_titles = highlight_keywords_from_claim(
                        claim, evidence_texts, evidence_titles
                    )

                    turn_params = {
                        "user_utterance": cur_dlg_turn.user_utterance,
                        "dialog_history": dlg_history,
                        "claim": claim,
                        "evidence_titles": evidence_titles,
                        "evidence_texts": evidence_texts,
                    }
                    content, _ = _fill_template(args.scale_template_file, turn_params)
                    # print(content)
                    # exit()
                    content_list.append(content)
                    metadata_list.append(
                        json.dumps(
                            {
                                "pipeline": simulation_pipeline,
                                "subset": args.subset,
                                "engine": args.engine,  # "atlas"
                                "id": str(example_id),
                                "turn_num": str(turn_num),
                                "agent_utterance": cur_dlg_turn.agent_utterance,
                            }
                        )
                    )
                    example_id += 1

                for c in claims:
                    dlg_claims.add(c[0])

            elif line.startswith("====="):
                turn_num = 0
                dlg_history = []
                dlg_claims = set()
            elif line.startswith("Topic:"):
                dlg_topic = line[7:].strip()
                # print("dialog topic = ", dlg_topic)
            else:
                raise ValueError("ERROR: Unknown line type %s" % line)

    df = pd.DataFrame({"text": content_list, "metadata": metadata_list})
    df.to_csv(args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_pipeline_arguments(parser)

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Where to read the partial conversations from, with the last line of each conversation being the model response.",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Where to write the outputs."
    )
    parser.add_argument(
        "--scale_template_file",
        type=str,
        default="benchmark/prompts/scale_factuality.prompt",
        help="prompt file to generate input data file for scale ai human evaluation.",
    )

    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        help="The subset of the benchamrk.",
    )

    args = parser.parse_args()
    check_pipeline_arguments(args)

    format_simulated_data(args)
    print("Total LLM cost: $%.2f" % get_total_cost())
