import os
import pathlib
import re

import spacy
import logging

logger = logging.getLogger(__name__)

spacy_nlp = spacy.load("en_core_web_sm")


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_chatbot(s: str):
    print(bcolors.OKGREEN + bcolors.BOLD + s + bcolors.ENDC)


def input_user() -> str:
    try:
        user_utterance = input(bcolors.OKCYAN + bcolors.BOLD + "User: ")
        # ignore empty inputs
        while not user_utterance.strip():
            user_utterance = input(bcolors.OKCYAN + bcolors.BOLD + "User: ")
    finally:
        print(bcolors.ENDC)
    return user_utterance


def make_parent_directories(file_name: str):
    """
    Creates the parent directories of `file_name` if they don't exist
    """
    pathlib.Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)


def is_everything_verified(ver_out):
    """
    Everything is verified when 1) we have only one claim and it is supported or 2) all claims are supported.
    """
    for label_fix in ver_out:
        if label_fix["label"] != "SUPPORTS":
            return False
    return True


def extract_year(title, passage):
    if title:
        passage = title + " | " + passage
    years = []
    year_pattern = r"\d{4}"
    year_duration_pattern = r"\b\d{4}[--–]\d{2}\b"
    year_to_pattern = r"\b\d{4} to \d{4}\b"
    # extract "1990 to 1998" before spacy because spacy would split it to 1990 and 1998
    re_year_tos = re.findall(year_to_pattern, passage)
    for re_year_to in re_year_tos:
        re_years = re.findall(year_pattern, re_year_to)
        if len(re_years) != 2:
            continue
        year1, year2 = re_years
        years.extend(list(range(int(year1), int(year2) + 1)))
        passage.replace(re_year_to, " ")

    doc = spacy_nlp(passage)
    dates = [(X.text, X.label_) for X in doc.ents if X.label_ == "DATE"]
    for date in dates:
        date = date[0]
        # "the 2006–07 season"
        re_year_durations = re.findall(year_duration_pattern, date)
        if re_year_durations:
            for re_year_duration in re_year_durations:
                if "–" in re_year_duration:
                    year1, year2 = re_year_duration.split("–")
                elif "-" in re_year_duration:
                    year1, year2 = re_year_duration.split("-")
                else:
                    continue
                year2 = year1[:2] + year2
                years.extend([year1, year2])
            continue
        # any 4 digits
        re_years = re.findall(year_pattern, date)
        if re_years:
            years.extend(re_years)
    years = list(sorted(set([int(year) for year in years])))
    return years
