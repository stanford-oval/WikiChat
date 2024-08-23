import logging
import os
import pathlib
import re

import spacy

logger = logging.getLogger(__name__)

spacy_nlp = None  # will load when needed

wikipedia_language_dict = {
    "aa": "Afar",
    "ab": "Abkhazian",
    "ace": "Achinese",
    "ady": "Adyghe",
    "af": "Afrikaans",
    "ak": "Akan",
    "alt": "Southern Altai",
    "am": "Amharic",
    "ami": "Amis",
    "an": "Aragonese",
    "ang": "Old English",
    "anp": "Angika",
    "ar": "Arabic",
    "arc": "Aramaic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "as": "Assamese",
    "ast": "Asturian",
    "atj": "Atikamekw",
    "av": "Avaric",
    "avk": "Kotava",
    "awa": "Awadhi",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "azb": "South Azerbaijani",
    "ba": "Bashkir",
    "ban": "Balinese",
    "bar": "Bavarian",
    "bbc": "Batak Toba",
    "bcl": "Central Bikol",
    "be": "Belarusian",
    "be-tarask": "Belarusian (Taraškievica orthography)",
    "bg": "Bulgarian",
    "bh": "Bhojpuri",
    "bi": "Bislama",
    "bjn": "Banjar",
    "blk": "Pa'O",
    "bm": "Bambara",
    "bn": "Bangla",
    "bo": "Tibetan",
    "bpy": "Bishnupriya",
    "br": "Breton",
    "bs": "Bosnian",
    "bug": "Buginese",
    "bxr": "Russia Buriat",
    "ca": "Catalan",
    "cbk-zam": "Chavacano",
    "cdo": "Mindong",
    "ce": "Chechen",
    "ceb": "Cebuano",
    "ch": "Chamorro",
    "cho": "Choctaw",
    "chr": "Cherokee",
    "chy": "Cheyenne",
    "ckb": "Central Kurdish",
    "co": "Corsican",
    "cr": "Cree",
    "crh": "Crimean Tatar",
    "cs": "Czech",
    "csb": "Kashubian",
    "cu": "Church Slavic",
    "cv": "Chuvash",
    "cy": "Welsh",
    "da": "Danish",
    "dag": "Dagbani",
    "de": "German",
    "dga": "Dagaare",
    "din": "Dinka",
    "diq": "Zazaki",
    "dsb": "Lower Sorbian",
    "dty": "Doteli",
    "dv": "Divehi",
    "dz": "Dzongkha",
    "ee": "Ewe",
    "el": "Greek",
    "eml": "Emiliano-Romagnolo",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "ext": "Extremaduran",
    "fa": "Persian",
    "fat": "Fanti",
    "ff": "Fula",
    "fi": "Finnish",
    "fj": "Fijian",
    "fo": "Faroese",
    "fon": "Fon",
    "fr": "French",
    "frp": "Arpitan",
    "frr": "Northern Frisian",
    "fur": "Friulian",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gag": "Gagauz",
    "gan": "Gan",
    "gcr": "Guianan Creole",
    "gd": "Scottish Gaelic",
    "gl": "Galician",
    "glk": "Gilaki",
    "gn": "Guarani",
    "gom": "Goan Konkani",
    "gor": "Gorontalo",
    "got": "Gothic",
    "gpe": "Ghanaian Pidgin",
    "gsw": "Alemannic",
    "gu": "Gujarati",
    "guc": "Wayuu",
    "gur": "Frafra",
    "guw": "Gun",
    "gv": "Manx",
    "ha": "Hausa",
    "hak": "Hakka Chinese",
    "haw": "Hawaiian",
    "he": "Hebrew",
    "hi": "Hindi",
    "hif": "Fiji Hindi",
    "ho": "Hiri Motu",
    "hr": "Croatian",
    "hsb": "Upper Sorbian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "hyw": "Western Armenian",
    "hz": "Herero",
    "ia": "Interlingua",
    "id": "Indonesian",
    "ie": "Interlingue",
    "ig": "Igbo",
    "ii": "Sichuan Yi",
    "ik": "Inupiaq",
    "ilo": "Iloko",
    "inh": "Ingush",
    "io": "Ido",
    "is": "Icelandic",
    "it": "Italian",
    "iu": "Inuktitut",
    "ja": "Japanese",
    "jam": "Jamaican Creole English",
    "jbo": "Lojban",
    "jv": "Javanese",
    "ka": "Georgian",
    "kaa": "Kara-Kalpak",
    "kab": "Kabyle",
    "kbd": "Kabardian",
    "kbp": "Kabiye",
    "kcg": "Tyap",
    "kg": "Kongo",
    "ki": "Kikuyu",
    "kj": "Kuanyama",
    "kk": "Kazakh",
    "kl": "Kalaallisut",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "koi": "Komi-Permyak",
    "kr": "Kanuri",
    "krc": "Karachay-Balkar",
    "ks": "Kashmiri",
    "ksh": "Colognian",
    "ku": "Kurdish",
    "kv": "Komi",
    "kw": "Cornish",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lad": "Ladino",
    "lb": "Luxembourgish",
    "lbe": "Lak",
    "lez": "Lezghian",
    "lfn": "Lingua Franca Nova",
    "lg": "Ganda",
    "li": "Limburgish",
    "lij": "Ligurian",
    "lld": "Ladin",
    "lmo": "Lombard",
    "ln": "Lingala",
    "lo": "Lao",
    "lrc": "Northern Luri",
    "lt": "Lithuanian",
    "ltg": "Latgalian",
    "lv": "Latvian",
    "lzh": "Literary Chinese",
    "mad": "Madurese",
    "mai": "Maithili",
    "map-bms": "Basa Banyumasan",
    "mdf": "Moksha",
    "mg": "Malagasy",
    "mh": "Marshallese",
    "mhr": "Eastern Mari",
    "mi": "Māori",
    "min": "Minangkabau",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mni": "Manipuri",
    "mnw": "Mon",
    "mr": "Marathi",
    "mrj": "Western Mari",
    "ms": "Malay",
    "mt": "Maltese",
    "mus": "Muscogee",
    "mwl": "Mirandese",
    "my": "Burmese",
    "myv": "Erzya",
    "mzn": "Mazanderani",
    "na": "Nauru",
    "nah": "Nāhuatl",
    "nan": "Minnan",
    "nap": "Neapolitan",
    "nds": "Low German",
    "nds-nl": "Low Saxon",
    "ne": "Nepali",
    "new": "Newari",
    "ng": "Ndonga",
    "nia": "Nias",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "nov": "Novial",
    "nqo": "N’Ko",
    "nrm": "Norman",
    "nso": "Northern Sotho",
    "nv": "Navajo",
    "ny": "Nyanja",
    "oc": "Occitan",
    "olo": "Livvi-Karelian",
    "om": "Oromo",
    "or": "Odia",
    "os": "Ossetic",
    "pa": "Punjabi",
    "pag": "Pangasinan",
    "pam": "Pampanga",
    "pap": "Papiamento",
    "pcd": "Picard",
    "pcm": "Nigerian Pidgin",
    "pdc": "Pennsylvania German",
    "pfl": "Palatine German",
    "pi": "Pali",
    "pih": "Norfuk / Pitkern",
    "pl": "Polish",
    "pms": "Piedmontese",
    "pnb": "Western Punjabi",
    "pnt": "Pontic",
    "ps": "Pashto",
    "pt": "Portuguese",
    "pwn": "Paiwan",
    "qu": "Quechua",
    "rm": "Romansh",
    "rmy": "Vlax Romani",
    "rn": "Rundi",
    "ro": "Romanian",
    "roa-tara": "Tarantino",
    "ru": "Russian",
    "rue": "Rusyn",
    "rup": "Aromanian",
    "rw": "Kinyarwanda",
    "sa": "Sanskrit",
    "sah": "Yakut",
    "sat": "Santali",
    "sc": "Sardinian",
    "scn": "Sicilian",
    "sco": "Scots",
    "sd": "Sindhi",
    "se": "Northern Sami",
    "sg": "Sango",
    "sgs": "Samogitian",
    "sh": "Serbo-Croatian",
    "shi": "Tachelhit",
    "shn": "Shan",
    "si": "Sinhala",
    "simple": "Simple English",
    "sk": "Slovak",
    "skr": "Saraiki",
    "sl": "Slovenian",
    "sm": "Samoan",
    "smn": "Inari Sami",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "srn": "Sranan Tongo",
    "ss": "Swati",
    "st": "Southern Sotho",
    "stq": "Saterland Frisian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "szl": "Silesian",
    "szy": "Sakizaya",
    "ta": "Tamil",
    "tay": "Tayal",
    "tcy": "Tulu",
    "te": "Telugu",
    "tet": "Tetum",
    "tg": "Tajik",
    "th": "Thai",
    "ti": "Tigrinya",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tly": "Talysh",
    "tn": "Tswana",
    "to": "Tongan",
    "tpi": "Tok Pisin",
    "tr": "Turkish",
    "trv": "Taroko",
    "ts": "Tsonga",
    "tt": "Tatar",
    "tum": "Tumbuka",
    "tw": "Twi",
    "ty": "Tahitian",
    "tyv": "Tuvinian",
    "udm": "Udmurt",
    "ug": "Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "ve": "Venda",
    "vec": "Venetian",
    "vep": "Veps",
    "vi": "Vietnamese",
    "vls": "West Flemish",
    "vo": "Volapük",
    "vro": "Võro",
    "wa": "Walloon",
    "war": "Waray",
    "wo": "Wolof",
    "wuu": "Wu",
    "xal": "Kalmyk",
    "xh": "Xhosa",
    "xmf": "Mingrelian",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Cantonese",
    "za": "Zhuang",
    "zea": "Zeelandic",
    "zgh": "Standard Moroccan Tamazight",
    "zh": "Chinese",
    "zu": "Zulu",
}


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)-5s : %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


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


def print_chatbot(utterance: str):
    print(bcolors.OKGREEN + bcolors.BOLD + utterance + bcolors.ENDC, flush=True)


def input_user() -> str:
    try:
        user_utterance = input(bcolors.OKCYAN + bcolors.BOLD + "User: ")
        # ignore empty inputs
        while not user_utterance.strip():
            user_utterance = input(bcolors.OKCYAN + bcolors.BOLD + "User: ")
    finally:
        print(bcolors.ENDC, end="\n", flush=True)
    return user_utterance.strip()


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


def extract_year(title, content):
    global spacy_nlp
    if spacy_nlp is None:
        spacy_nlp = spacy.load("en_core_web_sm")
    if title:
        content = title + " | " + content
    years = []
    year_pattern = r"\d{4}"
    year_duration_pattern = r"\b\d{4}[--–]\d{2}\b"
    year_to_pattern = r"\b\d{4} to \d{4}\b"
    # extract "1990 to 1998" before spacy because spacy would split it to 1990 and 1998
    re_year_tos = re.findall(year_to_pattern, content)
    for re_year_to in re_year_tos:
        re_years = re.findall(year_pattern, re_year_to)
        if len(re_years) != 2:
            continue
        year1, year2 = re_years
        years.extend(list(range(int(year1), int(year2) + 1)))
        content.replace(re_year_to, " ")

    doc = spacy_nlp(content)
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


def dict_to_command_line(
    default_parameters: dict, overwritten_parameters: dict
) -> list[str]:
    """
    This function merges the default options set in a dictionary with options
    that need to be overwritten. It then creates a command line argument
    list of key-value options for those parameters. Boolean True values
    are represented only by the key name, False and None values are omitted.

    Parameters:
    - default_parameters (dict): A dictionary of key-value pairs representing
      the default options.
    - overwritten_parameters (dict): A dictionary of key-value pairs
      that need to overwrite the default options.

    Returns:
    - List[str]: A list of strings where each string is a command line
      argument in the form '--key=value'.
    """

    command_line = []
    parameters = default_parameters.copy()
    for k, v in overwritten_parameters.items():
        parameters[k] = v
    for k, v in parameters.items():
        if v is None:
            continue
        if not isinstance(v, bool):
            command_line.append(f"--{k}={v}")
        else:
            if v == True:
                command_line.append(f"--{k}")
    return command_line
