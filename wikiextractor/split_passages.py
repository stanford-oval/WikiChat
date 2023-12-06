import asyncio
import json
import math
from multiprocessing import Pipe, Process, cpu_count
import os
import pathlib
from typing import List
import aiohttp
import nltk
import re
import html
import argparse
from urllib.parse import quote, unquote
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
from time import time
import stanza


nltk.download("punkt")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s : %(message)s")

# these filters should be applied in order
link_tag = re.compile(
    r'[<«]a href="(.*?)"?[>»]([\S\s]*?)[<«]/a[>»]'
)  # also matches empty tags, missing closing quotation in href, « instead of <, newline inside the tag
block_quote = re.compile(
    r'<templatestyles src="(Template:(Blockquote)|الگو:گفتاورد)/styles\.css"\s*/>(.*)\n'
)
css_tag_filters = [
    re.compile(
        r'(Notes|References|Explanatory notes|Citations|Primary sources|Secondary sources|Tertiary sources|یادداشت‌ها|منابع|پانویس|واژه‌نامه|توضیحات|برای مطالعه بیشتر|Notas|Referencias|Bibliografía|Notas y referencias|Citas|Enlaces externos)\.\n<templatestyles src="(Reflist|Refbegin|پانویس|آغاز منابع)/styles\.css"\s*/>'
    ),
    re.compile(r'px;top:px">.*\n'),
    re.compile(r"#تغییرمسیر"),
    re.compile(r"#تغییر_مسیر"),
    re.compile(r"\u200f"),  # right-to-left character
    re.compile(r"\u200e"),  # left-to-right character
    re.compile(r'<templatestyles src\s*=\s*".*\.css"\s*/>'),
    re.compile(r"</templatestyles>"),
    re.compile(r"\[ \[update\]\],"),
    re.compile(r"<onlyinclude></onlyinclude>"),
    re.compile(r"<noinclude>"),
    re.compile(r"</table>"),
    re.compile(r"!width=\d\d\|"),
    re.compile(r"</*td>"),
    re.compile(r"<table \[(class|style)\]=.*?>"),
    re.compile(r"\[\[(Categoría|Category):.*?\]\]"),
    re.compile(r"</*tr>"),
    re.compile(r"<br\s*>"),
    re.compile(r'<score vorbis="1">.*\|."'),
    re.compile(r'<score vorbis="1">.*?\s</score>'),
    re.compile(r"</score>"),
    re.compile(r"\{\{*Location map.*?coordinates\s=\s*\}+"),
    re.compile(
        r"-\{H.*?\}-"
    ),  # provides other Chinese variants for special Wikipedia phrases (e.g. "References")
    re.compile(r"-\{T.*;\}-"),  # provides otehr Chinese variants of article titles
    re.compile(r'</*(\s*rt\s*|\s*rp\s*|\s*ruby\s*|\s*rb\s*)>'),
    re.compile(r'(註释|註釋|注释|注釋)\.'),
    re.compile(r'\s\W\W(鏈接|链接|连接|連結)\.\s*'),
    re.compile(r'(參考|参考)\W\W\.\s'),
    re.compile(r'(\s\W\W|\s)(来源|來源)\.'),
    re.compile(r'\s(图书|圖書|建筑|建築|地点)\.'),
    re.compile(r'#重定向\s*'),
    
    # TODO check and uncomment
    # re.compile(r'-\{.*?\}-'),
    # re.compile(r'--\s\s\s/'),
    # re.compile(r'<table class=.*?>'),
    # re.compile(r'\{+#if.*?\}+'),
    # re.compile(r'\{+#ifeq.*?\}+'),
    # re.compile(r'\(*--\s*?\)*'),
    # re.compile(r'<section.*?/>'),
    # re.compile(r'<.*?/>'),
    # re.compile(r'\s\W{0,5}\.\s*'),
    # re.compile(r'(，|,)\](;|,|，|；、)?'),
    # re.compile(r'\(\s*\)'),
    # re.compile(r'（\s*?）'),  # Chinese brackets
    # re.compile(r'(“|")\s+("|”)'),
    # re.compile(r'(\(|（)(,|，|；|;|、)(\)|）)'),
]

# languages that don't use white space to denote word boundaries. For these languages, we use Stanza which is slower.
non_whitespace_language_list = ["zh"]

# cache for all-languages-to-english. e.g. global_translation_dict["fa"] is a dictionary of Farsi -> English translations
# values can be the emtpy string "", which means we have already looked up the translations in Wikidata, but did not find the English translation
# this is different from the case that the key is absent, which means we have never looked up that translation in Wikidata
global_translation_cache = {}


def get_from_translation_cache(source_language: str, entity: str):
    global global_translation_cache
    if (
        source_language not in global_translation_cache
        or entity not in global_translation_cache[source_language]
    ):
        return None
    return global_translation_cache[source_language][entity]


def load_global_cache_from_file(file_name: str):
    global global_translation_cache
    try:
        with open(file_name, "r") as f:
            global_translation_cache = json.load(f)
    except FileNotFoundError as e:
        logger.warning("Could not find the translation cache file at %s. Initializing the cache as an empty dictionary.", file_name)
        global_translation_cache = {}


def write_global_cache_to_file(file_name: str):
    global global_translation_cache
    with open(file_name, "w") as f:
        json.dump(global_translation_cache, f, indent=1, ensure_ascii=False)


def additional_cleaning(article_text: str):
    """
    clean what WikiExtractor misses
    """
    article_text = block_quote.sub(r'"\1"\n', article_text)
    for filter in css_tag_filters:
        article_text = filter.sub("", article_text)

    return article_text


def get_all_article_titles(path: str):
    all_titles = set()
    with open(path, "r") as wiki_file:
        for line in wiki_file:
            line = html.unescape(line)
            if line[:4] == "<doc":
                # new article starts
                to_get_title = True
            elif to_get_title:
                # the article title is the only thing in this line
                title = line.strip()
                all_titles.add(title)
                to_get_title = False
    return all_titles


def split_wiki_file_batch(
    paths: List[str], max_block_words: int, language: str, pipe: Pipe
) -> List[str]:
    if language in non_whitespace_language_list:
        stanza_pipeline = stanza.Pipeline(lang=language, processors="tokenize")
    else:
        stanza_pipeline = None

    file_batch_blocks = []
    for path in tqdm(paths, desc="Splitting into blocks", smoothing=0):
        file_blocks = split_wiki_file(path, max_block_words, language, stanza_pipeline)
        file_batch_blocks.extend(file_blocks)

    pipe.send(file_batch_blocks)
    pipe.close()


def split_wiki_file(
    path: str, max_block_words: int, language: str, stanza_pipeline=None
) -> List[str]:
    file_blocks = []
    with open(path, "r") as wiki_file:
        for line in wiki_file:
            line = html.unescape(line)
            if line[:4] == "<doc":
                # new article starts
                to_get_title = True
                article_content = ""
            elif line[:5] == "</doc":
                # the article has ended
                article_content = additional_cleaning(article_content)

                # add English translation to title
                cached_english = get_from_translation_cache(language, title)
                if (
                    language != "en"
                    and cached_english is not None
                    and len(cached_english) > 0
                ):
                    title = title + f" (In English: {cached_english})"
                # add English translation to article_content
                if language != "en":
                    # we handle links here, so that we can find the English translation of entities when needed
                    # including link tags won't affect block sizes, since they are not counted as extra "words"
                    link_matches = link_tag.findall(article_content)
                    for href, link_text in link_matches:
                        link_title = unquote(href)
                        cached_english = get_from_translation_cache(
                            language, link_title
                        )
                        if cached_english is not None and len(cached_english) > 0:
                            article_content = article_content.replace(
                                f'<a href="{href}">{link_text}</a>',
                                link_text + " (In English: " + cached_english + ")",
                            )
                        else:
                            logger.debug(
                                "Did not find link entity in Wikidata for %s",
                                link_title,
                            )

                # replace all remaining links with the text inside the tag
                article_content = link_tag.sub(r"\2", article_content)
                # if "href=" in article_content:
                # print(article_content)
                # exit()

                if language in non_whitespace_language_list:
                    doc = stanza_pipeline(article_content)
                    sentences = doc.sentences
                else:
                    sentences = nltk.sent_tokenize(
                        article_content, language="english"
                    )  # might need to use Stanza to support non-western language, but that would be much slower
                if len(sentences) == 0:
                    continue
                block_word_count = 0
                block = []
                for sentence in sentences:
                    if language in non_whitespace_language_list:
                        sentence_text = sentence.text
                    else:
                        sentence_text = sentence

                    sentence_text = sentence_text.strip().replace("\n", " ")

                    if language in non_whitespace_language_list:
                        sentence_word_count = len([t.text for t in sentence.tokens])
                    else:
                        sentence_word_count = len(sentence_text.split())
                    # if sentence_word_count > max_block_words:
                    # print(title+" | "+sentence)
                    if block_word_count + sentence_word_count > max_block_words:
                        # do not use the current sentence would push the block over the limit
                        if len(block) > 0:
                            file_blocks.append((title, " ".join(block)))
                        # start a new block
                        block = [sentence_text]
                        block_word_count = sentence_word_count
                    else:
                        block.append(sentence_text)
                        block_word_count += sentence_word_count
                # append the last block
                file_blocks.append((title, " ".join(block)))
            elif to_get_title:
                # the article title is the only thing in this line
                title = line.strip()
                to_get_title = False
            else:
                # normal line from the article
                article_content += line
    return file_blocks


async def get_wikidata_english_name(article_title: str, session, language: str):
    """
    Returns
        (english_name: str, new_translation_dict: dict)
    """
    global global_translation_cache
    if get_from_translation_cache(language, article_title) is not None:
        return get_from_translation_cache(language, article_title), {}
    try:
        # the API expects a user agent
        async with session.get(
            url=f"https://www.wikidata.org/w/api.php?action=wbgetentities&normalize=0&sites={language}wiki&titles={quote(article_title, safe='')}&format=json",
            headers={"User-Agent": "wikichat/1.0"},
        ) as response:
            a = await response.json()
            wikidata_entity = a["entities"]
            assert len(wikidata_entity) == 1, "found 0 or >1 Wikidata entities"

            wikidata_entity = list(wikidata_entity.items())[0][1]
            english_locale = None
            if "labels" not in wikidata_entity:
                logger.debug(
                    "Did not find any labels in the Wikidata entry %s",
                    str(wikidata_entity),
                )
                return None, {language: {article_title: ""}}

            if "en" in wikidata_entity["labels"]:
                english_locale = "en"
            elif "en-gb" in wikidata_entity["labels"]:
                english_locale = "en-gb"
            elif "en-ca" in wikidata_entity["labels"]:
                english_locale = "en-ca"
            else:
                logger.debug(
                    "Did not find any English labels in Wikidata for %s", article_title
                )
                return None, {language: {article_title: ""}}
            english_name = wikidata_entity["labels"][english_locale]["value"]

            new_translation_dict = {}
            for lang in wikidata_entity["labels"].keys():
                new_translation_dict[lang] = {
                    wikidata_entity["labels"][lang]["value"]: english_name
                }
            # Add the actual article title as well. Sometimes it is a different version of what we find in Wikidata.
            # E.g. the article title is "Legazpi (Madrid)" but the Wikidata returns "Legazpi"
            if language not in new_translation_dict:
                # this sometimes happens due to the Wikidata entry being incomplete
                new_translation_dict[language] = {}
            new_translation_dict[language][article_title] = english_name

            return english_name, new_translation_dict
    except Exception as e:
        logger.error(
            "Unable to get entry for article '%s' due to error %s.",
            article_title,
            str(e),
        )
        return -1, {language: {article_title: ""}}


async def batch_get_wikidata_english_name(article_titles: List[str], language: str):
    global global_translation_cache
    async with aiohttp.ClientSession() as session:
        with logging_redirect_tqdm():
            minibatch_size = 100  # The wikipedia API only allows 100 requests per second, so we batch the requests.
            for i in trange(
                0,
                len(article_titles),
                minibatch_size,
                desc="Getting Wikidata entries",
                smoothing=0,
            ):
                start_time = time()
                batch = article_titles[i : min(len(article_titles), i + minibatch_size)]
                ret = await asyncio.gather(
                    *[
                        get_wikidata_english_name(article_title, session, language)
                        for article_title in batch
                    ]
                )  # ret is a list of tuples

                # convert to two lists
                batch_english_names, batch_new_translation_dicts = tuple(zip(*ret))
                batch_english_names, batch_new_translation_dicts = list(
                    batch_english_names
                ), list(batch_new_translation_dicts)

                if -1 in batch_english_names:
                    logger.debug("Reached the Wikidata rate limit. Will wait longer.")
                    # -1 is used to signal a rate limit. wait for longer if we get an rate limit error
                    await asyncio.sleep(2)

                # Add to the global translation dictionary
                for translation_dict in batch_new_translation_dicts:
                    for lang in translation_dict.keys():
                        if lang not in global_translation_cache:
                            global_translation_cache[lang] = {}
                        for k, v in translation_dict[lang].items():
                            global_translation_cache[lang][k] = v
                time_passed = time() - start_time
                time_to_wait = 1 - time_passed
                if time_to_wait > 0:
                    await asyncio.sleep(time_to_wait)


def split_passages(all_input_files, args):
    if args.language != "en":
        file_titles = process_map(
            get_all_article_titles,
            all_input_files,
            max_workers=cpu_count(),
            chunksize=10,
            desc="Reading article titles",
        )
        all_titles = []
        for t in file_titles:
            all_titles.extend(t)
        logger.info("Number of unique articles: %d", len(all_titles))

        non_cached_titles = []
        for article_title in all_titles:
            if get_from_translation_cache(args.language, article_title) is None:
                non_cached_titles.append(article_title)

        logger.info(
            "Did not find %d articles in the cache, will call the Wikidata API for them",
            len(non_cached_titles),
        )

        asyncio.run(batch_get_wikidata_english_name(non_cached_titles, args.language))

    # since we have a lot of input files, we easily parallelize the work with multiprocessing
    # we cannot use process_map() because we want to keep a fixed number of Stanza pipelines
    num_workers = cpu_count()
    all_parent_connections = []
    all_processes = []
    split_size = math.ceil(len(all_input_files) / num_workers)
    for batch_idx in range(num_workers):
        parent_conn, child_connection = Pipe()
        p = Process(
            target=split_wiki_file_batch,
            args=(
                all_input_files[
                    batch_idx
                    * split_size : min(
                        len(all_input_files), (batch_idx + 1) * split_size
                    )
                ],
                args.max_block_words,
                args.language,
                child_connection,
            ),
        )
        all_parent_connections.append(parent_conn)
        all_processes.append(p)
        p.start()

    all_blocks = []
    for p_idx in range(len(all_processes)):
        file_batch_blocks = all_parent_connections[p_idx].recv()
        all_blocks.extend(file_batch_blocks)

        all_parent_connections[p_idx].close()
        all_processes[p_idx].join()

    # file_blocks = process_map(
    #     split_wiki_file,
    #     all_input_files,
    #     [args.max_block_words] * len(all_input_files),
    #     [args.language] * len(all_input_files),
    #     max_workers=2,  # min(cpu_count(), 4) if args.language in non_whitespace_language_list else cpu_count(), # have to lower it so that stanza doesn't overwhelm the GPU
    #     chunksize=10,
    #     desc="Splitting into blocks",
    #     smoothing=0,
    # )
    # for b in file_blocks:
    # all_blocks.extend(b)

    logger.info("Number of blocks: %d", len(all_blocks))

    # write the resulting blocks to file
    # make parent directories
    pathlib.Path(os.path.dirname(args.output_path)).mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as output_file:
        for i in trange(len(all_blocks), desc=f"Writing blocks to {args.output_path}"):
            title, text = all_blocks[i]
            output_file.write(
                str(i) + "\t" + title + " " + args.title_separator + " " + text + "\n"
            )


def load_collection(collection_path):
    collection = []

    logger.info("Collection path is %s", collection_path)

    with open(collection_path) as f:
        for line_idx, line in enumerate(f):
            pid, passage, *rest = line.strip("\n\r ").split("\t")
            assert pid == "id" or int(pid) == line_idx

            if len(rest) >= 1:
                title = rest[0]
                passage = title + " | " + passage

            collection.append(passage)

    return collection


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--input_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)
    arg_parser.add_argument(
        "--translation_cache",
        type=str,
        help="Where to read/write the translation cache.",
    )
    arg_parser.add_argument("--title_separator", type=str, default="|")
    arg_parser.add_argument(
        "--max_block_words",
        type=int,
        default=120,
        help="Maximum number of words to include in each output",
    )
    arg_parser.add_argument(
        "--num_input_files",
        type=int,
        default=-1,
        help="Specify a positive integer to limit the number of files. Used for debugging. -1 to split all files in the directory.",
    )
    arg_parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="If specified other than `en`, will query Wikidata to find English translation of titles.",
    )

    args = arg_parser.parse_args()

    all_input_files = []
    for dir in os.listdir(args.input_path):
        for file in os.listdir(os.path.join(args.input_path, dir)):
            all_input_files.append(os.path.join(args.input_path, dir, file))

    if args.num_input_files > 0:
        all_input_files = all_input_files[: args.num_input_files]
        logger.info("all_input_files = %s", str(all_input_files))

    if args.language != "en":
        logger.info("Loading the translation cache file at %s", args.translation_cache)
        load_global_cache_from_file(args.translation_cache)

    split_passages(all_input_files, args)

    if args.language != "en":
        logger.info("Saving the translation cache file to %s", args.translation_cache)
        write_global_cache_to_file(args.translation_cache)

    # test to see the output file is readable by ColBERT's indexing code
    # load_collection(args.output_path)
