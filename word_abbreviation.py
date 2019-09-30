from collections import defaultdict
import json
import logging
import pickle
from pathlib import Path
import re
import spacy
import time

logger = logging.getLogger("word_abbreviation")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def cleanup_punc(input: list) -> list:
    """Clean up the input vocabulary and return a new vocabulary as list"""
    re_punc = re.compile(
        r"\w*[!\"#$%&\'()*+,.\/\-_:;<=>?@\[\]\\^`{|}~]+\w*", flags=re.ASCII
    )
    re_letter_start = re.compile(r"^[a-zA-Z]+")
    output = []

    for i, w in enumerate(input):
        if re_punc.match(w) or not re_letter_start.match(w):
            continue
        output.append(w)
    return output


def abbreviation(short: str, long: str) -> bool:
    """Determine whether input `short` is part of input `long`.
    Note it could be discontinuous.
    """
    shortc = [c for c in short]
    longc = [c for c in long]
    cs = None
    while longc:
        cl = longc.pop() if longc else None
        if not cs:
            cs = shortc.pop() if shortc else None
        if cl == cs:
            cs = None
    if len(shortc) == 0 and cs is None:
        return True
    else:
        return False


def map_vocab_to_index(v: list) -> dict:
    """Clean up the input vocabulary and return a dictionary of keys as individual
    characters and each value containing a subset of vocabulary containing the character."""
    vc = cleanup_punc(v)
    logger.info(f"Raw vocabulatory size: {len(v)}")
    logger.info(f"Cleaned-up vocabulatory size: {len(v)}")

    vocab_index = defaultdict(set)
    for i, w in enumerate(vc):
        for c in w:
            vocab_index[c].add(w)
    return vocab_index


def get_candidate_words(word: str, vocab_index: dict) -> set:
    """
    Input:
      word: str, input word
      vocab_index, dict of word index: The keys are individual characters
            and each value containing a subset of vocabulary containing the character.
    Return a candidate set of words in the vocabulary that could be abbreviated as input word
    """
    ns = set()
    for c in word:
        if len(ns) == 0:
            ns = vocab_index[c].copy()
        else:
            ns = ns.intersection(vocab_index[c])

    matched = set()
    for w in ns:
        if abbreviation(word, w):
            matched.add(w)
    return matched


def main(word_list, rebuild_index=False):
    """Process a list of input words
    Input:
      word_list: list of strings as words
      rebuild_index: bool  If True, force rebuilding of vocabulary index
    """
    vocab_index = None
    if rebuild_index or not Path("vocab_index.pkl").exists():
        if rebuild_index:
            logger.info("Rebuilding vocabulary index...")
        else:
            logger.info("Building vocabulary index...")
        tic = time.time()
        nlp = spacy.load("en_core_web_md")
        toc = time.time()
        logger.info(f"It took {toc - tic:.1f} secs to load spaCy language model.")

        tic = time.time()
        v = [k.text for k in nlp.vocab]
        vocab_index = map_vocab_to_index(v)
        toc = time.time()
        logger.info(f"It took {toc - tic:.1f} secs to create the vocabulary index.")
        try:
            with open("vocab_index.pkl", "wb") as f:
                pickle.dump(vocab_index, f)
        except pickle.PickleError as e:
            logger.error(f"Error saving pickled vocab index file: {e}.")
            quit(2)
    else:
        logger.info("Found existing vocabulary index.  Loading...")
        tic = time.time()
        try:
            with open("vocab_index.pkl", "rb") as f:
                vocab_index = pickle.load(f)
        except pickle.PickleError as e:
            logger.error(f"Error loading pickled vocab index file: {e}.")
            logger.error("Try recreate the vocab index.")
            quit(1)
        toc = time.time()
        logger.info(
            f"It took {toc - tic:.1f} secs to load the existing vocabulary index."
        )

    tic = time.time()
    if len(word_list) == 0:
        matched = get_candidate_words("coffee", vocab_index)
        output = {"coffee": list(matched)}
    else:
        output = {}
        for w in word_list:
            matched = get_candidate_words(w, vocab_index)
            output.update({w: list(matched)})
    toc = time.time()
    logger.info(
        f"It took {toc - tic:.1f} secs for {len(word_list)} search{'es' if len(word_list)>1 else ''}."
    )
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Return candidate full words for input abbreviated words"
    )
    parser.add_argument(
        "input",
        metavar="input",
        type=str,
        nargs="*",
        default="coffee",
        help="a list of words",
    )
    parser.add_argument(
        "--rebuild-index",
        type=bool,
        default=False,
        help="force rebuilding the vocab index",
    )
    args = vars(parser.parse_args())
    inputs = args["input"]
    rebuild_index = args["rebuild_index"]

    if not isinstance(inputs, list):
        inputs = [inputs]
    output = main(inputs, rebuild_index=rebuild_index)
    print(json.dumps(output, indent=2))
