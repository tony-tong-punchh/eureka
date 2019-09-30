from collections import defaultdict
import json
import spacy
import time


def cleanup_punc(input: list) -> list:
    """Clean up the input vocabulary and return a new vocabulary as list"""
    import re

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
    print(f"Raw vocabulatory size: {len(v)}")
    print(f"Cleaned-up vocabulatory size: {len(v)}")

    ds = defaultdict(set)
    for i, w in enumerate(vc):
        for c in w:
            ds[c].add(w)
    return ds


def get_candidate_words(word="large") -> set:
    """Return a candidate set of words in the vocabulary that could be abbreviated as input word
    """
    ns = set()
    for c in word:
        if len(ns) == 0:
            ns = ds[c].copy()
        else:
            ns = ns.intersection(ds[c])

    matched = set()
    for w in ns:
        if abbreviation(word, w):
            matched.add(w)
    return matched


if __name__ == "__main__":
    tic = time.time()
    nlp = spacy.load("en_core_web_md")
    toc = time.time()
    print(f"It took {toc - tic:.1f} secs to load spaCy language model.")

    tic = time.time()
    v = [k.text for k in nlp.vocab]
    ds = map_vocab_to_index(v)
    toc = time.time()
    print(f"It took {toc - tic:.1f} secs to create index.")

    tic = time.time()
    matched = get_candidate_words()
    toc = time.time()
    print(f"It took {toc - tic:.1f} secs for one search.")
    print("Total number of candidate words: ", len(matched))
    print(json.dumps(list(matched), indent=2))
