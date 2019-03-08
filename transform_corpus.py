import re
import os
import sys
from itertools import product

MARKER = '_CHLOE_ROCKS'

CONTRACTIONS = [
    "'ll",
    "'s",
    "'d",
    "'ve",
]


def replace(string, replacements):
    """Replace a set of words with another.

    This function uses a temporary marked version to prevent "mutual"
    replacements from overwriting each other.

    Arguments:
        string (str): The string in which the words should be replaced.
        replacements (List[Tuple[str, str]]): A list of replacement word pairs.

    Returns:
        str: The original string with the words replaced.
    """
    for word1, word2 in replacements:
        # "forwards"
        pattern = r'\b' + word1 + r'\b'
        replacement = word2 + MARKER
        string = re.sub(pattern, replacement, string, flags=re.IGNORECASE)
        # "backwards"
        pattern = r'\b' + word2 + r'\b'
        replacement = word1 + MARKER
        string = re.sub(pattern, replacement, string, flags=re.IGNORECASE)
    return string.replace(MARKER, '')


def load_swap_words(filepath):
    with open(filepath) as fd:
        root_swaps = [tuple(line.strip().split()) for line in fd]
    return root_swaps + [
        (word1 + contraction, word2 + contraction)
        for (word1, word2), contraction
        in product(root_swaps, CONTRACTIONS)
    ]


def get_new_corpus_filepath(orig_filepath):
    stub, ext = os.path.splitext(os.path.basename(orig_filepath))
    new_filename = stub + '-swapped' + ext
    return os.path.join(os.path.dirname(orig_filepath), new_filename)


def create_pronoun_swapped_corpus(orig_corpus_file, swap_file):
    replacements = load_swap_words(swap_file)
    new_filepath = get_new_corpus_filepath(orig_corpus_file)
    if os.path.exists(new_filepath):
        return
    with open(orig_corpus_file) as fin:
        with open(new_filepath, 'w') as fout:
            for line in fin:
                fout.write(replace(line, replacements))
    with open(orig_corpus_file) as fin:
        with open(new_filepath, 'a') as fout:
            for line in fin:
                fout.write(line)


def main():
    if len(sys.argv) == 3:
        corpus_file = os.path.realpath(os.path.expanduser(sys.argv[1]))
        swap_file = os.path.realpath(os.path.expanduser(sys.argv[2]))
        create_pronoun_swapped_corpus(corpus_file, swap_file)
    else:
        print('usage: ' + sys.argv[0] + ' <corpus_file> <swap_file>')
        exit(1)


if __name__ == '__main__':
    main()
