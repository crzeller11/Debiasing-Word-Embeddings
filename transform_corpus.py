import re
import os
import sys

MARKER = '_CHLOE_ROCKS'

REPLACEMENTS = [
    ['she', 'he'],
    ['he', 'she'],
    ["she'll", "he'll"],
    ["he'll", "she'll"],
    ['her', 'him'],
    ['him', 'her'],
    ['hers', 'his'],
    ['his', 'hers'],
    ['herself', 'himself'],
    ['himself', 'herself'],
    ["she's", "he's"],
    ["he's", "she's"],
    ["she'd", "he'd"],
    ["he'd", "she'd"]
]


def replace(string, replacements):
    """Replace a set of words with another.

    This function uses a temporary marked version to prevent "mutual"
    replacements from overwriting each other.

    Arguments:
        string (str): The string in which the words should be replaced.
        replacements (List[List[str, str]]): A list of [old, new] replacement word pairs.

    Returns:
        str: The original string with the words replaced.
    """
    for old, new in replacements:
        pattern = r'\b' + old + r'\b'
        replacement = new + MARKER
        string = re.sub(pattern, replacement, string, flags=re.IGNORECASE)
    return string.replace(MARKER, '')


def get_new_corpus_filepath(orig_filepath):
    stub, ext = os.path.splitext(os.path.basename(orig_filepath))
    new_filename = stub + '-swapped' + ext
    return os.path.join(os.path.dirname(orig_filepath), new_filename)


def create_pronoun_swapped_corpus(orig_filepath):
    new_filepath = get_new_corpus_filepath(orig_filepath)
    if os.path.exists(new_filepath):
        return
    with open(orig_filepath) as fin:
        with open(new_filepath, 'w') as fout:
            for line in fin:
                fout.write(replace(line, REPLACEMENTS))
    with open(orig_filepath) as fin:
        with open(new_filepath, 'a') as fout:
            for line in fin:
                fout.write(line)


def main():
    filepath = None
    if len(sys.argv) == 2:
        filepath = os.path.realpath(os.path.expanduser(sys.argv[1]))
    else:
        print('usage: ' + sys.argv[0] + ' <filepath>')
        exit(1)
    create_pronoun_swapped_corpus(filepath)


if __name__ == '__main__':
    main()
