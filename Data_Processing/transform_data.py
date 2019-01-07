import re
import os

PROJECT_PATH = os.path.realpath(os.path.dirname(__file__) + '/..')

MODEL1VERSION1 = PROJECT_PATH + '/fastText/data.txt'
MODEL1VERSION2 = PROJECT_PATH + '/fastText/gender_neutral_data.txt'

MODEL2VERSION1 = PROJECT_PATH + '/fastText/model2initialdata.txt'
MODEL2VERSION2 = PROJECT_PATH + '/fastText/model2transformeddata.txt'

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
    Arguments:
        string (str): The string in which the words should be replaced.
        replacements (List[List[str, str]]): A list of [old, new] replacement word pairs.
    Returns:
        str: The original string with the words replaced.
    """
    s = string
    # replace everything to a marked version
    # this prevents "mutual" replacements to overwrite each other
    for old, new in replacements:
        pattern = r'\b' + old + r'\b'
        replacement = new + MARKER
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)
    # remove the mark
    return s.replace(MARKER, '')


def write_me(original_filepath, new_filepath):
    with open(original_filepath) as fin:
        with open(new_filepath, 'w') as fout:
            for line in fin:
                fout.write(replace(line, REPLACEMENTS))
    with open(original_filepath) as fin:
        with open(new_filepath, 'a') as fout:
            for line in fin:
                fout.write(line)


def main():
    write_me(MODEL2VERSION1, MODEL2VERSION2)


if __name__ == '__main__':
    main()
