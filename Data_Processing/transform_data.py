import re

ORIGINAL_FILEPATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/fastText/data.txt'
NEW_FILEPATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/fastText/gender_neutral_data.txt'

MARKER = '_CHLOE_ROCKS'

REPLACEMENTS = [
    ['she','he'],
    ['her','him'],
    ['hers','his'],
    ['he','she'],
    ['him','her'],
    ['his','hers'],
    ['herself', 'himself'],
    ['himself', 'herself'],
    ["she's", "he's"]
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


with open(ORIGINAL_FILEPATH) as fin:
    with open(NEW_FILEPATH, 'w') as fout:
        for line in fin:
            fout.write(replace(line, REPLACEMENTS))

with open(ORIGINAL_FILEPATH) as fin:
    with open(NEW_FILEPATH, 'a') as fout:
        for line in fin:
            fout.write(line)