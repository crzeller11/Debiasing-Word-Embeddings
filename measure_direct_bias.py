import numpy as np
from gensim.models import FastText
from sklearn.decomposition import PCA

from util import MODELS_PATH, DIRECTIONS_PATH, WORDS_PATH, list_files


def define_gender_direction_pca(model, direction_file):
    """Create a gender direction using PCA.

    Arguments:
        model (Model): A Gensim word embedding model.
        direction_file (str): A file of male-female word pairs.

    Returns:
        Vector: A male->female vector.
    """
    with open(direction_file) as fd:
        male_words, female_words = list(zip(*(line.split() for line in fd)))
    female_vectors = []
    male_vectors = []
    for female_word, male_word in zip(female_words, male_words):
        if female_word in model and male_word in model:
            female_vectors.append(model[female_word])
            male_vectors.append(model[male_word])
    subtraction = np.array([
        np.subtract(female, male) for female, male in zip(female_vectors, male_vectors)
    ])
    pca = PCA()
    pca.fit(subtraction)
    return pca.components_[0]


def calculate_word_bias(model, direction, word, strictness=1):
    """Calculate the bas of a word.

    Arguments:
        model (Model): A Gensim word embedding model.
        direction (Vector): A direction against which to measure the bias.
        word (str): The word for which measure the bias.
        strictness (float): Exponential scaling parameter for the bias.

    Returns:
        float: The bias of the word.
    """
    if word not in model:
        return None
    word_vector = model[word] / np.linalg.norm(model[word], ord=1)
    direction_vector = direction / np.linalg.norm(direction, ord=1)
    return np.dot(word_vector, direction_vector)**strictness


def calculate_model_bias(model, direction, words, strictness=1):
    """Calculate the direct bias statistic for a model.

    Arguments:
        model (Model): A Gensim word embedding model.
        direction (Vector): A direction against which to measure the bias.
        words (List[str]): The words for which measure the bias.
        strictness (float): Exponential scaling parameter for the bias.

    Returns:
        float: The average bias of the model.
    """
    abs_total = 0
    count = 0
    for word in words:
        if word in model:
            count += 1
            abs_total += abs(calculate_word_bias(model, direction, word, strictness=strictness))
    return abs_total / count


def read_words_file(words_file):
    """Read in a words file.

    Arguments:
        words_file (str): A file path of neutral words.

    Returns:
        List(str): The words in the file.
    """
    words = []
    with open(words_file) as fd:
        for line in fd.readlines():
            words.extend(line.split())
    return words


def run_experiment(model, direction_file, words_file):
    """Run a word embedding bias experiment.

    Arguments:
        model (Model): A Gensim word embedding model.
        direction_file (str): A file path of direction word pairs.
        words_file (str): A file path of neutral words.

    Returns:
        float: The average bias of the model.
    """
    words = read_words_file(words_file)
    direction = define_gender_direction_pca(model, direction_file)
    return calculate_model_bias(model, direction, words)


def main():
    """Entry point for the project."""
    model_files = list_files(MODELS_PATH)
    model_files = [file for file in model_files if file.endswith('.bin')]
    direction_files = list_files(DIRECTIONS_PATH)
    words_files = list_files(WORDS_PATH)
    for model_file in model_files:
        model = FastText.load_fasttext_format(model_file)
        for direction_file in direction_files:
            for words_file in words_files:
                bias = run_experiment(model, direction_file, words_file)
                print(model_file)
                print(direction_file)
                print(words_file)
                print(bias)


if __name__ == '__main__':
    main()
