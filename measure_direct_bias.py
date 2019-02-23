import os
import subprocess

import numpy as np
from gensim.models import FastText
from sklearn.decomposition import PCA

from util import CORPORA_PATH, MODELS_PATH, DIRECTIONS_PATH, WORDS_PATH, list_files
from transform_corpus import create_pronoun_swapped_corpus

from src.embeddings import Embedding, WrappedEmbedding
from evaluate import read_dataset_directory, score_embedding


def simple_gender_direction(model, wrd_1, wrd_2):
    # ASSUMES WORD 1 IS FEMALE, WORD 2 is MALE
    fem_vec_norm = model.wv[wrd_1] / np.linalg.norm(model.wv[wrd_1], ord=1)
    male_vec_norm = model.wv[wrd_2] / np.linalg.norm(model.wv[wrd_2], ord=1)
    subtraction = np.array(np.subtract(fem_vec_norm, male_vec_norm))
    subtraction = subtraction / np.linalg.norm(subtraction, ord=1)
    return subtraction

def define_gender_direction_mean(model, direction_file):
    """
    Create a gender direction by averaging dimensions of all gender words.

    Arguments:
        model (Model): A Gensim word embedding model.
        direction_file (str): A file of male-female word pairs.

    Returns:
         Vector: a male->female vector.
    """
    with open(direction_file) as fd:
        male_words, female_words = list(zip(*(line.split() for line in fd)))
    fem_avg_vec, male_avg_vec = [],[]
    num_male_words, num_fem_words = len(male_words),len(female_words)
    for i in range(len(model.wv[female_words[0]])): # loop through all dimensions
        fem_sum, male_sum = 0, 0
        for j in range(len(male_words)):
            # MALE VECTORS
            if male_words[j] in model.wv:
                male_vec_norm = model.wv[male_words[j]] / np.linalg.norm(model.wv[male_words[j]], ord=1)
                male_sum += male_vec_norm[i]
            else:
                num_male_words -= 1
            # FEMALE VECTORS
            if female_words[j] in model.wv:
                fem_vec_norm = model.wv[female_words[j]] / np.linalg.norm(model.wv[female_words[j]], ord=1)
                fem_sum += fem_vec_norm[i]
            else:
                num_fem_words -= 1
        fem_avg_vec.append(fem_sum / num_fem_words)
        male_avg_vec.append(male_sum / num_male_words)
    fem_avg_vec = fem_avg_vec / np.linalg.norm(fem_avg_vec, ord=1)
    male_avg_vec = male_avg_vec / np.linalg.norm(male_avg_vec, ord=1)
    subtraction = np.array(np.subtract(fem_avg_vec, male_avg_vec))
    subtraction = subtraction / np.linalg.norm(subtraction, ord=1)
    return subtraction

def define_gender_direction_pca(model, direction_file):
    """
    Create a gender direction using PCA.

    Arguments:
        model (Model): A Gensim word embedding model.
        direction_file (str): A file of male-female word pairs.

    Returns:
        Vector: A male->female vector.
    """
    with open(direction_file) as fd:
        male_words, female_words = list(zip(*(line.split() for line in fd)))
    matrix = []
    for female_word, male_word in zip(female_words, male_words):
        if female_word in model.wv and male_word in model.wv:
            fem_vec_norm = model.wv[female_word] / np.linalg.norm(model.wv[female_word], ord=1)
            male_vec_norm = model.wv[male_word] / np.linalg.norm(model.wv[male_word], ord=1)
            center = (fem_vec_norm + male_vec_norm) / 2
            matrix.append(fem_vec_norm - center)
            matrix.append(male_vec_norm - center)
    matrix = np.array(matrix)
    pca = PCA()
    pca.fit(matrix)
    pca_norm = pca.components_[0] / np.linalg.norm(pca.components_[0], ord=1)
    return pca_norm


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
    if word not in model.wv:
        return None
    word_vector = model.wv[word] / np.linalg.norm(model.wv[word], ord=1)
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
        if word in model.wv:
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


def run_experiment_1(model, direction_file, words_file, subspace_type):
    """Run a word embedding bias experiment.

    Arguments:
        model (Model): A Gensim word embedding model.
        direction_file (str): A file path of direction word pairs.
        words_file (str): A file path of neutral words.

    Returns:
        List: The average bias of the model, wrt pca gender subspace and averaged gender subspace.
    """
    words = read_words_file(words_file)
    if subspace_type == 'MEAN':
        direction = define_gender_direction_mean(model, direction_file)
        bias = calculate_model_bias(model, direction, words)
    else:
        direction = define_gender_direction_pca(model, direction_file)
        bias = calculate_model_bias(model, direction, words)
    return bias

def run_experiment_2(model, direction_file, words_file):
    words = read_words_file(words_file)
    with open(direction_file) as fd:
        male_words, female_words = list(zip(*(line.split() for line in fd)))
    for i in range(len(male_words)):
        print('GENDER DIRECTION:', female_words[i], male_words[i])
        direction = simple_gender_direction(model, female_words[i], male_words[i])
        print("GENDER BIAS:", calculate_model_bias(model, direction, words))



def build_all_fasttext_models(model_type='skipgram'):
    if model_type not in ['skipgram', 'cbow']:
        raise ValueError('model_type must be "skipgram" or "cbow" but got "' + str(model_type) + '"')
    for corpus_file in list_files(CORPORA_PATH):
        if not corpus_file.endswith('-swapped'):
            create_pronoun_swapped_corpus(corpus_file)
    for corpus_file in list_files(CORPORA_PATH):
        model_stub = os.path.join(
            MODELS_PATH,
            os.path.basename(corpus_file) + '.' + model_type,
        )
        if not os.path.exists(model_stub + '.bin'):
            subprocess.run(
                args=['fasttext', 'skipgram', '-input', corpus_file, '-output', model_stub]
            )

def pretty_print(filename):
    print(filename.rsplit('/', 1)[-1])

def main():

    """Entry point for the project."""
    #EXP = int(input("ENTER EXPERIMENT NUMBER: "))
    mdl = 'FastText'
    corp = 'Wikipedia'
    subspaces = ['MEAN', 'PCA']
    kwargs = {'supports_phrases': False,
              'google_news_normalize': False}
    build_all_fasttext_models('skipgram')
    model_files = list_files(MODELS_PATH)
    model_files = [file for file in model_files if file.endswith('.bin')]
    direction_files = list_files(DIRECTIONS_PATH)
    words_files = list_files(WORDS_PATH)
    #if EXP == 1:
    print("Model Corpus Debias Gender_Words Gender_Subspace Bias_Words Bias")
    for model_file in model_files:
        if 'MODEL1' in model_file.rsplit('/', 1)[-1]:
            debias = 'None'
        else:
            debias = 'Pronoun-Swap'
        model = FastText.load_fasttext_format(model_file)
        embedding = WrappedEmbedding.from_fasttext(model_file, **kwargs)
        print("MODEL EVALUATION; {}:".format(str(model_file.rsplit('/', 1)[-1])))
        dataset = list(read_dataset_directory('wiki-sem-500/en'))
        score_embedding(embedding, dataset)
        for direction_file in direction_files:
            direction = direction_file.rsplit('/', 1)[-1]
            for words_file in words_files:
                bias_wrds = words_file.rsplit('/', 1)[-1]
                for g in subspaces:
                    bias = run_experiment_1(model, direction_file, words_file, g)
                    print(mdl, corp, debias, direction, g, bias_wrds, bias)


    '''
    Experiment 2 prints the measured direct bias for each category of gender-neutral words, on the basis of
    gender directions that are the pairwise subtractions of gender pairs.
    It should generate:
        - The bias using each one of the pairs alone
        - The bias using the averaged vector from each file
        - The bias using the averaged vector from all files

    if EXP == 2:
        for model_file in model_files:
            model = FastText.load_fasttext_format(model_file)
            embedding = WrappedEmbedding.from_fasttext(model_file, **kwargs)
            print("MODEL EVALUATION; {}:".format(str(model_file.rsplit('/', 1)[-1])))
            dataset = list(read_dataset_directory('wiki-sem-500/en'))
            score_embedding(embedding, dataset)
            print('\n\n')
            print("EXPERIMENTAL RESULTS - Individual Pairs:")
            for direction_file in direction_files:
                for words_file in words_files:
                    pretty_print(model_file)
                    pretty_print(direction_file)
                    pretty_print(words_file)
                    run_experiment_2(model, direction_file, words_file)
                    print()
    '''

if __name__ == '__main__':
    main()
