import os
import subprocess
import json
from itertools import groupby
from functools import lru_cache

import numpy as np
from sklearn.decomposition import PCA

from util import CORPORA_PATH, MODELS_PATH, DIRECTIONS_PATH, WORDS_PATH, list_files
from transform_corpus import create_pronoun_swapped_corpus

from permspace import PermutationSpace

from blair import WrappedEmbedding
from blair import read_dataset_directory, score_embedding
from bolukbasi.we import WordEmbedding as BolukbasiEmbedding
from bolukbasi.debias import debias as bolukbasi_debias


class EmbeddingAdaptor:

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.bolukbasi = not isinstance(wrapped, WrappedEmbedding)

    def __contains__(self, key):
        if self.bolukbasi:
            return key.lower() in self.wrapped.index
        else:
            return key in self.wrapped

    def __getitem__(self, key):
        if self.bolukbasi:
            return self.wrapped.v(key)
        else:
            return self.wrapped.get_vector(key)


def normalize(vector):
    return vector / np.linalg.norm(vector, ord=1)


def define_gender_direction_mean(model, male_words, female_words):
    """
    Create a gender direction by averaging dimensions of all gender words.

    Arguments:
        model (Model): A Gensim word embedding model.
        direction_file (str): A file of male-female word pairs.

    Returns:
         Vector: a male->female vector.
    """
    diff_vectors = []
    for male_word, female_word in zip(male_words, female_words):
        if not (male_word in model and female_word in model):
            continue
        diff_vector = model[female_word] - model[male_word]
        diff_vectors.append(normalize(diff_vector))
    if not diff_vectors:
        return None
    result = np.mean(np.array(diff_vectors), axis=0)
    return result


def define_gender_direction_pca(model, male_words, female_words):
    """
    Create a gender direction using PCA.

    Arguments:
        model (Model): A Gensim word embedding model.
        direction_file (str): A file of male-female word pairs.

    Returns:
        Vector: A male->female vector.
    """
    matrix = []
    for female_word, male_word in zip(female_words, male_words):
        if female_word in model and male_word in model:
            fem_vec_norm = normalize(model[female_word])
            male_vec_norm = normalize(model[male_word])
            center = (fem_vec_norm + male_vec_norm) / 2
            matrix.append(fem_vec_norm - center)
            matrix.append(male_vec_norm - center)
    if not matrix:
        return None
    matrix = np.array(matrix)
    pca = PCA()
    pca.fit(matrix)
    pca_norm = normalize(pca.components_[0])
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
    if word not in model:
        return None
    word_vector = normalize(model[word])
    direction_vector = normalize(direction)
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


def load_bias_words(bias_words):
    """Read in a words file.

    Arguments:
        words_file (str): The bias_words parameter.

    Returns:
        List(str): The words in the file.
    """
    words_file = 'words/' + bias_words + '.txt'
    words = []
    with open(words_file) as fd:
        for line in fd.readlines():
            words.extend(line.split())
    return words


def run_model_evaluation(model_file):
    kwargs = {'supports_phrases': False,
              'google_news_normalize': False}
    embedding = EmbeddingAdaptor(WrappedEmbedding.from_fasttext(model_file, **kwargs))
    dataset = list(read_dataset_directory('wiki-sem-500/en'))
    opp, accuracy = score_embedding(embedding, dataset)
    return opp, accuracy


def load_bolukbasi_model():
    embedding_filename = 'models/fasttext-biased.bin'
    definitional_filename = 'data/definitional_pairs.json'
    gendered_words_filename = 'data/gender_specific_full.json'
    equalize_filename = 'data/equalize_pairs.json'
    with open(definitional_filename, "r") as fd:
        defs = json.load(fd)
    with open(equalize_filename, "r") as fd:
        equalize_pairs = json.load(fd)
    with open(gendered_words_filename, "r") as fd:
        gender_specific_words = json.load(fd)
    embedding = BolukbasiEmbedding(embedding_filename)
    bolukbasi_debias(embedding, gender_specific_words, defs, equalize_pairs)
    return embedding


@lru_cache()
def load_debaised_model(model, debias):
    """Load a debiased word embedding model.

    Arguments:
        model (str): The model algorithm used.
        debias (str): The debiassing method used.

    Returns:
        Mapping[str, Vector]: A word embedding..
    """
    # FIXME deal with model algo
    kwargs = {
        'supports_phrases': False,
        'google_news_normalize': False,
    }
    if debias == 'none':
        return EmbeddingAdaptor(WrappedEmbedding.from_fasttext('models/fasttext-biased.bin', **kwargs))
    elif debias == 'wordswap':
        return EmbeddingAdaptor(WrappedEmbedding.from_fasttext('models/fasttext-wordswapped.bin', **kwargs))
    elif debias == 'bolukbasi':
        return EmbeddingAdaptor(load_bolukbasi_model())
    else:
        raise ValueError('unrecognized model/debiasing pair: {}, {}'.format(
            model, debias
        ))


def load_subspace_pairs(subspace_pairs):
    """Load the pairs used for gender subspace definition.

    Returns:
        Dict[str, List[Tuple[str, str]]]: Dictionary of gender word pairs. The
            value is a list of (male, female) words; the key is the name of
            that list.
    """
    filepairs = []
    for direction_file in list_files(DIRECTIONS_PATH):
        with open(direction_file) as fd:
            filepairs.extend(
                (
                    os.path.basename(direction_file),
                    tuple(line.strip().split())
                )
                for line in fd
            )
    if subspace_pairs == 'pair':
        keyfunc = (lambda filepair: '-'.join(filepair[1]))
    elif subspace_pairs == 'group':
        keyfunc = (lambda filepair: filepair[0])
    elif subspace_pairs == 'all':
        keyfunc = (lambda filepair: 'all')
    else:
        raise ValueError('unrecognized subspace pair parameter: {}'.format(
            subspace_pairs
        ))
    return {
        key: [filepair[1] for filepair in group] for key, group
        in groupby(filepairs, key=keyfunc)
    }


def run_experiment(parameters):
    embedding = load_debaised_model(parameters.model_algo, parameters.debiasing)
    # FIXME
    # opp, accuracy = run_model_evaluation(model_file)
    subspace_pair_groups = load_subspace_pairs(parameters.subspace_pairs)
    for group_name, subspace_pair_group in subspace_pair_groups.items():
        male_words, female_words = zip(*subspace_pair_group)
        if parameters.subspace_algo == 'mean':
            direction = define_gender_direction_mean(embedding, male_words, female_words)
        else:
            direction = define_gender_direction_pca(embedding, male_words, female_words)
        if direction is None:
            bias = 'N/A'
        else:
            words = load_bias_words(parameters.bias_words)
            bias = calculate_model_bias(embedding, direction, words)
        print(
            parameters.corpus,
            parameters.model_algo,
            parameters.debiasing,
            parameters.subspace_pairs,
            group_name,
            parameters.subspace_algo,
            parameters.bias_words,
            bias,
        )


def build_all_fasttext_models(model_type='skipgram'):
    if model_type not in ['skipgram', 'cbow']:
        raise ValueError('model_type must be "skipgram" or "cbow" but got "' + str(model_type) + '"')
    for corpus_file in list_files(CORPORA_PATH):
        if not corpus_file.endswith('-swapped'):
            create_pronoun_swapped_corpus(corpus_file, 'swap-pairs/pronouns')
    for corpus_file in list_files(CORPORA_PATH):
        model_stub = os.path.join(
            MODELS_PATH,
            os.path.basename(corpus_file) + '.' + model_type,
        )
        if not os.path.exists(model_stub + '.bin'):
            subprocess.run(
                args=['fasttext', 'skipgram', '-input', corpus_file, '-output', model_stub]
            )


def main():
    """Entry point for the project."""
    pspace = PermutationSpace(
        [
            'corpus',
            'model_algo',
            'debiasing',
            'subspace_pairs',
            'subspace_algo',
            'bias_words',
        ],
        corpus=['wikipedia'],
        model_algo=['fasttext'],
        debiasing=['none', 'wordswap', 'bolukbasi', 'all_pairs'],
        subspace_pairs=['pair', 'group', 'all'],
        subspace_algo=['mean', 'pca'],
        bias_words=['occupations', 'adjectives'],
    ).filter(
        lambda debiasing:
            debiasing == 'all_pairs'
    )
    print(' '.join(pspace.order))
    for parameter in pspace:
        run_experiment(parameter)


if __name__ == '__main__':
    main()
