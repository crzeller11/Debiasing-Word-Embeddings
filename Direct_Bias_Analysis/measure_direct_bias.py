# ALL IMPORTS
from gensim.test.utils import common_texts
from gensim.models import FastText
import numpy as np
from sklearn.decomposition import PCA

MODEL_FILEPATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/fastText/model.bin'
MODEL = FastText.load_fasttext_format(MODEL_FILEPATH)

def get_vecs(words):
    vectors = []
    for word in words:
        vec = MODEL[word]
        vectors.append(vec)
    return vectors

def gender_direction(female_wrds, male_wrds, model):
    # get the vectors for all the female words and male words
    # then, subtract all male from female (or all female from male)
    # save all those vectors to a set of vectors, and perform PCA on them
    female_vectors = get_vecs(female_wrds)
    male_vectors = get_vecs(male_wrds)
    subtraction = []
    for i in range(len(female_vectors)):
        subtraction.append(np.subtract(female_vectors[i], male_vectors[i]))
    pca = PCA(n_components=2) # still not sure what this component argument is expecting. The dimension of the vectors?
    # the number of vectors? The output dimension?
    pca.fit(subtraction)
    print(pca.singular_values_)

def direct_bias(gender_direction):
    # TODO: combine all elements of DirectBias statistic here
    pass

def main():
    # load the model from FastText using Gensim
    female_words = ['she', 'her', 'hers']
    male_words = ['he', 'him', 'his']
    loaded_model = FastText.load_fasttext_format(MODEL_FILEPATH)
    gender_direction(male_words, female_words, loaded_model)


if __name__ == '__main__':
    main()