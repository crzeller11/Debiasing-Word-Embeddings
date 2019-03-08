"""
Adapted from https://github.com/tolga-b/debiaswe

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016

Adapted by Chloe Zeller, February 2019
"""

from __future__ import print_function, division
import json
import numpy as np
import sys
if sys.version_info[0] < 3:
    import io
    open = io.open

from .we import WordEmbedding, drop, doPCA


def debias(E, gender_specific_words, definitional, equalize):
    # TODO: should we use our own version of PCA or theirs?
    gender_direction = doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    #print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()

def main():
    embedding_filename = 'models/NETWORK2MODEL1.bin'
    definitional_filename = 'data/definitional_pairs.json'
    gendered_words_filename = 'data/gender_specific_full.json'
    equalize_filename = 'data/equalize_pairs.json'
    debiased_filename = 'models/bolukbasi_hard_biased_model.bin'
    with open(definitional_filename, "r") as f:
        defs = json.load(f)
    with open(equalize_filename, "r") as f:
        equalize_pairs = json.load(f)
    with open(gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    E = WordEmbedding(embedding_filename)
    debias(E, gender_specific_words, defs, equalize_pairs)
    '''
    Not sure if I understand why we wouldn't save another copy of this model. If we don't, then we won't have an
    embedding to compared to the debiased version
    '''
    if embedding_filename[-4:] == debiased_filename[-4:] == ".bin":
        E.save_w2v(debiased_filename)
    else:
        E.save(debiased_filename)


if __name__ == "__main__":
    main()
