from gensim.corpora import WikiCorpus
import sys

'''
Adapted from Matthew Mayo, KDnuggts
https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
'''


def make_corpus(infile, outfile):
    output = open(outfile, 'w')
    wiki = WikiCorpus(infile)
    i = 0
    # "text" is actually each individual article
    for text in wiki.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        i += 1
        if (i % 100 == 0):
            print('Processed ' + str(i) + ' articles so far.')
    output.close()
    print('Processing complete! Yippee!')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 process_data.py <wikipedia_dump_file> <processed_text_file>')
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    make_corpus(infile, outfile)