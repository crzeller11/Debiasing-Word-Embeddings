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
        if (i % 10000 == 0):
            print('Processed ' + str(i) + ' articles so far.')
    output.close()
    print('Processing complete! Yippee!')

# helper method if you don't want to use command line
def run_me():
    infile = 'enwiki-latest-pages-articles21.xml-p21222161p22722161'
    outfile = 'output_corpus.txt'
    make_corpus(infile, outfile)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 process_data.py <wikipedia_dump_file> <processed_text_file>')
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    make_corpus(infile, outfile)
