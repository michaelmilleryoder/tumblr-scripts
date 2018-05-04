from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
import pdb

embs_path = '/usr2/kmaki/tumblr/top100posts-131M_fasttext-embeddings_3-6.bin.vec'
wembed = KeyedVectors.load_word2vec_format(embs_path)
pdb.set_trace()
