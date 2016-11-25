'''
Where we take care of word embedding
'''

import numpy as np
import gensim
import pickle


class WordEmbedding:


	def __init__(self):
		"""
		Initialize the WordEmbedding instance by loading Google pretrained word2vec embeddings
		"""
		print('loading words embeddings')
		self.model = gensim.models.Word2Vec.load_word2vec_format('../../model/GoogleNews-vectors-negative300.bin', binary=True)
		self.model_vocab = self.model.vocab 
		print "Google word2vec pre-trained embeddings loaded\n"		
		self.num_features = 300 # 300 if using word2vec (word2vec embedding dimension)
		self.new_words = {}

	def embed(self, word):
		'''
		embed words using model if words in model, or generating random embedding if word not in model
		'''
		if word not in self.model_vocab:
			with open('../../model/unk_words.model', 'rb') as handle:
				unknown_words_dict = pickle.load(handle)
			if word not in unknown_words_dict:
				word_vec = np.random.uniform(-1, 1, 300)
				print word, word_vec[:10]
				unknown_words_dict[word] =  word_vec
				print "word embedding added in dict "
				with open('../../model/unk_words.model', 'wb') as handle: 
					pickle.dump(unknown_words_dict,handle)
				print "dict saved back as model"
			else:
				word_vec = unknown_words_dict[word]
		else:
			word_vec = self.model[word]

		return word_vec



