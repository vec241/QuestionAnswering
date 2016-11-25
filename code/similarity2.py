'''
Available POS : 
['PRP$','VBG','VBD','VBN','VBP','WDT','JJ','WP','VBZ','DT','RP','NN','FW','POS',
'TO','LS','RB','NNS','NNP','VB','WRB','CC','PDT','RBS','RBR','CD','PRP','EX','IN',
'WP$','MD','NNPS','JJS','JJR','SYM','UH']
'''

import numpy as np
import numpy as np
import nltk
import inflect
import word_embedding

class Similarity2():
	'''
	Trying similarity method #2
	Check ipynb similarity_2 for more details
	'''

	def __init__(self, keep_pos=['NN', 'NNS', 'JJ', 'VB']):
		self.wordEmbeder = word_embedding.WordEmbedding()
		# POS to keep
		self.keep_pos = keep_pos


	def keep_nouns(self, question):
		'''
		From a sentence, return list of tuples [(word, POS)] if POS in self.keep_pos (initially nouns)
		'''
		question_tokens = nltk.word_tokenize(question)
		question_tagged = nltk.pos_tag(question_tokens)
		question_tagged_nouns = [question_tagged[i] 
								for i in range(len(question_tagged)) 
								if question_tagged[i][1] 
								in self.keep_pos]
		return question_tagged_nouns

	def plural_to_singular(self, tagged_nouns):
		'''
		Putting all nouns to singular form (because similarity(A,A') > similarity(A,As'))
		'''
		p = inflect.engine()

		question_nouns = []
		for nouns in tagged_nouns:
			if nouns[1] != 'NNS':
				question_nouns.append(nouns[0]) 
			else:
				if p.singular_noun(nouns[0]) != False:
					question_nouns.append(p.singular_noun(nouns[0]))
		return question_nouns

	def non_linear(self, x, non_linearity='none'):
		'''
		add a non linearity such as it penalizes values close to 0 (push them even more towards 0)
		'exponential' and 'sigmoid' only refer to the shape of the non linearities, but the functions
		are adapted to fit on [0,1] --> [0,1]
		'''
		if non_linearity=='none':
			y = x
		if non_linearity=='exponential':
			k = 0.5
			y = k*x/(k-x+1)
		if non_linearity=='sigmoid':
			m = 30
			y = 1/(1+ np.exp(-m*(x-0.5)))
		return y


	def word_cosine_sim(self, word1, word2):
		'''
		find pairwise cosine similarity between word1 and word2 using model embeddings
		'''
		w1, w2 = self.wordEmbeder.embed(word1), self.wordEmbeder.embed(word2)
		cosine_similarity = np.dot(w1, w2)/(np.linalg.norm(w1) * np.linalg.norm(w2))
		return cosine_similarity


	def find_sim(self, question_nouns_1, question_nouns_2, idf_dictionnary):
		'''
		From word pairwise cosine similarities, return similarity between 2 sentences, following method
		described in ipynb Similarity_method_2
		'''

		max_len = max(len(question_nouns_1), len(question_nouns_2))
		if len(question_nouns_1) == max_len:
			seq1, seq2 = question_nouns_1, question_nouns_2
		else:
			seq1, seq2 = question_nouns_2, question_nouns_1

		similarity = {}
		for word1 in seq1:
			if word1 in idf_dictionnary.keys():
				idf_w1 = idf_dictionnary[word1]
			else:
				idf_w1 = 5

			similarities = {}
			for word2 in seq2:
				similarities[word2] = self.word_cosine_sim(word1, word2)

			try:
				sim_max = max(similarities.values())
				word2_sim_max = similarities.keys()[similarities.values().index(sim_max)]
				if word2_sim_max in idf_dictionnary.keys():
					idf_w2 = idf_dictionnary[word2_sim_max]
				else:
					idf_w2 = 5
				# apply non linearity to penalize low similarities
				# Chose between none, exponential, sigmoid
				similarity[(word1, word2_sim_max)] = self.non_linear(sim_max, non_linearity='sigmoid') * (idf_w1 + idf_w2)/2.0
			except:
				continue

		similarity_score = 0

		if len(similarity.values()) > 0 :
			similarity_score = sum(similarity.values())/float(len(similarity.values()))

		return similarity_score


	def find_similarity(self, question1, question2, idf_dictionnary):
		question_tagged_nouns_1 = self.keep_nouns(question1)
		question_tagged_nouns_2 = self.keep_nouns(question2)
		question_nouns_1 = self.plural_to_singular(question_tagged_nouns_1)
		question_nouns_2 = self.plural_to_singular(question_tagged_nouns_2)
		similarity_score = self.find_sim(question_nouns_1, question_nouns_2, idf_dictionnary)
		return similarity_score

