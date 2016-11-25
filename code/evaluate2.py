import pandas as pd
import numpy as np
import similarity2
from scipy.spatial.distance import cdist

class Evaluator2():

	def softmax(self, x):
		"""Compute softmax values for each sets of scores in x."""
		return np.exp(x) / np.sum(np.exp(x), axis=0)


	def pseudo_distance(self, x, y):
		return np.linalg.norm(x - np.dot(x, y)*y/np.linalg.norm(y))


	def find_best_mult_match(self, hardcoded_question_set, user_question, idf_dictionnary, similarity):
		"""
		find best matches from the hardcoded_question_set to a user question using the pseudo_distance
		(fallback if more than 3 matches found)
		"""
		dim = hardcoded_question_set['hardcoded_question'].shape[0]
		scores = []
		for question in hardcoded_question_set['hardcoded_question']:
			scores.append(similarity.find_similarity(user_question, question, idf_dictionnary))
		
		s = self.softmax(scores)
		
		# Keep track of questions indexes when sorting s in descending order
		index =range(len(s))
		s_ind = zip(s, index)
		s_ind.sort(reverse=True)
		new_index = [ind for value, ind in s_ind]

		p = ['p%i' % i for i in xrange(1,dim+1)]
		df = pd.DataFrame(s.tolist())
		df = df.transpose()
		df.columns = p
		df = df.apply(lambda s: s.sort_values(ascending=False).values, axis=1)

		M = np.transpose([map(lambda x: 1./x if x >= i else 0, xrange(1, dim+1)) for i in xrange(1, dim+1)])

		df["pds"] = pd.DataFrame(cdist(df[p], M, self.pseudo_distance)).apply(pd.Series.argmin, axis=1)+1

		# number of matches to return
		n = df["pds"].iloc[0]

		if n > 3:
			best_match = ['fallback']
			best_score = [0]
		else:
			best_match = hardcoded_question_set['hardcoded_question'].iloc[new_index[:n]].tolist()
			best_score = sorted(s, reverse=True)[:n]

		return best_match, best_score


	def find_best_match(self, hardcoded_question_set, user_question, idf_dictionnary, similarity):
		"""
		find the best match from the hardcoded_question_set to a user question. 
		"""
		scores = []
		for question in hardcoded_question_set['hardcoded_question']:
			scores.append(similarity.find_similarity(user_question, question, idf_dictionnary))
		best_score = max(scores)
		best_match = hardcoded_question_set['hardcoded_question'][scores.index(best_score)]
		return [best_match], [best_score]


	def predict(self, test_set, hardcoded_question_set, idf_dictionnary, similarity, mult_answ = True):
		"""
		where we find the best matches to each user question in the test set from the hardcoded question set
		"""
		predictions = []
		scores = []
		for user_question in test_set['user_question']:
			if mult_answ == False:
				# perform cosine similarity and find best match
				best_match, best_score = self.find_best_match(hardcoded_question_set, user_question, idf_dictionnary, similarity)
				predictions.append(best_match)
				scores.append(best_score)
			elif mult_answ == True:
				# perform cosine similarity and find best match
				best_match, best_score = self.find_best_mult_match(hardcoded_question_set, user_question, idf_dictionnary, similarity)
				predictions.append(best_match)
				scores.append(best_score)
		return predictions, scores


	def merge_test_set_with_predictions(self, test_set, predictions, scores):
		'''
		returns a data frame with columns 'hardcoded_questions', 'user_question', 'prediction' and 'correct' (0 or 1)
		'''
		# prediction correct = 1, incorrect = 0						
		correct = []
		for i in range(len(predictions)) :
			if test_set['hardcoded_question'][i] in predictions[i]:
				correct.append(1)
			else :
				correct.append(0)

		pred = pd.DataFrame({'prediction': predictions})
		scor = pd.DataFrame({'scores': scores})
		corr = pd.DataFrame({'correct': correct})
		result = pd.concat([test_set, pred, scor, corr], axis=1, join_axes=[test_set.index])
		return result


	def compute_accuracy(self, results):
		correct_answers = results['correct'].sum() # 0 or 1 for correct answer is at position 3 in results
		accuracy = correct_answers/float(results.shape[0])
		return accuracy

