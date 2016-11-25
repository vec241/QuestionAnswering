'''
Where we load + process data
'''

import re
from sklearn import feature_extraction
import pandas as pd

class DataProcessor:

	def clean_str(self,string):
		"""
		Tokenization/string cleaning strings
		Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
		"""
		string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " \'s", string)
		string = re.sub(r"\'ve", " \'ve", string)
		string = re.sub(r"n\'t", " n\'t", string)
		string = re.sub(r"\'re", " \'re", string)
		string = re.sub(r"\'d", " \'d", string)
		string = re.sub(r"\'ll", " \'ll", string)
		string = re.sub(r",", " , ", string)
		string = re.sub(r"!", " ! ", string)
		string = re.sub(r"\(", " \( ", string)
		string = re.sub(r"\)", " \) ", string)
		string = re.sub(r"\?", " \? ", string)
		string = re.sub(r"\s{2,}", " ", string)
		return string.strip('\n').strip().lower()


	def load_data(self, path, file_name):		
		df = pd.read_csv(path + file_name, sep='\t', header=None)
		df.columns = ['hardcoded_question']
		for i in df.index:
			df['hardcoded_question'][i] = self.clean_str(df['hardcoded_question'][i])
		return df

	def load_datasetbuilding_data(self, path, file_name):
		df = pd.read_csv(path + file_name, sep='\t', header=None)
		df.columns = ['hardcoded_question']
		# Only load train questions that are not too long
		mask = (df['hardcoded_question'].str.split().apply(len) < 15)
		df = df.loc[mask]
		print 'df string < 15', df
		for i in df.index:
			df['hardcoded_question'][i] = self.clean_str(df['hardcoded_question'][i])
		return df

	def load_test_data(self, path, file_name):
		df = pd.read_csv(path + file_name, sep='\t', header=None)
		df.columns = ['hardcoded_question', 'user_question']
		for i in df.index:
			df['hardcoded_question'][i] = self.clean_str(df['hardcoded_question'][i])
			df['user_question'][i] = self.clean_str(df['user_question'][i])
		return df 

	def build_sklearn_idf_dic(self, df):
		'''
		build idf dictionnary using sklearn
		'''
		model = feature_extraction.text.TfidfVectorizer()
		model.fit(df['hardcoded_question'])
		vocab = model.vocabulary_ # returns doc vocab as dictionnary 
		idf = model.idf_ # returns array of idfs where index corresponds to value in vocab
		idf_dictionnary = {word : idf[vocab[word]] for word in vocab.iterkeys()}
		return idf_dictionnary 










