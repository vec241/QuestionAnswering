'''
Main with automated evaluation on a test set using similarity method #2
'''

import similarity2
import data_processing
import evaluate2

# upload data
print "loading data...\n"
print "loading hardcoded_question_set...\n"
dataProcessor = data_processing.DataProcessor()
path_train = '../data/'
file_name_train = 'hardcoded_questions.txt' 
hardcoded_question_set = dataProcessor.load_data(path_train, file_name_train)
print "loading test set...\n"
path_test = '../data/'
file_name_test = 'test_set.tsv' #'test_set2.tsv'
test_set = dataProcessor.load_test_data(path_test, file_name_test)#.iloc[[6]]
print "loading idf data...\n"
path_idf = '../data/'
file_name_idf = 'idf_train.txt'
idf_set = dataProcessor.load_data(path_idf, file_name_idf)


# idf
print "building idf dictionnary...\n"
idf_dictionnary = dataProcessor.build_sklearn_idf_dic(idf_set)


# predict
similarity = similarity2.Similarity2()
evaluator = evaluate2.Evaluator2()
print "predicting...\n"
predictions, scores = evaluator.predict(test_set, hardcoded_question_set, idf_dictionnary, similarity, mult_answ = True)
print "evaluation starts...\n"
results = evaluator.merge_test_set_with_predictions(test_set, predictions, scores)
accuracy = evaluator.compute_accuracy(results)
print results
print "accuracy:"
print accuracy
print "evaluations ends...\n"


# write results in a tsv file
print "writting results in a text file...\n"
with open ('../Data/results.tsv', 'w') as test_set:
	# First, write hardcoded questions (and modify manually user questions)
	for hardcoded_question, user_question, prediction, score, correct in zip(
			results['hardcoded_question'], 
			results['user_question'],
			results['prediction'],
			results['scores'],
			results['correct']):
		test_set.write("{}\t{}\t{}\t{}\t{}\n".format(hardcoded_question, user_question, prediction, score, correct))

