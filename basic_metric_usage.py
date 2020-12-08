from bert_score import score as bert_score
from nltk.translate import bleu_score as bleu




'''
https://github.com/Maluuba/nlg-eval
'''
candidate = "This is my favorite!"
answer = ["I love this", "I'm loving it!"]

from nlgeval import NLGEval
nlgeval = NLGEval(metrics_to_omit=["SkipThoughtCS"])  # loads the models
metrics_dict = nlgeval.compute_individual_metrics(answer, candidate)
print(metrics_dict)

'''
How to use bert score.
'''
candidate = ['I love this', 'I like this']
answer = ["I don't like this", "I'm loving it!"]

result = bert_score(candidate, answer, rescale_with_baseline=True, lang="en")[-1]
print(result)
