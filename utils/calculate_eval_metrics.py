## okay this broke, so think about how to mitigate.
from rouge_score import rouge_scorer
import evaluate
import numpy as np
from sklearn.metrics import f1_score

def calculate_f1(preds, refs):
    "Calculating F1 score between prediction and reference"

    f1_out = f1_score(y_true=refs, y_pred=preds, average="macro")

    return {'f1': f1_out}

def calculate_str_contains(preds, refs):

    pred_contains = (refs.isin(preds)).mean()

    return {'str_contains': pred_contains}

def calculate_bleu(preds, refs):
	"""n-gram matching that treats generation as bag of words.
	Combines score from different n-gram sizes. High score for precise but incomplete output
	"""

	scorer = evaluate.load("bleu")
	scores = scorer.compute(predictions=preds,
                    references=refs,
                    max_order=4
					)
	
	return {'bleu': scores['bleu']}
	
	
def calculate_rouge(preds, refs):
	"""n-gram matching that assigns high score for good overlap but with unnecessary info"""
	
	scorer = evaluate.load("rouge")
	scores = scorer.compute(predictions=preds,
                    references=refs)
	return {'rouge1': scores['rouge1'], 'rouge2': scores['rouge2'], 'rougeL': scores['rougeL'], 'rougeLsum': scores['rougeLsum']}


def calculate_exact_string_match(preds, refs):
	
	accuracy = (preds==refs).sum()/len(preds)

	return {'exact_string_match_accuracy': accuracy}


def calculate_sequence_accuracy(preds, refs):
	"""For task like POS tagging, checks if each token at timestep t was predicted correctly.
	"""

	acc = []
	for (p,r) in zip(preds, refs):
		p_toks = p.split(' ')
		r_toks = r.split(' ')
		p_toks = [i.lower() for i in p_toks]
		r_toks = [i.lower() for i in r_toks]
		
		if len(p_toks) < len(r_toks):
			p_toks += [-100] * (len(r_toks) - len(p_toks))
		elif len(p_toks) > len(r_toks):
			p_toks = p_toks[:len(r_toks)]

		count = 0
		for idx in range(len(r_toks)):
			count += (r_toks[idx] == p_toks[idx]) * 1
			
		acc.append(count / len(r_toks))

	return {"sequence_accuracy": np.mean(acc)}
