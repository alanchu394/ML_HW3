
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from transformers import RobertaConfig, RobertaModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1) 

    f1 = 0
    accuracy = 0
    precision = 0
    recall = 0
    precision, recall, fbeta, support = precision_recall_fscore_support(labels, preds,average="binary")
    accuracy = accuracy_score(labels, preds)
    f1 = 2 * (precision * recall) / (precision + recall)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.

    return {'f1' : f1, "accuracy" : accuracy, "precision": precision, "recall" : recall}

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.

    return RobertaForSequenceClassification.from_pretrained("roberta-base")
