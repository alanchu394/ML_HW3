import pandas as pd
import torch
import unittest

from boolq import BoolQDataset
from transformers import RobertaTokenizerFast


class TestBoolQDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4
        self.boolq_dataset = BoolQDataset(
            self.dataset, self.tokenizer, self.max_seq_len
        )

    def test_len(self):
        ## TODO: Test that the length of self.boolq_dataset is correct.
        ## len(self.boolq_dataset) should equal len(self.dataset).
        flag =  (len(self.boolq_dataset) == len(self.dataset))
        print('test_len: ' + str(flag))
        return flag

    def test_item(self):
        ## TODO: Test that, for each element of self.boolq_dataset, 
        ## the output of __getitem__ (accessible via self.boolq_dataset[idx])
        ## has the correct keys, value dimensions, and value types.
        ## Each item should have keys ["input_ids", "attention_mask", "labels"].
        ## The input_ids and attention_mask values should both have length self.max_seq_len
        ## and type torch.long. The labels value should be a single numeric value.
        flag = True
        for element in self.boolq_dataset:
            input_ids = element['input_ids']
            attention_mask = element['attention_mask']
            labels = element['labels']
            if(len(input_ids) != self.max_seq_len or not (input_ids.dtype == torch.long)):
                flag = False
                break
            if(len(attention_mask) != self.max_seq_len or not (input_ids.dtype == torch.long)):
                flag = False
                break
            if(labels != 0 and labels != 1):
                flag = False
                break

        print('test_item: ' + str(flag))
        
        return flag


if __name__ == "__main__":
    unittest.main()
