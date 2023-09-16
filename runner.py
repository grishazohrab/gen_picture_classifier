from typing import Dict, List

import torch
from transformers import BertTokenizer

from bert_classification import BertClassifier
from data_processing import remove_name
from config import MODEL_PATH, CONVERSATION_THRESHOLD


class ModelRunner:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        self.model = BertClassifier()
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()

        self.model.to(self.device)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def process_data(self, input_data: List[Dict[str, str]]):

        n = len(input_data)
        text = ""
        for i in range(max(0, n - CONVERSATION_THRESHOLD), n):
            text += f" [{input_data[i]['from']}] " + remove_name(input_data[i]["value"])

        return self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    def run(self, input_data: List[Dict[str, str]]):
        tokens = self.process_data(input_data)

        mask = tokens['attention_mask'].to(self.device)
        input_id = tokens['input_ids'].squeeze(1).to(self.device)

        output = self.model(input_id, mask)

        return output.argmax(dim=1)
