from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 9)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False, )
        # dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(pooled_output)
        final_layer = self.Sigmoid(linear_output)

        return final_layer
