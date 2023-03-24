from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(768, 384)
        self.Relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(384, 192)
        self.Relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(192, 9)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        # dropout_output = self.dropout(pooled_output)
        linear_1_output = self.linear_1(pooled_output)
        relu_1_layer = self.Relu_1(linear_1_output)
        linear_2_output = self.linear_2(relu_1_layer)
        relu_2_layer = self.Relu_2(linear_2_output)
        linear_3_output = self.linear_3(relu_2_layer)
        final_layer = self.Sigmoid(linear_3_output)

        return final_layer
