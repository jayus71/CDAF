import torch
from torch import nn
import numpy as np
from tqdm import tqdm

class MyModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=10):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding_net = nn.Linear(self.input_dim, self.hidden_dim)
        self.predictor = nn.Linear(self.hidden_dim, self.output_dim)
        self.m = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden_feature = self.relu(self.embedding_net(x))
        out = self.m(self.predictor(hidden_feature))
        return hidden_feature, out

source_expert = MyModel(2,2,10)
torch.save(source_expert.state_dict(), 'source_model.pkl')
source_model = MyModel(2,2,10)

# load source expert
source_model_dict = torch.load('source_model.pkl')
source_model.load_state_dict(source_model_dict)
print(source_model_dict)
print(source_model)

