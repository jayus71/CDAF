import mindspore
from mindspore import nn
from mindspore import ops
import numpy as np
from tqdm import tqdm

class MyModel(nn.Cell):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=10):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # change "Linear" to "Dense"
        self.embedding_net = nn.Dense(self.input_dim, self.hidden_dim)
        self.predictor = nn.Dense(self.hidden_dim, self.output_dim)
        self.m = nn.LogSoftmax(axis=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden_feature = self.relu(self.embedding_net(x))
        out = self.m(self.predictor(hidden_feature))
        return hidden_feature, out
    
source_expert = MyModel(2,2,10)
for param in source_expert.get_parameters():
    print(param.name, param.data.asnumpy())

mindspore.save_checkpoint(source_expert, "source_model.ckpt")

source_model = MyModel(2,2,10)
source_model_dict = mindspore.load_checkpoint("source_model.ckpt")
params_not_load,_ = mindspore.load_param_into_net(source_model, source_model_dict)
for param in source_model.get_parameters():
    print(param.name, param.data.asnumpy())


# print(source_model_dict)
# print(source_model)