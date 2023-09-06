####
# An example of ulitizing CDAF to train a model on source domain and target domain.
# You can replace the model with your own model or mainstream recommendation models.

# a demo of transfering pytorch_impl to mindspore_impl
# reference: https://bbs.huaweicloud.com/forum/thread-196696-1-1.html
####

# import torch
# from torch import nn
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
        # change the param "dim" to "axis"
        self.m = nn.LogSoftmax(axis=-1)
        self.relu = nn.ReLU()

    def construct(self, x):
        hidden_feature = self.relu(self.embedding_net(x))
        out = self.m(self.predictor(hidden_feature))
        return hidden_feature, out

class DualModel(nn.Cell):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=10):
        super(DualModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding_net = nn.Dense(self.input_dim, self.hidden_dim)
        self.predictor1 = nn.Dense(self.hidden_dim, self.output_dim)
        self.predictor2 = nn.Dense(self.hidden_dim, self.output_dim)
        self.m = nn.LogSoftmax(axis=-1)
        self.relu = nn.ReLU()

    def construct(self, x):
        hidden_feature = self.relu(self.embedding_net(x))
        out1 = self.m(self.predictor1(hidden_feature))
        out2 = self.m(self.predictor2(hidden_feature))
        return hidden_feature, out1, out2

def load_data():
    moon_data = np.load('moon_data.npz')
    # x_s: source domain data
    x_s = moon_data['x_s']
    # y_s: source domain label
    y_s = moon_data['y_s']
    # x_t: target domain data
    x_t = moon_data['x_t']
    # y_t: target domain label
    y_t = moon_data['y_t']
    return x_s, y_s, x_t, y_t

def sort_rows(matrix, num_rows):
    matrix_T = matrix.transpose(0, 1)
    # change "torch.topk" to "ops.topk", maybe means operations
    sorted_matrix_T, _ = ops.topk(matrix_T, num_rows, dim=1)
    # change "transpose" to "swapaxes"
    return sorted_matrix_T.swapaxes(0, 1)


def wasserstein_discrepancy(p1, p2):
    s = p1.shape
    if p1.shape[1] > 1:
        # For data more than one-dimensional, perform multiple random projection to 1-D
        # change "torch.randn" to "ops.randn",etc
        proj = ops.randn(p1.shape[1], 128)
        proj *= ops.rsqrt(ops.sum(proj**2, dim=0, keepdim=True))
        p1 = ops.matmul(p1, proj)
        p2 = ops.matmul(p2, proj)
    p1 = sort_rows(p1, s[0])
    p2 = sort_rows(p2, s[0])
    wdist = ops.mean((p1 - p2)**2)
    return ops.mean(wdist)


def discrepancy_l1(out1, out2):
    return ops.mean(ops.abs(out1 - out2))

def discrepancy_l2(out1, out2):
    return ops.mean(ops.square(out1 - out2))

# add the mindspore's train functions

def pre_train_source_mindspore():
    source_expert = MyModel(2,2,10)
    source_expert_optimizer = nn.Adam(source_expert.parameters(), learning_rate=0.001)

    x_s, y_s, _, _ = load_data()
    x_s = mindspore.Tensor(x_s).float()
    y_s = mindspore.Tensor(y_s).long().reshape(-1)

    criterion = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    def net_forward(input,target):
        logit = source_expert(input)
        loss = criterion(logit,target)
        return loss,logit   
    
    net_backward = mindspore.value_and_grad(net_forward, None, source_expert_optimizer.parameters, has_aux=True)

    def train_step(input, target):
        (loss, _), grad = net_backward(input, target)
        source_expert_optimizer(grad)
        return loss

    for epoch in tqdm(range(1000)):
        loss = float(train_step(x_s, y_s).asnumpy())
        print('Epoch: {}, Loss: {}'.format(epoch, loss))

    mindspore.save_checkpoint(source_expert, "source_model.ckpt")



def pre_train_source():
    source_expert = MyModel(2,2,10)

    # change "torch.optim.Adam" to "nn.Adam" and "lr" to "learning_rate" 
    source_expert_optimizer = nn.Adam(source_expert.parameters(), learning_rate=0.001)

    x_s, y_s, _, _ = load_data()

    # change "mindspore.Tensor().float" to "mindspore.Tensor().float()", 
    # reference: "https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor"
    x_s = mindspore.Tensor(x_s).float()
    y_s = mindspore.Tensor(y_s).long().reshape(-1)

    criterion = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    for epoch in tqdm(range(1000)):
        source_expert_optimizer.zero_grad()
        _, source_out = source_expert(x_s)
        loss = criterion(source_out, y_s)
        loss.backward()
        source_expert_optimizer.step()
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
    # Save the model
    # torch.save(source_expert.state_dict(), 'source_model.pkl')
    mindspore.save_checkpoint(source_expert, "source_model.ckpt")

def train():
    # creat model
    source_model = MyModel(2,2,10)
    target_model = DualModel(2,2,10)
    
    # load source expert
    #----------save in torch-------------
    # source_model_dict = torch.load('source_model.pkl')
    # source_model.load_state_dict(source_model_dict)
    #------------end save--------------
    # change the way to save model,
    # reference: https://mindspore.cn/tutorials/zh-CN/master/beginner/save_load.html
    source_model_dict = mindspore.load_checkpoint("source_model.ckpt")
    params_not_load,_ = mindspore.load_param_into_net(source_model, source_model_dict)

    
    # initialize target model with source expert
    # 
    # target_model_dict = target_model.state_dict()
    target_model_dict = target_model.parameters_dict()
    pretrained_dict = {}
    for k, _ in target_model_dict.items():
        if k in source_model_dict:
            pretrained_dict[k] = source_model_dict[k]
        elif 'predictor' in k:
            pretrained_dict[k] = source_model_dict['predictor.'+k.split('.')[1]]

    # update the params with the source
    target_model_dict.update(pretrained_dict)
    # target_model.load_state_dict(target_model_dict)
    params_not_load,_ = mindspore.load_param_into_net(target_model,target_model_dict)
    
    # optimizer
    target_optimizer = nn.Adam(target_model.parameters(), learning_rate=0.001)

    # load data
    x_s, y_s, x_t, y_t = load_data()

    x_s = mindspore.Tensor(x_s).float()
    y_s = mindspore.Tensor(y_s).long().reshape(-1)
    x_t = mindspore.Tensor(x_t).float()
    y_t = mindspore.Tensor(y_t).long().reshape(-1)

    criterion = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    source_model.eval()
    target_model.train()
    
    for p in source_model.get_parameters():
        p.requires_grad = False
    
    for p in target_model.get_parameters():
        p.requires_grad = True

        
    for epoch in tqdm(range(1000)):
        target_optimizer.zero_grad()
        source_feature, _ = source_model(x_s)
        x_t = ops.cat((x_s,x_t),0)
        joint_feature, t_source_out, t_target_out = target_model(x_t)
        
        # wasserstein discrepancy loss (L_j in Eq.(5))
        feat_loss = wasserstein_discrepancy(source_feature,joint_feature[:x_s.shape[0],:])
        feat_loss += wasserstein_discrepancy(source_feature,joint_feature[x_s.shape[0]:,:])

        # L_t^s in Eq.(7)
        predict_loss_t_source = criterion(t_source_out[:x_s.shape[0],:],y_s)
        # L_t^t in Eq.(8)
        predict_loss_t_target = criterion(t_target_out[x_s.shape[0]:,:],y_t)

        # L_d in Eq.(9)
        l1_loss = discrepancy_l1(t_source_out[:x_s.shape[0],:],t_target_out[x_s.shape[0]:,:])

        # total loss
        loss = feat_loss + predict_loss_t_source + predict_loss_t_target + l1_loss
                
        loss.backward()
        target_optimizer.step()
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

if __name__ == '__main__':
    # pre_train_source()
    train()