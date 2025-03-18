import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.ensemble import RandomForestRegressor

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hs=50, num_layers=1, output_size=1, sequence_length=None, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hs = int(hs)
        self.num_layers =int(num_layers)
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hs, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hs, output_size)

    def forward(self, x):
        batch_size = x.size(0)  # 动态计算batch_size
        h0 = torch.zeros(self.num_layers, batch_size, self.hs).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hs).to(x.device)
        
        # LSTM的输出
        output, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # 取最后一个时间步的隐藏状态
        last_hidden_state = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        out = self.fc(last_hidden_state)
        return out

# 随机森林图计算
def rf_graph(x):   
    G = np.zeros((len(x), len(x))) 
    for i in range(len(x)):      
        G[i, np.where(x == x[i])[0]] = 1
    nodes = Counter(x)
    nodes_num = np.array([nodes[i] for i in x])       
    return G, G / nodes_num.reshape(-1, 1)   

# 获取随机森林的权重（通过叶节点）
def get_rfweight3(rf, x): 
    n = x.shape[0]
    leaf = rf.apply(x)
    ntrees = leaf.shape[1]
    G_unnorm = np.zeros((n, n))
    G_norm = np.zeros((n, n))  
    for i in range(ntrees):     
        tmp1, tmp2 = rf_graph(leaf[:, i])
        G_unnorm += tmp1
        G_norm += tmp2    
    return G_unnorm / ntrees, G_norm / ntrees

# 定义BWL模型
class BWL():
    def __init__(self, num_layers, dropout, hs, device='cuda', quantile=0.05):
        self.hs = int(hs)
        self.device = device
        self.quantile = quantile
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)

    def fit(self, x, y, weight,tau,bs,lr,tol, n_iter,d=False,  verbose=True):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        weight = torch.FloatTensor(weight)
        lr=float(lr)
        tau=float(tau)
        bs=int(bs)
        tol=float(tol)
        n_iter=int(n_iter)
        q = 1 - self.quantile
        n = x.shape[0]  # 获取样本的总数
        p = x.shape[2]  # 获取特征数，x.shape[2] 代表每个时间步的特征数

        self.fnet = LSTMModel(input_size=p, hs=self.hs, num_layers=self.num_layers, output_size=1, dropout=self.dropout).to(self.device)
        optimizer = torch.optim.Adam([{'params': self.fnet.parameters()}], lr=lr)

        # 量化回归损失函数
        def quantile_loss(y_pred, y_true, q):
            errors = y_true - y_pred.detach()
            max_errors = torch.max(q * errors, (q - 1) * errors)
            return max_errors

        self.loss_count = []
        last_loss = np.inf
        flag = 0
        for i_iter in range(n_iter):
            # 确保采样时不超出数据大小
            csample = np.random.choice(n, bs, replace=False)  # 使用np.random.choice来采样，避免超出索引范围
            tmp_x = x[csample].to(self.device)
            tmp_y = y[csample].to(self.device)
            tmp_w = weight[csample].reshape(bs, -1).T.to(self.device)

            # 确保tmp_x的形状为 (batch_size, sequence_length, input_size)
            tmp_x = tmp_x.view(bs, -1, p)  # Reshape为 (bs, sequence_length, input_size)

            
            

            # 处理 tau != None 时的操作
            if tau is not None:   
                if tmp_w.shape[0] != tmp_w.shape[1]:
                    tmp_w = tmp_w.reshape(bs, -1)

                # 获取 tmp_w 的对角线并调整为与样本维度匹配的形状
                tmp_w_diag = torch.diagonal(tmp_w, dim1=-2, dim2=-1)  # 获取每个样本的对角线元素
                tmp_w_diag = tmp_w_diag.unsqueeze(1).expand_as(tmp_w)  # 扩展为 (837, 50)
                tmp_w = tmp_w - tmp_w_diag 

            tmp_fx = self.fnet(tmp_x)
            tmp_my = torch.tile(tmp_y, (bs, 1))
            tmp_mfx = torch.tile(tmp_fx, (1, bs))

            if tau is None:       
                loss = quantile_loss(tmp_y, tmp_fx.ravel(), q).mean()      
            else:     
                loss1 = quantile_loss(tmp_y, tmp_fx.ravel(), q).mean()
                loss2 = torch.mean(quantile_loss(tmp_y, tmp_fx.ravel(), q) * tmp_w.T) * n / (n - 1)
                loss = tau * loss1 + (1 - tau) * loss2

            self.loss_count.append(loss.data.cpu().tolist())

            if (np.abs(last_loss - loss.data.cpu().numpy()) <= tol) & (i_iter >= 100):
                if verbose:
                    print(f'Algorithm converges for BWL model at iter {i_iter}, loss: {self.loss_count[-1]}')
                flag = 1
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.data.cpu().numpy()

        if flag == 0 and verbose:
            print(f'Algorithm may not converge for BWL model, loss: {self.loss_count[-1]}')

    def predict(self, x_new):
        x_new = torch.FloatTensor(x_new).to(self.device)
        return self.fnet(x_new).cpu().data.numpy().ravel()