import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class Row_Wise_Linear(nn.Module):
    def __init__(self, feature_dim, inner_output_dim, max_action_num):
        super(Row_Wise_Linear, self).__init__()
        self.linear = nn.ModuleList([nn.Linear(feature_dim, inner_output_dim) for _ in range (max_action_num)])
    
    def forward(self, x):
        out_list = []
        for i in range(x.size(1)):  # for each row of x
            out = x[:, i, :]
            out = torch.relu(self.linear[i](out))
            out_list.append(out)
        out = torch.stack(out_list, dim=1)
        return out


class QNet(nn.Module):
    def __init__(self,
                 max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                 nhead_1, nhead_2, encoder_layer_num, feature_dict):
        super(QNet, self).__init__()
        self.fd = feature_dict
        self.emb_category = nn.Embedding(category_num, category_emb_dim)
        self.emb_industry = nn.Embedding(industry_num, industry_emb_dim)

        feature_dim = 2 * (category_emb_dim + industry_emb_dim + 1) + self.fd['others_sec'][1] - self.fd['others_sec'][0]
        self.row_wise_linear1 = Row_Wise_Linear(feature_dim, inner_output_dim, max_action_num)

        trans_encoder_layer1 = nn.TransformerEncoderLayer(d_model=inner_output_dim, dim_feedforward=inner_output_dim*4, nhead=nhead_1, batch_first=True)
        # self.trans_encoder1 = nn.TransformerEncoder(trans_encoder_layer1, num_layers=encoder_layer_num, norm=nn.LayerNorm(64))
        self.trans_encoder1 = nn.TransformerEncoder(trans_encoder_layer1, num_layers=encoder_layer_num)

        self.row_wise_linear2 = Row_Wise_Linear(inner_output_dim, inner_output_dim, max_action_num)

        trans_encoder_layer2 = nn.TransformerEncoderLayer(d_model=inner_output_dim, dim_feedforward=inner_output_dim*4, nhead=nhead_2, batch_first=True)
        # self.trans_encoder2 = nn.TransformerEncoder(trans_encoder_layer2, num_layers=1, norm=nn.LayerNorm(64))
        self.trans_encoder2 = nn.TransformerEncoder(trans_encoder_layer2, num_layers=1)

        self.row_wise_linear3 = Row_Wise_Linear(inner_output_dim, 1, max_action_num)

    def forward(self, x):
        row_x = torch.cat(tensors=(self.emb_category(x[:, :, self.fd['category_sec'][0]].long()),  # category_emb of a_p
                                   torch.mean(self.emb_category(x[:, :, self.fd['category_sec'][0]+1:self.fd['category_sec'][1]].long()), dim=2),  # mean category_emb of l_p
                                   self.emb_industry(x[:, :, self.fd['industry_sec'][0]].long()),  # industry_emb of a_p
                                   torch.mean(self.emb_industry(x[:, :, self.fd['industry_sec'][0]+1:self.fd['industry_sec'][1]].long()), dim=2),  # mean industry_emb of l_p
                                   x[:, :, self.fd['award_sec'][0]].unsqueeze(-1),  # award of a_p
                                   torch.mean(x[:, :, self.fd['award_sec'][0]+1:self.fd['award_sec'][1]], dim=2).unsqueeze(-1),  # mean award of l_p
                                   x[:, :, self.fd['others_sec'][0]:self.fd['others_sec'][1]]  # gaps of a_p and mean l_p and qualities of a_p
                                   ), 
                          dim=-1)                                   # (batch_size, max_action_num, feature_dim)
        row_out = self.row_wise_linear1(row_x)                      # (batch_size, max_action_num, inner_output_dim)
        transformer_out = self.trans_encoder1(row_out)              # (batch_size, max_action_num, inner_output_dim)
        row_out = self.row_wise_linear2(transformer_out)            # (batch_size, max_action_num, inner_output_dim)
        transformer_out = self.trans_encoder2(row_out)              # (batch_size, max_action_num, inner_output_dim)
        out = self.row_wise_linear3(transformer_out).squeeze(-1)    # (batch_size, max_action_num)
        
        return out


class DDQNAgent():
    def __init__(self,
                 max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                 nhead_1, nhead_2, encoder_layer_num, feature_dict,
                 lr, factor, patience, improve,
                 tau, gamma,
                 device):
        self.max_action_num = max_action_num
        self.curr_Qnet = QNet(max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                              nhead_1, nhead_2, encoder_layer_num, feature_dict).to(device)
        self.targ_Qnet = QNet(max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                              nhead_1, nhead_2, encoder_layer_num, feature_dict).to(device)
        self.optimizer = optim.Adam(params=self.curr_Qnet.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience, 
                                             threshold=improve, threshold_mode='rel')
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.curr_Qnet.apply(self.init_weights)
        self.targ_Qnet.load_state_dict(self.curr_Qnet.state_dict())
    

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.normal_()
        elif isinstance(m, nn.TransformerEncoderLayer):
                for sub_m in m.modules():
                    if isinstance(sub_m, (nn.Linear, nn.LayerNorm)):
                        torch.nn.init.normal_(sub_m.weight)
                        if hasattr(sub_m, 'bias') and sub_m.bias is not None:
                            sub_m.bias.data.fill_(0.01)

    
    def get_action_train(self, net, states, legal_max_actions):
        Q = net(states)
        mask_condition = torch.arange(self.max_action_num).expand(states.shape[0], self.max_action_num).to(self.device) \
                         < legal_max_actions
        inf = torch.full((states.shape[0], self.max_action_num), float('inf')).to(self.device)

        return torch.argmax(torch.where(mask_condition, Q, inf), dim=1).unsqueeze(1)


    def turn_to_eval(self):
        self.curr_Qnet.eval()


    def get_action_eval(self, states, legal_max_actions, batch_idx, batch_cnt):
        with torch.no_grad():
            curr_Q = self.curr_Qnet(states)
            # 对于每个样本 若索引值小于合法最大动作的索引 则变为True 否则是False
            mask_condition = torch.arange(self.max_action_num).expand(states.shape[0], self.max_action_num).to(self.device) \
                             < legal_max_actions
            inf = torch.full((states.shape[0], self.max_action_num), float('-inf')).to(self.device)
            if batch_idx == 0 and batch_cnt % 200 == 0:
                print(torch.where(mask_condition, curr_Q, inf)[100:150:10, ::10])
            # 将False位置用无穷小填充
            actions = torch.argmax(torch.where(mask_condition, curr_Q, inf).cpu(), dim=-1)

        return actions
    
        
    def learn(self, experiences, train_for):
        self.curr_Qnet.train()

        if train_for == 'w':
            states, actions, rewards, _, next_states, _, _, legal_next_max_actions, _ = experiences
        else:
            states, actions, _, rewards, _, next_states, _, _, legal_next_max_actions = experiences
        
        curr_argmax_next_a = self.get_action_train(self.curr_Qnet, next_states, legal_next_max_actions)  # argmax_a' Q(s',a';θ)
        with torch.no_grad():
            targ_next_Q = self.targ_Qnet(next_states).gather(1, curr_argmax_next_a).squeeze()  # Q(s',a';θ')
            targ_Q = rewards + (self.gamma * targ_next_Q)  # Q(s,a;θ') = r + γQ(s',a';θ')
        curr_Q = self.curr_Qnet(states).gather(1, actions.unsqueeze(1)).squeeze()  # Q(s,a;θ)

        bellman_error = F.mse_loss(curr_Q, targ_Q)

        self.optimizer.zero_grad()
        bellman_error.backward()
        # clip_grad_norm_(self.curr_Qnet.parameters(), 1)  # 梯度的二范数截断到≤1
        self.optimizer.step()
        self.scheduler.step(bellman_error)

        # 目标网络软更新
        self.soft_update(self.curr_Qnet, self.targ_Qnet)

        return bellman_error.detach().item()
 
        
    def soft_update(self, curr_model, targ_model):
        for targ_param, curr_param in zip(targ_model.parameters(), curr_model.parameters()):
            targ_param.data.copy_(self.tau * curr_param.data + (1.0-self.tau) * targ_param.data)