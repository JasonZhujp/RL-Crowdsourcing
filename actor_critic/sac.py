import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
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


class Net(nn.Module):
    def __init__(self,
                 max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                 nhead_1, nhead_2, encoder_layer_num, feature_dict):
        super(Net, self).__init__()
        self.fd = feature_dict
        self.emb_category = nn.Embedding(category_num, category_emb_dim)
        self.emb_industry = nn.Embedding(industry_num, industry_emb_dim)

        feature_dim = 2 * (category_emb_dim + industry_emb_dim + 1) + self.fd['others_sec'][1] - self.fd['others_sec'][0]
        self.row_wise_linear1 = Row_Wise_Linear(feature_dim, inner_output_dim, max_action_num)

        trans_encoder_layer1 = nn.TransformerEncoderLayer(d_model=inner_output_dim, dim_feedforward=inner_output_dim*4, nhead=nhead_1, batch_first=True)
        self.trans_encoder1 = nn.TransformerEncoder(trans_encoder_layer1, num_layers=encoder_layer_num)

        self.row_wise_linear2 = Row_Wise_Linear(inner_output_dim, inner_output_dim, max_action_num)

        trans_encoder_layer2 = nn.TransformerEncoderLayer(d_model=inner_output_dim, dim_feedforward=inner_output_dim*4, nhead=nhead_2, batch_first=True)
        self.trans_encoder2 = nn.TransformerEncoder(trans_encoder_layer2, num_layers=1)

        self.row_wise_linear3 = Row_Wise_Linear(inner_output_dim, 1, max_action_num)

    def forward(self, x):
        row_x = torch.cat(tensors=(self.emb_category(x[:, :, self.fd['category_sec'][0]].long()),  # category_emb of a_p
                                   torch.mean(self.emb_category(x[:, :, self.fd['category_sec'][0]+1:self.fd['category_sec'][1]].long()), dim=2),  # mean category_emb of l_p
                                   self.emb_industry(x[:, :, self.fd['industry_sec'][0]].long()),  # industry_emb of a_p
                                   torch.mean(self.emb_industry(x[:, :, self.fd['industry_sec'][0]+1:self.fd['industry_sec'][1]].long()), dim=2),  # mean industry_emb of l_p
                                   x[:, :, self.fd['award_sec'][0]].unsqueeze(-1),  # award of a_p
                                   torch.mean(x[:, :, self.fd['award_sec'][0]+1:self.fd['award_sec'][1]], dim=2).unsqueeze(-1),  # mean award of l_p
                                   x[:, :, self.fd['others_sec'][0]:self.fd['others_sec'][1]]  # start_gap, end_gap, worker_quality and project_quality of a_p
                                   ), 
                          dim=-1)                                   # (batch_size, max_action_num, feature_dim)
        row_out = self.row_wise_linear1(row_x)                      # (batch_size, max_action_num, inner_output_dim)
        transformer_out = self.trans_encoder1(row_out)              # (batch_size, max_action_num, inner_output_dim)
        row_out = self.row_wise_linear2(transformer_out)            # (batch_size, max_action_num, inner_output_dim)
        transformer_out = self.trans_encoder2(row_out)              # (batch_size, max_action_num, inner_output_dim)
        out = self.row_wise_linear3(transformer_out).squeeze(-1)    # (batch_size, max_action_num)
 
        return out


class SACAgent():
    def __init__(self,
                 max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                 nhead_1, nhead_2, encoder_layer_num, feature_dict,
                 a_update_frequency, temperature, tau, gamma,
                 a_lr, q_lr, a_factor, a_patience, a_improve, q_factor, q_patience, q_improve,
                 device):
        self.max_action_num = max_action_num
        self.actor = Net(max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                         nhead_1, nhead_2, encoder_layer_num, feature_dict).to(device)
        self.curr_Qnet1 = Net(max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                              nhead_1, nhead_2, encoder_layer_num, feature_dict).to(device)
        self.curr_Qnet2 = Net(max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                              nhead_1, nhead_2, encoder_layer_num, feature_dict).to(device)
        self.targ_Qnet1 = Net(max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                              nhead_1, nhead_2, encoder_layer_num, feature_dict).to(device)
        self.targ_Qnet2 = Net(max_action_num, category_num, category_emb_dim, industry_num, industry_emb_dim, inner_output_dim,
                              nhead_1, nhead_2, encoder_layer_num, feature_dict).to(device)
        self.a_optimizer = optim.Adam(self.actor.parameters(), lr=a_lr)
        self.q_optimizer = optim.Adam(list(self.curr_Qnet1.parameters()) + list(self.curr_Qnet2.parameters()), lr=q_lr)
        self.a_scheduler = ReduceLROnPlateau(self.a_optimizer, mode='min', factor=a_factor, patience=a_patience, threshold=a_improve, threshold_mode='rel')
        self.q_scheduler = ReduceLROnPlateau(self.q_optimizer, mode='min', factor=q_factor, patience=q_patience, threshold=q_improve, threshold_mode='rel')

        self.curr_Qnet1.apply(self._init_weights)
        self.curr_Qnet2.apply(self._init_weights)
        self.actor.apply(self._init_weights)
        self.targ_Qnet1.load_state_dict(self.curr_Qnet1.state_dict())
        self.targ_Qnet2.load_state_dict(self.curr_Qnet2.state_dict())

        self.a_update_frequency = a_update_frequency
        self.temperature = temperature
        self.tau = tau
        self.gamma = gamma
        self.device = device


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.TransformerEncoderLayer):
                for sub_m in m.modules():
                    if isinstance(sub_m, (nn.Linear, nn.LayerNorm)):
                        torch.nn.init.normal_(sub_m.weight)
                        if hasattr(sub_m, 'bias') and sub_m.bias is not None:
                            sub_m.bias.data.fill_(0.01)


    def turn_to_eval(self):
        self.curr_Qnet1.eval()
        self.curr_Qnet2.eval()
        self.actor.eval()


    def _get_action_train(self, actor, states, legal_max_actions):
        logits = actor(states)
        # 对于每个样本 若索引值小于合法最大动作的索引 则变为True 否则是False
        mask_condition = torch.arange(self.max_action_num).expand(states.shape[0], self.max_action_num).to(self.device) \
                         < legal_max_actions
        zero = torch.full((states.shape[0], self.max_action_num), 0.0).to(self.device)
        # 将False位置用概率0.0填充
        logits_masked = torch.where(mask_condition, logits, zero)
        policy_dist = Categorical(logits=logits_masked)  # 离散概率分布
        # actions = policy_dist.sample()  # 随机性
        a_probs = policy_dist.probs  # 动作概率向量(相加为1)
        actions = torch.argmax(a_probs, dim=1)  # 确定性
        a_log_probs = F.log_softmax(logits_masked, dim=1)  # 动作对数概率向量        

        return actions, a_probs, a_log_probs
    
    
    def get_action_eval(self, states, legal_max_actions, batch_idx, batch_cnt):
        with torch.no_grad():
            logits = self.actor(states)
            mask_condition = torch.arange(self.max_action_num).expand(states.shape[0], self.max_action_num).to(self.device) \
                            < legal_max_actions
            zero = torch.full((states.shape[0], self.max_action_num), 0.0).to(self.device)
            logits_masked = torch.where(mask_condition, logits, zero)
            if batch_idx == 0 and batch_cnt % 200 == 0:
                print(logits_masked[100:150:10, ::10])
            policy_dist = Categorical(logits=logits_masked)  # 离散概率分布
            a_probs = policy_dist.probs  # 动作概率向量(相加为1)
            actions = torch.argmax(a_probs, dim=1).cpu()  # 评估时采用最大概率的动作(确定性)，而不是sample采样(随机性)

        return actions


    def learn(self, experiences, train_for, batch_cnt):
        self.curr_Qnet1.train()
        self.curr_Qnet2.train()
        self.actor.train()
        if train_for == 'w':
            states, actions, rewards, _, next_states, _, legal_max_actions, legal_next_max_actions, _ = experiences
        else:
            states, actions, _, rewards, _, next_states, legal_max_actions, _, legal_next_max_actions = experiences
        
        # TRAIN CRITIC
        with torch.no_grad():
            _, next_a_probs, next_a_log_probs = self._get_action_train(self.actor, next_states, legal_next_max_actions)
            # V(s') = Π(s') * [Q(s',a') - α * logΠ(s')]
            min_targ_next_q = next_a_probs * (torch.min(self.targ_Qnet1(next_states), self.targ_Qnet2(next_states)) - self.temperature * next_a_log_probs)
            # E_{s'}[V(s')]
            min_targ_expected_next_q = min_targ_next_q.sum(1)
            # B`Q(s,a) = r(s,a) + γE_{s'}[V(s')]
            min_targ_q = rewards + self.gamma * min_targ_expected_next_q
        curr_q1 = self.curr_Qnet1(states).gather(1, actions.unsqueeze(1)).squeeze()  # Q(s,a;θ) given a
        curr_q2 = self.curr_Qnet2(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_loss = F.mse_loss(curr_q1, min_targ_q) + F.mse_loss(curr_q2, min_targ_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        clip_grad_norm_(list(self.curr_Qnet1.parameters()) + list(self.curr_Qnet2.parameters()), 1)
        self.q_optimizer.step()
        self.q_scheduler.step(q_loss)

        # TRAIN ACTON (Delayed)
        a_losses = []
        if batch_cnt % self.a_update_frequency == 0:
            for _ in range(self.a_update_frequency):  # Loop time is same as update frequency
                with torch.no_grad():
                    min_curr_q = torch.min(self.curr_Qnet1(states), self.curr_Qnet2(states))  # Q(s,a)
                _, a_probs, a_log_probs = self._get_action_train(self.actor, states, legal_max_actions)
                # minimize: E_{s}[Π(a|s) * [α * logΠ(a|s) - Q(s,a)]]
                a_loss = (a_probs * ((self.temperature * a_log_probs) - min_curr_q)).mean()
                self.a_optimizer.zero_grad()
                a_loss.backward()
                clip_grad_norm_(self.actor.parameters(), 1)
                self.a_optimizer.step()
                # self.a_scheduler.step(-1*a_loss)
                a_losses.append(a_loss.detach().item())

        # soft update for target network
        self._soft_update(self.curr_Qnet1, self.targ_Qnet1)
        self._soft_update(self.curr_Qnet2, self.targ_Qnet2)

        return q_loss.detach().item(), a_losses
    
       
    def _soft_update(self, curr_model, targ_model):
        for targ_param, curr_param in zip(targ_model.parameters(), curr_model.parameters()):
            targ_param.data.copy_(self.tau * curr_param.data + (1.0-self.tau) * targ_param.data)
