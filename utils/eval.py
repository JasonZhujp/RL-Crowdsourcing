import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# 对w和r的Q值/动作概率分别归一化之后，在用w_weight加权，以消除二者Q值/动作概率绝对值的大小差异
# 对w和r的Q值/动作概率分别归一化之后，在用w_weight加权，以消除二者Q值/动作概率绝对值的大小差异
# 对w和r的Q值/动作概率分别归一化之后，在用w_weight加权，以消除二者Q值/动作概率绝对值的大小差异
# 对w和r的Q值/动作概率分别归一化之后，在用w_weight加权，以消除二者Q值/动作概率绝对值的大小差异
# 对w和r的Q值/动作概率分别归一化之后，在用w_weight加权，以消除二者Q值/动作概率绝对值的大小差异
# 对w和r的Q值/动作概率分别归一化之后，在用w_weight加权，以消除二者Q值/动作概率绝对值的大小差异

# BC
def get_action_eval(self, states, legal_max_actions, batch_idx):
    with torch.no_grad():
        prob_worker = F.softmax(self.net_w(states), dim=1)
        prob_requester = F.softmax(self.net_r(states), dim=1)
        prob_both = self.w_weight * F.softmax(self.net_w(states), dim=1) + (1 - self.w_weight) * F.softmax(self.net_r(states), dim=1)
        mask_condition = torch.arange(self.max_action_num).expand(states.shape[0], self.max_action_num).to(self.device) \
                            < legal_max_actions
        zero = torch.full((states.shape[0], self.max_action_num), 0.0).to(self.device)
        if batch_idx == 0:
            print('w', prob_worker[:5, :9])
            print('r', prob_requester[:5, :9])

        actions_worker = torch.argmax(torch.where(mask_condition, prob_worker, zero).cpu(), dim=-1)
        actions_requester = torch.argmax(torch.where(mask_condition, prob_requester, zero).cpu(), dim=-1)
        actions_both = torch.argmax(torch.where(mask_condition, prob_both, zero).cpu(), dim=-1)

    return actions_worker, actions_requester, actions_both


# DDQN
def get_action_eval(self, states, legal_max_actions, batch_idx, batch_cnt):
        with torch.no_grad():
            # 3种加权Q值
            curr_Q_worker = self.curr_Qnet_w(states)
            curr_Q_requester = self.curr_Qnet_r(states)
            curr_Q_both = self.w_weight * self.curr_Qnet_w(states) + (1 - self.w_weight) * self.curr_Qnet_r(states)
            # 对于每个样本 若索引值小于合法最大动作的索引 则变为True 否则是False
            mask_condition = torch.arange(self.max_action_num).expand(states.shape[0], self.max_action_num).to(self.device) \
                             < legal_max_actions
            inf = torch.full((states.shape[0], self.max_action_num), float('-inf')).to(self.device)
            if batch_idx == 0 and batch_cnt % 200 == 0:
                print('w', curr_Q_worker[:5, :9])
                print('r', curr_Q_requester[:5, :9])
            # 将False位置用无穷小填充
            actions_worker = torch.argmax(torch.where(mask_condition, curr_Q_worker, inf).cpu(), dim=-1)
            actions_requester = torch.argmax(torch.where(mask_condition, curr_Q_requester, inf).cpu(), dim=-1)
            actions_both = torch.argmax(torch.where(mask_condition, curr_Q_both, inf).cpu(), dim=-1)

        return actions_worker, actions_requester, actions_both


# CQL
def get_action_eval(self, states, legal_max_actions, batch_idx, batch_cnt):
        with torch.no_grad():
            # 3种加权Q值
            curr_Q_worker = self.curr_Qnet_w(states)
            curr_Q_requester = self.curr_Qnet_r(states)
            curr_Q_both = self.w_weight * self.curr_Qnet_w(states) + (1 - self.w_weight) * self.curr_Qnet_r(states)
            # 对于每个样本 若索引值小于合法最大动作的索引 则变为True 否则是False
            mask_condition = torch.arange(self.max_action_num).expand(states.shape[0], self.max_action_num).to(self.device) \
                             < legal_max_actions
            inf = torch.full((states.shape[0], self.max_action_num), float('-inf')).to(self.device)
            if batch_idx == 0 and batch_cnt % 200 == 0:
                print('w', curr_Q_worker[:5, :9])
                print('r', curr_Q_requester[:5, :9])
            # 将False位置用无穷小填充
            actions_worker = torch.argmax(torch.where(mask_condition, curr_Q_worker, inf).cpu(), dim=-1)
            actions_requester = torch.argmax(torch.where(mask_condition, curr_Q_requester, inf).cpu(), dim=-1)
            actions_both = torch.argmax(torch.where(mask_condition, curr_Q_both, inf).cpu(), dim=-1)

        return actions_worker, actions_requester, actions_both


# SAC
def get_action_eval(self, states, legal_max_actions):
        with torch.no_grad():
            mask_condition = torch.arange(self.max_action_num).expand(states.shape[0], self.max_action_num).to(self.device) \
                            < legal_max_actions
            zero = torch.full((states.shape[0], self.max_action_num), 0.0).to(self.device)

            logits_w = self.actor_w(states)
            # print(logits_w)
            logits_masked_w = torch.where(mask_condition, logits_w, zero)
            policy_dist_w = Categorical(logits=logits_masked_w)  # 离散概率分布
            a_probs_w = policy_dist_w.probs  # 动作概率向量(相加为1)

            logits_r = self.actor_r(states)
            logits_masked_r = torch.where(mask_condition, logits_r, zero)
            policy_dist_r = Categorical(logits=logits_masked_r)
            a_probs_r = policy_dist_r.probs

            a_probs = self.w_weight * a_probs_w  + (1 - self.w_weight) * a_probs_r  # 离散动作概率加权
            policy_dist_both = Categorical(probs=a_probs)  # 加权的离散概率分布

        return policy_dist_w.sample().cpu(), policy_dist_r.sample().cpu(), policy_dist_both.sample().cpu()


