import numpy as np
import torch
import argparse
import pickle
import gc
import random
from torch.utils.data import DataLoader, TensorDataset, RandomSampler


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, default="configs/random_config.txt", help="The path where the config_file for Random experiment saves.")
    parser.add_argument("--others", type=str, default="", required=False, help="other requirements")
    cfg = parser.parse_args()
    return cfg


def get_eval_tensordata(data_dir):
    filename = data_dir + "test_transition.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)

    e_tensordata = TensorDataset(torch.from_numpy(data['states']).float(),
                                 torch.from_numpy(data['actions']).long(),
                                 torch.from_numpy(data['rewards_w']).float(),
                                 torch.from_numpy(data['rewards_r']).float(),
                                 torch.from_numpy(data['infos']).float())
    del data
    gc.collect()

    return e_tensordata


def evaluate(tensordata, args):
    sampler = RandomSampler(tensordata, generator=torch.Generator().manual_seed(args['seed']))
    dataloader = DataLoader(tensordata, batch_size=args['eval_batch_size'], sampler=sampler, num_workers=0)

    total_sample = len(dataloader.dataset)
    idx = args['info_idx']
    absolute_cnt, relative_cnt = 0, 0
    reward_sum_w, reward_sum_r = 0.0, 0.0
    for batch_idx, data in enumerate(dataloader):
        random.seed(batch_idx)
        _, actions, rewards_w, rewards_r, infos = data
        legal_max_actions = infos[:, :, idx['legal_max_action']].long()

        for i in range(len(actions)):
            # count absolute completed samples; add reward
            e_action = random.randint(0, legal_max_actions[i, actions[i]])

            if e_action == actions[i]:
                absolute_cnt += 1
                reward_sum_w += rewards_w[i].item()
                reward_sum_r += rewards_r[i].item()
            # count relative completed samples
            if infos[i, actions[i], idx['category']] == infos[i, e_action, idx['category']] and \
            infos[i, actions[i], idx['industry']] == infos[i, e_action, idx['industry']] and \
            (1-args['award_tolerance']) * infos[i, actions[i], idx['award']] <= infos[i, e_action, idx['award']] and \
            (1+args['award_tolerance']) * infos[i, actions[i], idx['award']] >= infos[i, e_action, idx['award']]:
                relative_cnt += 1
    
    return absolute_cnt/total_sample, relative_cnt/total_sample, \
           reward_sum_w.item() if type(reward_sum_w)==torch.Tensor else reward_sum_w, \
           reward_sum_r.item() if type(reward_sum_r)==torch.Tensor else reward_sum_r


if __name__ == "__main__":
    cfg = get_config()
    with open(cfg.exp_config, 'r') as f:
        args = eval(f.read())
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    e_tensordata = get_eval_tensordata(args['data_dir'])
    absolute_CR, relative_CR, total_reward_w, total_reward_r = evaluate(e_tensordata, args)
    print("Abso CR:%.5f | Rela CR:%.5f | W Rew:%d | R Rew:%.3f" % (absolute_CR, relative_CR, total_reward_w, total_reward_r))
