import torch
import pickle
import gc
from torch.utils.data import DataLoader, TensorDataset, RandomSampler


def get_train_dataloader(args, partition_id=0, award_tolerance=0.0, reward_r_factor=1.0, seed=0):
    filename = args['data_dir'] + "MDP_transition_" + str(partition_id) + ".pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)
    rewards_w = data['rewards_w']
    rewards_r = data['rewards_r']

    if args['reward_reshape_factor'] > 0:  # reward reshape
        actions = data['actions']
        infos = data['infos']
        best_infos = data['best_infos']
        idx = args['info_idx']
        best_idx = args['best_info_idx']
        for i in range(len(actions)):
            if rewards_w[i] == rewards_r[i] == 0 and \
               infos[i, actions[i], idx['category']] == best_infos[i, best_idx['category']] and \
               infos[i, actions[i], idx['industry']] == best_infos[i, best_idx['industry']] and \
               (1 - award_tolerance) * best_infos[i, best_idx['award']] <= infos[i, actions[i], idx['award']] and \
               (1 + award_tolerance) * best_infos[i, best_idx['award']] >= infos[i, actions[i], idx['award']]:
                rewards_w[i] = best_infos[i, best_idx['best_reward_w']] * args['reward_reshape_factor']
                rewards_r[i] = best_infos[i, best_idx['best_reward_r']] * args['reward_reshape_factor']

    tensordata = TensorDataset(torch.from_numpy(data['states']).float(),
                               torch.from_numpy(data['actions']).long(),
                               torch.from_numpy(rewards_w * args['reward_w_factor']).float(),
                               torch.from_numpy(rewards_r * reward_r_factor).float(),
                               torch.from_numpy(data['next_states_w']).float(),
                               torch.from_numpy(data['next_states_r']).float(),
                               torch.from_numpy(data['infos']).float(),
                               torch.from_numpy(data['next_infos_w']).float(),
                               torch.from_numpy(data['next_infos_r']).float())
    sampler = RandomSampler(tensordata, generator=torch.Generator().manual_seed(seed))
    dataloader = DataLoader(tensordata, batch_size=args['train_batch_size'], sampler=sampler, num_workers=0)
    del data, tensordata, sampler
    gc.collect()

    return dataloader


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


def evaluate(agent, tensordata, args, device, batch_cnt, seed=32):
    sampler = RandomSampler(tensordata, generator=torch.Generator().manual_seed(seed))
    dataloader = DataLoader(tensordata, batch_size=args['eval_batch_size'], sampler=sampler, num_workers=0)

    total_sample = len(dataloader.dataset)
    idx = args['info_idx']
    abso_cnt, rela_cnt = 0, 0
    reward_sum = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            states, actions, rewards_w, rewards_r, infos = data
            states = states.to(device)
            legal_max_actions = infos[:, :, idx['legal_max_action']].long().to(device)
            
            e_actions = agent.get_action_eval(states, legal_max_actions, batch_idx, batch_cnt)

            for i in range(len(states)):
                # count absolute completed samples; add reward
                if e_actions[i] == actions[i]:
                    abso_cnt += 1
                    if args['for'] == 'w':
                        reward_sum += rewards_w[i].item()
                    else:
                        reward_sum += rewards_r[i].item()
                # count relative completed samples
                if infos[i, actions[i], idx['category']] == infos[i, e_actions[i], idx['category']] and \
                   infos[i, actions[i], idx['industry']] == infos[i, e_actions[i], idx['industry']] and \
                   (1-args['eval_award_tolerance']) * infos[i, actions[i], idx['award']] <= infos[i, e_actions[i], idx['award']] and \
                   (1+args['eval_award_tolerance']) * infos[i, actions[i], idx['award']] >= infos[i, e_actions[i], idx['award']]:
                    rela_cnt += 1
    
    return abso_cnt/total_sample, rela_cnt/total_sample, reward_sum
