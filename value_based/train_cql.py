import numpy as np
import torch
import wandb
import argparse
import pickle
import datetime
import random
from cql import CQLAgent
import sys
sys.path.append('../')
from utils.train_eval_utils import get_train_dataloader, get_eval_tensordata, evaluate
import os


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, default="configs/cql_w_config.txt", help="The path where the config file saves.")
    parser.add_argument("--others", type=str, default="", required=False, help="other requirements")
    cfg = parser.parse_args()
    return cfg


def train(args, e_tensordata):
    run_name = args['model_name'] + '_' + args['for'] + '_' + datetime.datetime.now().strftime("%m%d_%H:%M:%S")+ '_s' + str(args['seed'])
    if not os.path.exists(args['save_dir'] + run_name):
        os.mkdir(args['save_dir'] + run_name)
    if args['use_wandb']:
        run = wandb.init(project=args['wandb_project_name'],
                        group=args['wandb_group_name'],
                        tags=args['wandb_tags_name'],
                        config=args,
                        name=run_name)
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        train_metrics = ["Q Loss "+args['for'], "CQL Loss "+args['for'], "Bellman Error "+args['for']]
        for metric in train_metrics:
            wandb.define_metric(metric, step_metric="train_step")
        eval_metrics = ["Absolute CR "+args['for'], "Relative CR "+args['for'], "Total Reward "+args['for']]
        for metric in eval_metrics:
            wandb.define_metric(metric, step_metric="eval_step")

    device = torch.device("cpu" if (not torch.cuda.is_available() or args['device']=='cpu') else args['device'])
    agent = CQLAgent(**args['CQLAgent_param'])

    epoch_cnt, batch_cnt, eval_cnt = 0, 0, 0
    q_loss_list, cql_loss_list, bellman_error_list = [], [], []
    abso_CR_list, rela_CR_list, total_reward_list = [], [], []
    award_tolerance_list = np.linspace(args['train_award_tolerance_start'], args['train_award_tolerance_end'], args['epoch']-1).tolist()
    award_tolerance_list.append(award_tolerance_list[-1])
    if args['for'] == 'r':
        reward_r_factor_list = np.linspace(args['reward_r_factor_start'], args['reward_r_factor_end'], args['epoch']-1).tolist()
        reward_r_factor_list.append(reward_r_factor_list[-1])
    idx = args['info_idx']

    for epoch in range(args['epoch']):
        epoch_cnt += 1
        award_tolerance = award_tolerance_list[epoch]
        np.random.seed(epoch_cnt)
        partition_order = np.arange(27)  # 0~26的训练集
        np.random.shuffle(partition_order)
        partition_order = partition_order.tolist()

        if args['for'] == 'w':
            print("Epoch%d - Award Tolerance:%.2f | Partition order:%s" %
                  (epoch_cnt, award_tolerance, partition_order))
        else:
            reward_r_factor = reward_r_factor_list[epoch]
            print("Epoch%d - Award Tolerance:%.2f | Reward r Factor:%.2f | Partition order:%s" % 
                  (epoch_cnt, award_tolerance, reward_r_factor, partition_order))

        for partition_id in partition_order:
            if args['for'] == 'w':
                dataloader = get_train_dataloader(args, partition_id, award_tolerance, 1, seed=partition_id+epoch)
            else:
                dataloader = get_train_dataloader(args, partition_id, award_tolerance, reward_r_factor, seed=partition_id+epoch)
            print("Epoch%d - Partition%d" % (epoch_cnt, partition_id))

            for batch_idx, data in enumerate(dataloader):
                states, actions, rewards_w, rewards_r, next_states_w, next_states_r, infos, next_infos_w, next_infos_r = data
                states = states.to(device)
                actions = actions.to(device)
                rewards_w = rewards_w.to(device)
                rewards_r = rewards_r.to(device)
                next_states_w = next_states_w.to(device)
                next_states_r = next_states_r.to(device)
                legal_max_actions = infos[:, :, idx['legal_max_action']].long().to(device)
                legal_next_max_actions_w = next_infos_w[:, :, idx['legal_max_action']].long().to(device)
                legal_next_max_actions_r = next_infos_r[:, :, idx['legal_max_action']].long().to(device)

                q_loss, cql_loss, bellman_error = agent.learn((states, actions, rewards_w, rewards_r, next_states_w, next_states_r, \
                                                               legal_max_actions, legal_next_max_actions_w, legal_next_max_actions_r), \
                                                               args['for'])
                q_loss_list.append(q_loss); cql_loss_list.append(cql_loss); bellman_error_list.append(bellman_error)
                if args['use_wandb']:
                    wandb.log({"Q Loss "+args['for']: q_loss, "CQL Loss "+args['for']: cql_loss, "Bellman Error "+args['for']: bellman_error,
                               "train_step": batch_cnt})

                if batch_cnt % args['eval_frequency'] == 0:
                    agent.turn_to_eval()
                    abso_CR, rela_CR, total_reward = evaluate(agent, e_tensordata, args, device, batch_cnt)
                    abso_CR_list.append(abso_CR); rela_CR_list.append(rela_CR); total_reward_list.append(total_reward)
                    if args['use_wandb']:
                        wandb.log({"Absolute CR "+args['for']: abso_CR, "Relative CR "+args['for']: rela_CR, 
                                   "Total Reward "+args['for']: total_reward, 
                                   "eval_step": eval_cnt})
                    eval_cnt += 1

                if args['save_model'] and \
                   epoch > 0 and \
                   batch_idx >= len(dataloader)-55 and \
                   batch_cnt % args['save_frequency'] == 0:
                    #    batch_cnt > args['sample_num'] * args['epoch'] / args['train_batch_size'] - 61 and \
                    torch.save({'curr_Qnet': agent.curr_Qnet.state_dict()},
                                args['save_dir'] + run_name + "/batch" + str(batch_cnt) + ".pt")
                    
                if batch_cnt % args['print_frequency'] == 0:
                    print("E[%d/%d]B[%5d/%d]%s -- Q Loss: %.6f | CQL Loss: %.6f | Bellman Error: %.6f \n\t\t\t Abso CR: %.5f | Rela CR: %.5f Rew: %.3f\n\t\t\t lr: %.4e" \
                          % (epoch_cnt, args['epoch'], batch_cnt, (args['sample_num']//args['train_batch_size']+27)*args['epoch'], args['for'], \
                             q_loss, cql_loss, bellman_error, abso_CR, rela_CR, total_reward, \
                             agent.optimizer.param_groups[0]["lr"]))
                
                batch_cnt += 1

    if args['use_wandb']:
        run.finish()

    intermediate_result_filename = args['save_dir'] + run_name + "/"+ run_name + "_result.pkl"
    with open(intermediate_result_filename, "wb") as f:
        pickle.dump({'q_loss_'+args['for']: q_loss_list,
                     'cql_loss_'+args['for']: cql_loss_list,
                     'bellman_error_'+args['for']: bellman_error_list,
                     'abso_CR_'+args['for']: abso_CR_list,
                     'rela_CR_'+args['for']: rela_CR_list,
                     'total_reward_'+args['for']: total_reward_list,
                     'args': args
                     }, f)
    print("Intermediate result file is saved!")
    print(args['model_name'] + '_' + args['for'] + " Training finished.")


if __name__ == "__main__":
    cfg = get_config()
    with open(cfg.exp_config, 'r') as f:
        args = eval(f.read())
    for key, value in args.items():
        print(key, value)

    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True

    e_tensordata = get_eval_tensordata(args['data_dir'])
    train(args, e_tensordata)

    # nohup python -u train_cql.py --exp_config configs/cql_w_config.txt > logs/cql_w_0701_xxxx_s1111.log 2>&1 &
