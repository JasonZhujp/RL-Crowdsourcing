import numpy as np
import torch
import wandb
import argparse
import pickle
import datetime
import random
from bc import BCAgent
import sys
sys.path.append('../')
from utils.train_eval_utils import get_train_dataloader, get_eval_tensordata, evaluate
import os


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, default="configs/bc_w_config.txt", help="The path where the config file saves.")
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
                        # tags=args['wandb_tags_name'],
                        config=args,
                        name=run_name)
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        train_metrics = ["Cross Entropy "+args['for']]
        for metric in train_metrics:
            wandb.define_metric(metric, step_metric="train_step")
        eval_metrics = ["Absolute CR "+args['for'], "Relative CR "+args['for'], "Total Reward "+args['for']]
        for metric in eval_metrics:
            wandb.define_metric(metric, step_metric="eval_step")

    device = torch.device("cpu" if (not torch.cuda.is_available() or args['device']=='cpu') else args['device'])
    agent = BCAgent(**args['BCAgent_param'])

    epoch_cnt, batch_cnt, eval_cnt = 0, 0, 0
    ce_list = []
    abso_CR_list, rela_CR_list, total_reward_list = [], [], []
    for epoch in range(args['epoch']):
        epoch_cnt += 1
        np.random.seed(epoch_cnt)
        partition_order = np.arange(27)  # 0~26的训练集
        np.random.shuffle(partition_order)
        partition_order = partition_order.tolist()
        print("Epoch%d - Whole partition order:%s" % (epoch_cnt, partition_order))

        for partition_id in partition_order:
            dataloader = get_train_dataloader(args, partition_id, award_tolerance=0, reward_r_factor=1, seed=partition_id+epoch)
            print("Epoch%d - Partition%d" % (epoch_cnt, partition_id))

            for batch_idx, data in enumerate(dataloader):
                states, actions, _, _, _, _, _, _, _ = data
                states = states.to(device)
                actions = actions.to(device)
                
                ce = agent.learn((states, actions))
                ce_list.append(ce)
                if args['use_wandb']:
                    wandb.log({"Cross Entropy "+args['for']: ce,
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
                    torch.save({'net': agent.net.state_dict()},
                                args['save_dir'] + run_name + "/batch" + str(batch_cnt) + ".pt")
                    
                if batch_cnt % args['print_frequency'] == 0:
                    print("E[%d/%d]B[%5d/%d]%s -- CE Loss: %.6f \n\t\t\t Abso CR: %.5f | Rela CR: %.5f Rew: %.3f\n\t\t\t lr: %.4e"
                          % (epoch_cnt, args['epoch'], batch_cnt, (args['sample_num']//args['train_batch_size']+27)*args['epoch'], args['for'], \
                             ce, abso_CR, rela_CR, total_reward, \
                             agent.optimizer.param_groups[0]["lr"]))
                
                batch_cnt += 1

    if args['use_wandb']:
        run.finish()

    intermediate_result_filename = args['save_dir'] + run_name + "/"+ run_name + "_result.pkl"
    with open(intermediate_result_filename, "wb") as f:
        pickle.dump({'ce_'+args['for']+args['for']: ce_list,
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

    # nohup python -u train_bc.py --exp_config configs/bc_w_config.txt> logs/bc_w_0701_xxxx.log 2>&1 &
