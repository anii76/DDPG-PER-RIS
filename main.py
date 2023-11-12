
import argparse
import json
import os, shutil
import time
import json
import traceback
from plot import plot
from utils import create_new_experience, save2json
import test, train, train_2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Choose the type of the experiment
    parser.add_argument('--experiment_type', default='custom', #choices=['custom', 'power', 'rsi_elements', 'noise','learning_rate', 'prioritized'],
                            help='Choose one of the experiment types to reproduce the learning curves given in the paper')
    # train, test pre-load, eval after training
    parser.add_argument("--test", default=False, type=bool, help="Specify trained model to load.")
    parser.add_argument("--save_to", default="./Results", help="Specify Experiment results folder path.")
    parser.add_argument("--verbose_t", default=False, type=bool, help="Print each episode steps")

    # Experiment parameters
    parser.add_argument("--policy", default="DDPG", help='Algorithm (default: DDPG)', choices=['DDPG', 'DDPG-PER', 'TD3'])
    parser.add_argument("--prioritized", default=False, type=bool, help='Prioritized Replay Buffer (default: False)')
    parser.add_argument("--env", default="RIS_MISO", help='Environment name')
    parser.add_argument("--verbose", default=True, type=bool, help='Display training information')
    parser.add_argument("--seed", default=0, type=int, help='Seed number for PyTorch and NumPy (default: 0)')

    # Training-specific parameters
    parser.add_argument("--eval_time_steps", default=0, type=int, metavar="N", help='Number of exploration time steps sampling random actions (default: 0)')
    parser.add_argument("--buffer_size", default=10000, type=int, metavar="N", help='Size of the experience replay buffer (default: 1000000)')
    parser.add_argument("--batch_size", default=128, type=int, metavar='N', help='Batch size (default: 64)')
    parser.add_argument("--save_model", default=True, type=bool, help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')
    parser.add_argument("--iteration", default=0, type=int, help="iteration where to resume training")
    
    # Environment-specific parameters
    parser.add_argument("--num_antennas", default=4, type=int, metavar="N", help="Number of BS transmit antennas")
    parser.add_argument("--num_RIS_elements", default=10, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--num_users", default=1, type=int, metavar="N", help="Number of users")
    parser.add_argument("--num_eve", default=1, type=int, metavar="N", help="Number of eavesdroppers")
    parser.add_argument("--power_t", default=30, type=float, metavar="N", help="Maximum transmit power at the BS")
    parser.add_argument("--awgn_var", default=-80, type=float, metavar="N", help="Additive White Guassian Noise variance")
    parser.add_argument("--path_loss_0", default=-30, type=float, metavar="N", help="Path loss at reference distance")
    parser.add_argument("--path_loss_exponent", default=3, type=float, metavar="N", help="Path loss exponent")
    parser.add_argument("--num_eps", default=10, type=int, metavar='N', help="Numbre of training episodes")
    parser.add_argument("--num_steps_per_ep", default=10000, type=int ,metavar='N', help="Number of time steps per episode")
    parser.add_argument("--random_pos", default=False, type=bool, help="Specify if user/eve positions are generated randomly." )
    parser.add_argument("--save_channels", default=True, type=bool, help="Save generated channels for comparaison with benchmark.") 
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> if fixed specify positions
    #parser.add_argument("--grid_size") #Aw bayen kima articles mara7ch nbdl

    parser.add_argument("--RIS_pos", default=[150, 100], type=str, help="RIS position")
    parser.add_argument("--BS_pos", default=[0, 0], type=str, help="BS position")
    parser.add_argument("--LoS", default=True, type=bool, help="Direct link between BS & User")
    parser.add_argument("--without_RIS", default=False, type=bool, help="Test system performance without RIS")
    parser.add_argument("--variant_CSI", default=False, type=bool, help="Test system when CSI is imperfect")

    # Algorithm-specific parameters (even OU noise params are changed (DDPG google impl))
    parser.add_argument("--lr", default=0.001, type=float, help="Neural Network Learning rate.")
    parser.add_argument("--critic_lr", default=0.001, type=float, help="Critic Network learning rate.")
    parser.add_argument("--exploration_noise", default="OU", choices=["OU", "Normal"], help="DDPG Exploration noise added to the action.")
    parser.add_argument("--noise_scalar", default=1, type=float, help="Normal noise variance value.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Rewards Discount factor (gamma).")
    parser.add_argument("--tau", default=0.001, type=float, help="Target networks soft update rate.")
    parser.add_argument("--decay", default=1e-2, type=float, help="Learning rate discount factor (in optimiser).")
    parser.add_argument("--per_alpha", default=0.6, type=float, help="Prioritized replay exponent (degree of prioritization).")
    parser.add_argument("--per_beta", default=0.4, type=float, help="Importance-Sampling exponent.")
    parser.add_argument("--per_beta_annealed", default=True, type=bool, help="Schedule a linear annealing to determine better beta value through iterations.")
    parser.add_argument("--per_importance_sampling", default=False, type=bool, help="Enable/Disable Ordinary importance sampling weights.") #true / false


    #number of hidden layers & their neurones (change directly in ddpg)
    args = parser.parse_args()

    if isinstance(args.BS_pos, str):
        args.BS_pos = list(map(int, args.BS_pos.split(',')))

    if isinstance(args.RIS_pos, str):
        args.RIS_pos = list(map(int, args.RIS_pos.split(',')))
    
    print(args)

    # Create folders & files
    if args.load_model =="":
        path = create_new_experience(args)
        #save params in json file
        params = save2json(args, path)
    else :
        path = args.load_model 
        #load stored params
        with open(f'{path}/params.json', 'r') as json_file:
            params = json.load(json_file)
        params["experiment"]["load_model"] = args.load_model
        params = json.dumps(params,indent=4)
    
    #train(experiment) [reads from params.json]
    #or
    #test(experiment)
    if args.test :
        test.test(params)
    else:
        try :
             choice = 1
             if choice == 1:
                 train_2.train(params)
             else :
                  train.train(params)
        except Exception as e:
            # Print the exception details
            print("An exception occurred:")
            print("Type:", type(e).__name__)
            print("Exception:", e)
            print("Traceback:")
            traceback.print_exc()

    #plot results
    plot(path, args.test)

    print("done")