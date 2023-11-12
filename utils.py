import json
import os
import random
import time
import numpy as np
import scipy.io

from env.env import Environment

# In a 100x100 grid
def generate_random_positions(num_users):
    grid_min_x, grid_max_x = 100, 200
    grid_min_y, grid_max_y = 0, 100
    positions = []

    for _ in range(num_users):
        x = random.randint(grid_min_x, grid_max_x)
        y = random.randint(grid_min_y, grid_max_y)
        positions.append([x, y])

    return positions

# called in each episode (after reset)
def save_channels(env,ch):
    ch["H_BR"].append(env.H_BR.channel_matrix)
    #if single user
    if env.K == 1 and env.P == 1:
      ch["h_BU"].append(env.h_BU[0].channel_matrix)
      ch["h_RU"].append(env.h_RU[0].channel_matrix)
      ch["h_BE"].append(env.h_BE[0].channel_matrix)
      ch["h_RE"].append(env.h_RE[0].channel_matrix)
    else:
      ch["h_BU"].append({f"user_{i}": env.h_BU[i].channel_matrix for i in range(env.K)})
      ch["h_RU"].append({f"user_{i}": env.h_RU[i].channel_matrix for i in range(env.K)})
      ch["h_BE"].append({f"eve_{i}": env.h_BE[i].channel_matrix for i in range(env.P)})
      ch["h_RE"].append({f"eve_{i}": env.h_RE[i].channel_matrix for i in range(env.P)})

# generate a series of random channels (seed)
def generate_channels(env_config, num_eps, seed):
  channels = {
  "H_BR" : [],
  "h_BU" : [],
  "h_RU" : [],
  "h_BE" : [],
  "h_RE" : [],
  }
  env = Environment(env_config, seed=seed)
  env.seed(seed)
  np.random.seed(seed)
  for i in range(num_eps):
    _ = env.reset()
    save_channels(env,channels)

  del env
  return channels

# generate a series of random channels (seed)
def generate_channels_rewrads(env_config, num_eps, seed):
  channels = {
  "H_BR" : [],
  "h_BU" : [],
  "h_RU" : [],
  "h_BE" : [],
  "h_RE" : [],
  }
  rewards = np.zeros(num_eps)
  env = Environment(env_config, seed=seed)
  env.seed(seed)
  np.random.seed(seed)
  for i in range(num_eps):
    _ = env.reset()
    rewards[i] = env._reward()
    save_channels(env,channels)

  del env
  return channels, rewards

# load generated channels from matlab
# TODO: load_channels()
 
def save2mat(channels, filename):
    for k,h in channels.items():
        mat_dict = {f'sample_{i+1}': h[i] for i in range(len(h))}
        #scipy.io.savemat(k,mat_dict)
        scipy.io.savemat(f"{filename}/{k}.mat", mat_dict)

def save2json(args, path):
    experiment = {}
    experiment["filename"] = f"{path}"
    experiment["policy"] = args.policy
    experiment["seed"] = args.seed
    experiment["num_eps"] = args.num_eps
    experiment["num_steps"] = args.num_steps_per_ep
    experiment["eval_time_steps"] = args.eval_time_steps
    experiment["save_channels"] = args.save_channels
    experiment["save_model"] = args.save_model
    experiment["load_model"] = args.load_model #file path
    experiment["verbose"] = args.verbose_t

    env = {}
    env["num_antennas"] = args.num_antennas
    env["num_RIS_elements"] = args.num_RIS_elements
    env["num_users"] = args.num_users
    env["num_eve"] = args.num_eve
    env["power_t"] = args.power_t
    env["awgn_var"] = args.awgn_var
    env["path_loss_0"] = args.path_loss_0
    env["path_loss_exponent"] = args.path_loss_exponent
    env["LoS"] = args.LoS
    env["without_RIS"] = args.without_RIS
    env["CSI"] = "Perfect" if not args.variant_CSI else "Imperfect"
    env["random_pos"] = args.random_pos
    env["positions"] = {
        "BS": args.BS_pos,
        "RIS": args.RIS_pos,
        "users": [150, 0] if not args.random_pos else generate_random_positions(args.num_users),
        "eves": [145, 0] if not args.random_pos else generate_random_positions(args.num_eve)
    },
    
    algo = {}
    algo["actor_lr"] = args.lr
    algo["critic_lr"] = args.critic_lr
    algo["critic_decay"] = args.decay
    algo["actor_decay"] = args.decay
    algo["discount"] = args.gamma
    algo["tau"] = args.tau
    algo["exploration_noise"] = args.exploration_noise
    algo["noise_scalar"] = args.noise_scalar
    
    buffer = {}
    buffer["batch_size"] = args.batch_size
    buffer["buffer_size"] = args.buffer_size
    buffer["prioritized"] = args.prioritized
    buffer["alpha"] = args.per_alpha
    buffer["beta_0"] = args.per_beta
    buffer["beta_annealed"] = args.per_beta_annealed
    buffer["importance_sampling"] = args.per_importance_sampling

    params = {
        "experiment": experiment,
        "env": env,
        "algo": algo,
        "buffer": buffer
    }

    params = json.dumps(params,indent=4)
    with open(f"{path}/params.json", "w") as f:
        f.write(params)

    return params

def create_new_experience(args):
    base = args.save_to
    if not os.path.exists(base): 
        os.mkdir(base)
    
    folders = ["Learning Curves", "Learning Figures", "MATLAB"]
    timestamp = (time.strftime('/%Y-%m-%d %H_%M_%S',time.localtime(time.time())))[1:]
    if args.prioritized :
        path=f"{base}/Prioritized_{args.experiment_type}_{timestamp}"
    else:
        path=f"{base}/{args.experiment_type}_{timestamp}"
    
    os.mkdir(path)
    for folder in folders:
        os.makedirs(f"{path}/{folder}/test")
        os.makedirs(f"{path}/{folder}/train")

    if args.save_model and not os.path.exists(f"{path}/Models"):
        os.makedirs(f"{path}/Models")
    
    return path