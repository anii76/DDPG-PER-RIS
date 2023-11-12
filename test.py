import json
import numpy as np
from env.env import *
from algo.ddpg import DDPG
import torch

from utils import generate_channels


def whiten(state):
    return (state - np.mean(state)) / np.std(state)

# test policy
#TODO: add save rewards & loss & rates 
def test_policy(policy, env, eps, upload_channels=False, channels=None, filename=None): #same dataset as train
    print("****************Evaluation********************")
    episode_rewards = np.zeros(eps) #same number of eps as training
    eves_rate = [] 
    users_rate = []
    max_rewards = []
    for i in range(eps):
        max_reward = 0
        if upload_channels:
          s, _ = env.reset(data=True, channels=channels, eps_num=i)
        else :
          s, _ = env.reset()

        s = whiten(s)

        done = False

        while not done:
            #get action from policy
            a = policy.select_action(s, test=True)

            #observe new state, reward and whether the episode is done
            s_new, r, done, _, _ = env.step(a)

            #add reward to episode reward
            episode_rewards[i] += r

            #max rewards
            if r > max_reward:
               max_reward = r
            
            users_rate.append([env.user_list[i].rate for i in range(env.K)])
            eves_rate.append([env.attacker_list[i].rate for i in range(env.P)])

            #update states
            s = s_new

            s = whiten(s)
      
        max_rewards.append(max_reward)
        print("Episode: "+str(i)+ " Episode Reward: "+str(episode_rewards[i])+" Max Reward: "+str(max_reward)) 
        np.save(f"{filename}/Learning Curves/test/training_episodes", episode_rewards)
        np.save(f"{filename}/Learning Curves/test/users_rate", users_rate)
        np.save(f"{filename}/Learning Curves/test/eves_rate", eves_rate)


    #calculate average episode reward over eps tests
    print("evaluation score all episodes", np.mean(episode_rewards)) #all episodes
    print("evaluation mean reward", np.mean(episode_rewards)/env._max_episode_steps)
    return episode_rewards, users_rate, eves_rate, 

#stupid :))
def test_one(agent, env, eps_num, upload_channels=False, channels=None):
    if upload_channels:
          s, _ = env.reset(data=True, channels=channels, eps_num=eps_num)
    else :
          s, _ = env.reset()
    s = whiten(s)
    a = agent.select_action(s, test=True)
    s_new, r, done, _, _ = env.step(a)
    print(r)
    return r


def test(params):
  # environment setup
  params = json.loads(params)
  env_config = params["env"]
  ddpg_config = params["algo"]
  buffer_config = params["buffer"]
  experiment = params["experiment"]

  # Initialize experiment params
  load_channels = experiment["save_channels"]
  load_model = experiment["load_model"]
  filename = experiment["filename"]
  seed = experiment["seed"]
  num_eps = experiment["num_eps"] 
  num_steps_per_ep = experiment["num_steps"]
  total_steps = num_eps * num_steps_per_ep
   
  # if load_channels/matlab
  if load_channels:
     pass #for now work with the same seed , later import matlab channels
  channels = generate_channels(env_config, num_eps,seed)
  
  # Create environment
  env = Environment(
      env_config, 
      steps_per_ep=num_steps_per_ep,
      seed=seed)
  
  # Set seeds
  env.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)

  # Initialize agent
  state_dim = env._get_state_dim()
  action_dim = env._get_action_dim()
  ddpg_config["state_dim"]= state_dim
  ddpg_config["action_dim"] = action_dim
  ddpg_config["device"] = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
  ddpg_config["max_action"] = 1

  agent = DDPG(**ddpg_config)

  #Init DDPG & load model
  
  agent.load(f"{filename}/Models/")

  # test_scores
  print('..........................................')
  print('Running Tests of: '+ filename)
  print('..........................................')
  print("Total Steps :",total_steps)

  instant_rewards, users_rate, eves_rate = test_policy(
     policy=agent, env=env, eps=num_eps, upload_channels=True ,channels=channels, filename=filename)

  # save test scores
  np.save(f"{filename}/Learning Curves/test/training_episodes", instant_rewards)
  np.save(f"{filename}/Learning Curves/test/users_rate", users_rate)
  np.save(f"{filename}/Learning Curves/test/eves_rate", eves_rate)

  