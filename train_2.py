import json
import time
import torch
import numpy as np
import multiprocessing
from tensorboardX import SummaryWriter

from env.env import *
from algo.ddpg import DDPG
from algo.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer
from algo.per_utils import LinearSchedule
from utils import *

def whiten(state):
    return (state - np.mean(state)) / np.std(state)


def train(params):
    params = json.loads(params)
    env_config = params["env"]
    ddpg_config = params["algo"]
    buffer_config = params["buffer"]
    experiment = params["experiment"]
    
    # Initialize experiment params
    save_channels = experiment["save_channels"]
    save_model = experiment["save_model"]
    filename = experiment["filename"]
    seed = experiment["seed"]
    num_eps = experiment["num_eps"] 
    num_steps_per_ep = experiment["num_steps"]
    total_steps = num_eps * num_steps_per_ep
    load_model = experiment["load_model"] != ""
    verbose = experiment["verbose"]
    
    
    # Create environment
    env = Environment(
        env_config, 
        steps_per_ep=num_steps_per_ep,
        seed=seed)
    
    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    #render env
    multiprocessing.Process(target=env.render, args=(filename,)).start() 


    # Initialize agent
    state_dim = env._get_state_dim()
    action_dim = env._get_action_dim()
    ddpg_config["state_dim"]= state_dim
    ddpg_config["action_dim"] = action_dim
    ddpg_config["device"] = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    ddpg_config["max_action"] = 1

    agent = DDPG(**ddpg_config)

    if load_model:
        agent.load(f"{filename}/Models/")
    
    #Initialize buffer
    buffer_size = buffer_config["buffer_size"]
    batch_size = buffer_config["batch_size"]
    prioritized = buffer_config["prioritized"]
    prioritized_replay_alpha = buffer_config["alpha"]
    prioritized_replay_beta0 = buffer_config["beta_0"]
    prioritized_replay_beta_iters = None
    prioritized_replay_eps = 1e-6
    if prioritized :
        if buffer_config["beta_annealed"] :
            prioritized_replay_beta_iters = total_steps
            # Create annealing schedule (for beta value)
            beta_schedule = LinearSchedule(
                prioritized_replay_beta_iters, 
                initial_p=prioritized_replay_beta0, 
                final_p=1.0)
        
        replay_buffer = PrioritizedReplayBuffer(buffer_size, prioritized_replay_alpha)
    else:
        replay_buffer = UniformReplayBuffer(state_dim, action_dim, buffer_size)


    # generate fixed dataset
    channels = generate_channels(env_config, num_eps,seed)
    if save_channels :
        save2mat(channels, filename+"/MATLAB")
    

    # SET UP NOISE LINEAR SCHEDUALAR
    exploration_fraction = 0.1
    total_timesteps = num_steps_per_ep # all over episodes
    exploration_final_eps = 0.001 
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=  int(exploration_fraction * total_timesteps),
                                    initial_p=agent.noise_scalar,
                                    final_p=exploration_final_eps)

    # For stats
    instant_rewards = []
    episode_rewards = []
    loss = []
    max_rewards = []
    max_reward = 0
    users_rate = []
    eves_rate = []
    beta_values = []
    test_scores = []
    ssr = []
    writer = SummaryWriter(filename)

    #Training loop
    episode_t = 0
    total_t = 0
    test_time = 0
    episode = 0
    test_freq = 2000
    checkpoint = num_steps_per_ep
    plotting_interval = 1000 #try it out
    start_iter = 0

    #to resume uncompleted trainig
    if load_model:
        print("......................................")
        print("....... Loading previous setup .......")
        print("......................................")
        instant_rewards = np.load(f"{filename}/Learning Curves/train/training_episodes.npy").tolist()
        users_rate = np.load(f"{filename}/Learning Curves/train/users_rate.npy").tolist()
        eves_rate = np.load(f"{filename}/Learning Curves/train/eves_rate.npy").tolist()
        ssr = np.load(f"{filename}/Learning Curves/train/secrecy_rate.npy").tolist()
        loss = np.load(f"{filename}/Learning Curves/train/training_loss.npy").tolist()
        start_iter = len(instant_rewards)
    
    
    print('..........................................')
    print('Running: '+ filename)
    print('..........................................')
    print("Total Steps :",total_steps)

    #test_scores = [test_policy(env=env, eps=5, channels=channels)]


    t_0 = time.time()

    decay_factor = 0.9998


    for episode in range(start_iter,num_eps):
        s, _ = env.reset(data=True, channels=channels ,eps_num=episode)
        s = whiten(s)
        done = False

        episode_rewards = []
        episode_t = 0
        episode_r = 0
        max_reward = 0
        ep_loss = []

        #if episode == 1: #for test berk
        #    num_steps_per_ep = 40000    for   ./Results/DDPG_linear_anneal_noise_10_critic_3_2023-08-25 22_25_38
        #    env._max_episode_steps = num_steps_per_ep

        #Initialize noise
        #agent.noise.sigma = agent.noise_scalar
        
        #each episode with a diff noise :33
        #noise_rate = exploration.value(episode_t)
        #print(noise_rate)

        #noise_scalar= agent.noise_scalar
        for episode_t in range(num_steps_per_ep):
            total_t += 1

            # Exploration noise annealed value
            # noise_scalar = exploration.value(episode_t)

            #Given current state, get action
            a = agent.select_action(np.array(s))

            #Apply exploration noise to action
            a = a #+ np.random.normal(0, 0.5, size=action_dim)

            #Using action, take step in environment, observe new state, reward and episode status
            s_new, r, done, _, _ = env.step(a)

            done = 1.0 if episode_t == num_steps_per_ep - 1 else float(done)

            #if r>3:
            #    done = 1.0

            # Store data in the experience replay buffer
            replay_buffer.add(s, a, r, s_new, done)

            if episode > 128*3 : #not (episode_t < 128 and episode == 0): #khalih yt3lm for one episode to fill the buffer
                if not prioritized:
                    ep_loss.append(agent.update_parameters(replay_buffer, batch_size))
                
                else : #if total_t % batch_size == 0: #like original paper [alternatives : if episode_t > buffer_size]
                    
                    #set beta value used for importance sampling weights
                    if buffer_config["beta_annealed"]:
                        beta_value = beta_schedule.value(total_t) 
                    else :
                        beta_value = prioritized_replay_beta0 
                    beta_values.append(beta_value)
                    ep_loss.append(agent.train(replay_buffer, prioritized, beta_value, prioritized_replay_eps, 
                                        batch_size, batch_size, buffer_config["importance_sampling"]))
                    
                writer.add_scalar('/Loss/critic_loss', ep_loss[-1][1], total_t)
                writer.add_scalar('/Loss/actor_loss', ep_loss[-1][0], total_t)

            s = s_new
            s = whiten(s)

            #reduce noise
            #if episode > 0 : noise_scalar = (1 - 0.005)/num_steps_per_ep
            noise_scalar = agent.noise_scalar*(0.9998**episode_t)
            agent.noise.sigma = noise_scalar
            ########################

            episode_r += r
            episode_rewards.append(r)

            if max_reward < r:
                max_reward = r

            users_rate.append([env.user_list[i].rate for i in range(env.K)])
            ssr.append(sum([env.user_list[i].secrcy_rate for i in range(env.K)]))
            eves_rate.append([env.attacker_list[i].rate for i in range(env.P)])

            if verbose :
                print("Episode "+str(episode)+ " Step "+str(episode_t)+" Max reward: "+str(max_reward)+" Step Reward: "+str(r))

            writer.add_scalar('/Rewards/total_rewards', r, total_t)
            writer.add_scalar('/Rewards/Cumulative_episode_rewards', sum(episode_rewards)/(episode_t+1), total_t)
            

            if done:
                print("done", episode_t)
                instant_rewards.append(episode_rewards)
                loss.append(ep_loss)
                if not prioritized : loss.append(ep_loss)

                writer.add_scalar('/Rewards/mean_rewards', np.mean(episode_rewards), total_t)
                #writer.add_scalar('/Rewards/Cumulative_mean_rewards', sum(np.mean(instant_rewards,axis=1))/(episode+1), total_t)

                # save episode
                print("...............checkpoint............")
                print("Elapsed time :", time.time() - t_0)
                np.save(f"{filename}/Learning Curves/train/training_episodes", instant_rewards)
                np.save(f"{filename}/Learning Curves/train/users_rate", users_rate)
                np.save(f"{filename}/Learning Curves/train/secrecy_rate", ssr)
                np.save(f"{filename}/Learning Curves/train/eves_rate", eves_rate)
                np.save(f"{filename}/Learning Curves/train/training_loss", loss)
                
                #save_model
                if save_model:
                    agent.save(f'{filename}/Models/')
                
                #break

        print("Episode: "+str(episode)+ " Episode Reward: "+str(episode_r)+" Max Reward: "+str(max_reward)+" Runtime: "+str(int(time.time() - t_0)))
        if (episode_r == 0 or max_reward == 0):pass
#            print([sum(users_rate[i][-num_steps_per_ep:-1] for i in range(env.K))])
#            print([sum(eves_rate[i][-num_steps_per_ep:-1] for i in range(env.P))])

     #save stats to not fall for same trap everytime
    print("...............saving finals............")
    print("Elapsed time :", time.time() - t_0)
    np.save(f"{filename}/Learning Curves/train/training_episodes", instant_rewards)
    np.save(f"{filename}/Learning Curves/train/users_rate", users_rate)
    np.save(f"{filename}/Learning Curves/train/secrecy_rate", ssr)
    np.save(f"{filename}/Learning Curves/train/eves_rate", eves_rate)
    np.save(f"{filename}/Learning Curves/train/training_loss", loss)
    
    #save_model
    if save_model:
        agent.save(f'{filename}/Models/')
    
    if prioritized: print("beta values", beta_values)

