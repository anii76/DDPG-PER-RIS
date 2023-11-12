import json
import time
import torch
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

from env.env import *
from algo.ddpg import DDPG
from algo.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer
from algo.per_utils import LinearSchedule
from utils import *

def whiten(state):
    return (state - np.mean(state)) / np.std(state)

def test_policy(policy, env, eps, upload_channels=False, channels=None):
    print("****************Evaluation********************")
    episode_rewards = np.zeros(eps)
    for i in range(eps):

        if upload_channels:
          s, _ = env.reset(data=True, channels=channels, eps_num=i)
        else :
          s, _ = env.reset()

        s = whiten(s)

        done = False

        while not done:
            #get action from policy
            a = policy.select_action(s)

            #observe new state, reward and whether the episode is done
            s_new, r, done, _, _ = env.step(a)

            #add reward to episode reward
            episode_rewards[i] += r

            #update states
            s = s_new

            s = whiten(s)

    #calculate average episode reward over eps tests
    print("evaluation score", np.mean(episode_rewards))
    return np.mean(episode_rewards) / env._max_episode_steps

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

    if experiment["load_model"] != "":
        agent.load(f"{filename}/Models")
    
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

    #Training loop
    episode_t = 0
    total_t = 0
    test_time = 0
    episode = 0
    test_freq = 2000
    checkpoint = num_steps_per_ep
    plotting_interval = 1000 #try it out
    
    
    print('..........................................')
    print('Running: '+ filename)
    print('..........................................')
    print("Total Steps :",total_steps)
    verbose = True
    done = True

    #test_scores = [test_policy(env=env, eps=5, channels=channels)]


    t_0 = time.time()

    while total_t < total_steps:

        if done:

            if total_t != 0:
                if verbose:
                    print("Total Time Steps: "+str(total_t)+ " Episode Reward: "+str(episode_r)+" Max Reward: "+str(max_reward)+" Runtime: "+str(int(time.time() - t_0)))

                #set beta value used for importance sampling weights
                beta_value = 0
                if prioritized:
                    beta_value = beta_schedule.value(total_t) #prioritized_replay_beta0 #
                    beta_values.append(beta_value)

                #train DDPG
                loss.append(agent.train(replay_buffer, prioritized, beta_value, prioritized_replay_eps, 
                                        episode_t, batch_size, buffer_config["importance_sampling"]))


                max_rewards.append(max_reward)
                instant_rewards.append(episode_rewards)

            #Checkpoint
            #Check if we need to need to save DDPG
            if total_t % checkpoint == 0:
                #save stats to not fall for same trap everytime
                print("...............checkpoint............")
                t_f = time.time() - t_0
                print(f"Elapsed time :{t_f:.3f}")
                np.save(f"{filename}/Learning Curves/train/training_episodes", instant_rewards)
                np.save(f"{filename}/Learning Curves/train/users_rate", users_rate)
                np.save(f"{filename}/Learning Curves/train/eves_rate", eves_rate)
                np.save(f"{filename}/Learning Curves/train/training_loss", loss)
                
                #save_model
                if save_model:
                    agent.save(f'{filename}/Models/')

            # plotting time
            #if total_t % plotting_interval == 0:
            #    _plot(
            #        total_t, 
            #        episode_rewards, 
            #        actor_losses=[loss[i][0] for i in range(len(loss))], 
            #        critic_losses=[loss[i][1] for i in range(len(loss))],
            #    )

            #Check if we need to need to test DDPG
            #if test_time >= test_freq :
                #test_scores.append(test_policy(agent,env, 3,True, test_channels))
                #np.save(f"{path}/Learning Curves/test_scores/%s" % (file_name), test_scores)


            #reset environment
            #get intial state
            #reset episode statistics
            print("Episode : ",episode)
            s, _ = env.reset(data=True, channels=channels ,eps_num=episode)
            s = whiten(s)
            done = False

            #save genrated channels
            #save_channels(env, ch)

            episode_rewards = []
            episode_t = 0
            episode_r = 0
            max_reward = 0
            episode += 1

        #Given current state, get action
        a = agent.select_action(np.array(s))
        #Apply exploration noise to action
        a = a #+ np.random.normal(0, 0.1, size=action_dim)

        #Using action, take step in environment, observe new state, reward and episode status
        s_new, r, done, _, _ = env.step(a)
        done_bool = 0 if episode_t + 1 == env._max_episode_steps else float(done)
        episode_r += r
        episode_rewards.append(r)

        if max_reward < r:
            max_reward = r

        users_rate.append([env.user_list[i].rate for i in range(env.K)])
        eves_rate.append([env.attacker_list[i].rate for i in range(env.P)])

        #print(f"Time step: {episode_t} Episode Num: {episode} Reward: {r:.3f}")
        # Store data in replay buffer
        if not prioritized:
            replay_buffer.add(s, a, r, s_new, done_bool)
        else:
            replay_buffer.add(s, a, r, s_new, done_bool)

        #update state and episode statistics
        s = s_new
        s = whiten(s)
        episode_t += 1
        total_t += 1
        test_time += 1

    print("Total Time Steps: "+str(total_t)+ " Episode Reward: "+str(episode_r)+" Max Reward: "+str(max_reward)+" Runtime: "+str(int(time.time() - t_0)))
    instant_rewards.append(episode_rewards)
    max_rewards.append(max_reward)

     #save stats to not fall for same trap everytime
    print("...............saving finals............")
    print("Elapsed time :", time.time() - t_0)
    np.save(f"{filename}/Learning Curves/train/training_episodes", instant_rewards)
    np.save(f"{filename}/Learning Curves/train/users_rate", users_rate)
    np.save(f"{filename}/Learning Curves/train/eves_rate", eves_rate)
    np.save(f"{filename}/Learning Curves/train/training_loss", loss)
    
    #save_model
    if save_model:
        agent.save(f'{filename}/')




def _plot(
        frame_idx, 
        scores, 
        actor_losses, 
        critic_losses, 
    ):
        """Plot the training progresses."""
        def subplot(loc: int, title: str, values):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]
        
        clear_output(True)
        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()

