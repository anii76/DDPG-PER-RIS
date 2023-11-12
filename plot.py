#save plots & figs
import numpy as np
import matplotlib.pyplot as plt


def compute_avg_reward(reward):
    avg_reward = np.zeros_like(reward)

    for i in range(len(reward)):
        avg_reward[i] = np.sum(reward[:(i + 1)]) / (i + 1)

    return avg_reward


def plot(path,test):
    if test :
        ext = "test"
    else:
        ext = "train"

    base_dir = f"{path}/Learning Curves/{ext}"
    fig_dir = f"{path}/Learning Figures/{ext}"


    instant_rewards = np.load(f"{base_dir}/training_episodes.npy", allow_pickle=True).squeeze()
    instant_rewards = instant_rewards[0:]


    loss = np.load(f"{base_dir}/training_loss.npy", allow_pickle=True).squeeze()

    loss_a , loss_c = loss[:,:,0] , loss[:,:,1]


    ##training
    #reawrds in all episodes
    avg_r = [np.mean(instant_rewards[i]) for i in range(len(instant_rewards))]
    plt.figure(0)
    plt.plot(avg_r, label="Instant reward")
    plt.ylabel("Rewards (SR)")
    plt.xlabel("Episodes")
    plt.title("Mean episode rewards")
    plt.savefig(f"{fig_dir}/mean_r.png")

    #rewards per episode
    rng = len(instant_rewards)
    for i in range(rng):
        avg_r = compute_avg_reward(instant_rewards[i])
        plt.figure(i+1)
        plt.plot(instant_rewards[i], label="Instant reward")
        plt.plot(avg_r, label="Average reward")
        plt.title("Episode ("+str(i)+")")
        plt.ylabel("Rewards (SR)")
        plt.xlabel("Steps")
        plt.savefig(f"{fig_dir}/reward_ep{i}.png")


    # loss in all episodes
    avg_c = [np.mean(l) for l in loss_c]
    avg_a = [np.mean(l) for l in loss_a]
    plt.figure(len(instant_rewards)+2)
    plt.plot(avg_c)
    plt.ylabel("Critic loss")
    plt.xlabel("Episodes")
    plt.title("Mean episode loss")
    plt.savefig(f"{fig_dir}/mean_critic_loss.png")

    plt.plot(avg_a)
    plt.ylabel("Actor loss")
    plt.xlabel("Episodes")
    plt.title("Mean episode loss")
    plt.savefig(f"{fig_dir}/mean_actor_loss.png")

    # loss
    rng = len(loss_c)
    for i in range(rng):
        avg_r = compute_avg_reward(loss_c[i])

        plt.figure(len(instant_rewards)+3+i)
        plt.plot(avg_r)
        plt.title("Episode ("+str(i)+")")
        plt.ylabel("Critic loss")
        plt.xlabel("Steps")
        plt.savefig(f"{fig_dir}/loss_ep{i}.png")

    plt.close()