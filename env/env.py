import numpy as np
import matplotlib.pyplot as plt
from env.channel import Channel
from env.entities import *

import gymnasium as gym
from gymnasium import spaces
#from gymnasium.envs.registration import register


class Environment(gym.Env):
    def __init__(self,
                 env_config,
                 steps_per_ep=1000, #test only
                 seed=0): # frequency, coordinates (BS, RIS)

        super(Environment, self).__init__()

        self.if_dir_link = env_config["LoS"]
        self.if_with_RIS = not env_config["without_RIS"]
        self.M = env_config["num_antennas"]
        self.N = env_config["num_RIS_elements"]
        self.K = env_config["num_users"] #1
        self.P = env_config["num_eve"] #1
        self.awgn_var = env_config["awgn_var"]
        self.power_max = 10**(env_config["power_t"]/10)
        self.pl = env_config["path_loss_exponent"]
        self.PL_0 = env_config["path_loss_0"]
        random_pos = env_config["random_pos"]
        print(env_config["positions"])
        positions_u = env_config["positions"][0]["users"]
        positions_e = env_config["positions"][0]["eves"]
        self.CSI = env_config["CSI"]

        # Init Gym spaces
        self.action_space = spaces.Box(low=-5, high=5, shape=(self._get_action_dim(),), dtype=np.float64) # ~~
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._get_state_dim(),), dtype=np.float64)

        # 1. init entities: 1 BS , 1 RIS, 1 User, 1 Attacker
        # 1.1 init BS position and beamforming matrix
        self.BS = BS(
            coordinate= np.array(env_config["positions"][0]["BS"]),
            num_ant=self.M
        )

        # Initialize BS power to power_max
        self.power_factor = np.sqrt(self.power_max)/np.sqrt(self.BS.num_ant*self.K)
        # Initializing Beamforming vectors as ones
        self.BS.W = np.mat(np.ones((self.BS.num_ant, self.K),
                                   dtype=complex), dtype=complex) * self.power_factor
        self.BS.W_Pmax = np.real(np.trace(self.BS.W * self.BS.W.H))

        # 1.2 init RIS
        self.RIS = RIS(
            coordinate=np.array(env_config["positions"][0]["RIS"]), 
            num_elements=self.N
            )

        # generate random positions
        if random_pos :
            positions = positions_u #generate_random_positions(self.K)
        else :
            positions = [np.array([150,0])]
        print("user positions", positions)
        self.positions_u  = positions
        # 1.3 init users
        self.user_list = []
        for i in range(self.K):
            user = User(coordinate=positions[i], index=i , noise_power=self.awgn_var) 
            self.user_list.append(user)

        # generate random positions
        if random_pos :
            positions = positions_e  #generate_random_positions(self.P)
        else :
            positions = [np.array([145,0])]
        print("eve positions", positions)
        self.positions_e  = positions
        # 1.4 init attackers
        self.attacker_list = []
        for i in range(self.P):
            attacker = Attacker(coordinate=positions[i], index=i, noise_power=self.awgn_var) 
            attacker.rate = np.zeros((self.K)) #eve rate is a vector
            self.attacker_list.append(attacker)

        # 1.5 generate eavesdrop rate array (matrix), shape P x K
        self.eavesdrop_rate_array = np.zeros((self.P, self.K))

        # 2. init channels
        self.H_BR = Channel(transmitter=self.BS, reciever=self.RIS, 
                            path_loss_exp=2.2, PL0=self.PL_0)
        self.h_BU = [] # k
        self.h_RU = [] # k
        self.h_BE = [] # p
        self.h_RE = [] # p

        for user in self.user_list:
            self.h_BU.append(Channel(transmitter=self.BS, reciever=user, 
                                     path_loss_exp=self.pl, PL0=self.PL_0)) 
            self.h_RU.append(Channel(transmitter=self.RIS, reciever=user, 
                                     path_loss_exp=self.pl, PL0=self.PL_0))

        for attacker in self.attacker_list:
            self.h_BE.append(Channel(transmitter=self.BS, reciever=attacker, 
                                     path_loss_exp=self.pl, PL0=self.PL_0))
            self.h_RE.append(Channel(transmitter=self.RIS, reciever=attacker, 
                                     path_loss_exp=self.pl, PL0=self.PL_0))

        # 3. update users & attackers rate
        self._update_channel_rate()

        # 4. init episode
        self.episode_t = None
        self.state = None
        self.done = None
        self._max_episode_steps = steps_per_ep #np.inf
        self.info = {'episode': None}

        self.seed(seed)

    def seed(self,seed):
        np.random.seed(seed)

    def reset(self, *, seed=None, options=None, data=False, matlab=False, channels=None, eps_num=None): #add those in options later

        if seed != None:
            np.random.seed(seed)

        # 2. reset Beamforming & phase shift

        self.BS.W = np.mat(np.ones((self.BS.num_ant, self.K), dtype=complex), dtype=complex) #* self.power_factor
        self.BS.W_Pmax = np.real(np.trace(self.BS.W * self.BS.W.H))

        self.RIS.q = np.ones(self.RIS.num_ant, dtype=complex)
        self.RIS.Phi = np.mat(np.diag(self.RIS.q), dtype = complex)

        # 3. reset CSI
        if not data:
            self.H_BR.update_CSI()
            for h in self.h_RU + self.h_BU + self.h_RE + self.h_BE:
                h.update_CSI()
        elif matlab:
          self._upload_channels_mat(channels, eps_num)
        else :
          self._update_channels_data(channels,eps_num)

        # if NLOS
        #print(self.if_dir_link)
        if not self.if_dir_link:
          for h in self.h_BU + self.h_BE:
            h.channel_matrix = np.mat(np.zeros(shape = np.shape(h.channel_matrix)), dtype=complex)

        # 4. reset rates
        self._update_channel_rate()
        print("initial reward:",self._reward())

        # 5. reset episode timesteps
        self.episode_t = 0

        # initial action
        init_action_W = np.hstack((np.real(self.BS.W.reshape(1, -1)), np.imag(self.BS.W.reshape(1, -1))))
        init_action_q = np.hstack((np.real(self.RIS.q.reshape(1, -1)), np.imag(self.RIS.q.reshape(1, -1))))
        init_action = np.array(np.hstack((init_action_W, init_action_q)))[0]

        self.action = init_action

        self.state = self._observe()

        return self.state, self.info

    def step(self, action): 
        self.episode_t += 1

        if self.CSI == "Imperfect":
        #1. update CSI   
            self.H_BR.update_CSI()
            for h in self.h_RU + self.h_BU + self.h_RE + self.h_BE:
                h.update_CSI()
        # take into consideration with direct link (los) & without RIS
        # if NLOS
        if not self.if_dir_link:
          for h in self.h_BU + self.h_BE:
            h.channel_matrix = np.mat(np.zeros(shape = np.shape(h.channel_matrix)), dtype=complex)

        #2. update beamforming & phase shifts
        self.action = action#[0]

        W_real = self.action[0:self.M * self.K]
        W_imag = self.action[self.M * self.K : 2 * self.M * self.K]

        q_real = self.action[2 * self.M * self.K : 2 * self.M * self.K + self.N]
        q_imag = self.action[2 * self.M * self.K + self.N :]

        #q = np.exp(1j * self.action[2 * self.M * self.K : ] * np.pi)

        q = q_real + 1j * q_imag
        W = W_real.reshape(self.M, self.K) + 1j * W_imag.reshape(self.M, self.K)
        W = np.mat(W, dtype=complex)

        #Normalize q & W
        q = q / abs(q)

        #W normalisation
        if np.real(np.trace(W @ W.H)) > self.power_max: 
            current_power_t = np.sqrt(self.power_max) / np.sqrt(np.real(np.trace(W @ W.H)))
        else : current_power_t = 1

        Phi = np.mat(np.diag(q), dtype=complex)

        self.BS.W = W * current_power_t
        self.BS.W_Pmax = np.real(np.trace(self.BS.W * self.BS.W.H))

        self.RIS.q = q
        self.RIS.Phi = Phi

        #3. update channel rate
        self._update_channel_rate()

        #4. update state (get new state)
        new_state = self._observe()
        self.state = new_state

        #5. get reward
        reward = self._reward()

        #6. done (if old reward < new reward : not done)
        done = self.episode_t >= self._max_episode_steps

        # optimal
        snr = 0
        ssr = 0
        for i in range(self.K):
            snr += self._calculate_user_rate(i)[1]
            ssr += self._secrecy_rate_of_user_k(i)

        truncated = False

        return new_state, reward, done, truncated, self.info

    def render(self, filename=""):        
        i = 0
        for pos in self.positions_u:
            i += 1
            plt.scatter(pos[0],pos[1], color='blue')
            plt.text(pos[0],pos[1], f'U{i}', color='blue')

        i = 0
        for pos in self.positions_e:
            i += 1
            plt.scatter(pos[0],pos[1], color='red')
            plt.text(pos[0],pos[1], f'E{i}', color='red')

        #RIS
        plt.scatter(self.RIS.coordinate[0], self.RIS.coordinate[1], color='green')
        plt.text(self.RIS.coordinate[0], self.RIS.coordinate[1], 'RIS', color='green')

        #BS
        plt.scatter(self.BS.coordinate[0], self.BS.coordinate[1], color='black')
        plt.text(self.BS.coordinate[0], self.BS.coordinate[1], 'BS')
        
        plt.scatter(200, 200, color='white')
        plt.title('Grid with Users, Eavesdroppers, RIS, and BS')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(visible=True, which='both', linestyle='--', lw=0.5)
        plt.savefig(f"{filename}/setup.png")
        plt.show()
        
    # Not ready yet
    def draw_beam(self):

        F = self.BS.W
        magnitudes = np.abs(F)
        phases = np.angle(F, deg=True)  # Convert phases to degrees

        # Create a list of angles (θ)
        angles = np.linspace(-180, 180, 360)  # Covering -180 to 180 degrees

        # Calculate the resulting beamforming vector at each angle
        beamforming_vector = [np.dot(magnitudes, np.exp(1j * np.deg2rad(phases))) for _ in angles]

        # Create a polar plot
        plt.figure(figsize=(8, 8))
        plt.polar(np.deg2rad(angles), np.abs(beamforming_vector))
        plt.title('Beamforming Vector')
        plt.show()

    def close(self):
        pass

    ### Get reward
    def _reward(self): # secrecy rate calculation
        # les contraintes dans la fonction de récompense
        # probably i should drop |q| = 1 & tr(W*W.H)?
        reward = 0
        #is it average reward (/self.K)
        #or sum ? => I said sum secrecy rate
        for user in self.user_list:
            reward += user.secrecy_rate

        # I should penalize it if the power & |q| constraints is not correct & if  it has degraded in results (threshold)
        return reward # can be stuck in a local optimum (reward function doesn't improve or degrade)
    #np.tanh(reward)

    ### Get observation (state)
    def _observe(self): # returns the state tuple (CSI, R(k), R(p), action)
        """
        returns current state
        """
        # users and attackers comprehensive channel at (t)
        # Channels (h_k & h_p)
        entities = self.user_list + self.attacker_list
        for i in range(len(entities)):
          h_r , h_i = np.real(entities[i].channel).reshape(1,-1) , np.imag(entities[i].channel).reshape(1, -1)
          var = np.hstack((h_r, h_i))
          if i != 0 :
            comprehensive_channels = np.hstack((tmp, var))
          else :
            comprehensive_channels = var
          tmp = comprehensive_channels

        comprehensive_channels = np.array(comprehensive_channels)[0]

        # rates
        sum_user_rate = 0
        sum_secrecy_rate = 0
        for user in self.user_list:
            sum_user_rate += user.rate
            sum_secrecy_rate += user.secrecy_rate
        # 1 x K array representing eavesdropped rate of each user by the ensemble of p attackers
        sum_leaked_rate = np.sum(np.sum(self.eavesdrop_rate_array, axis=0)).reshape(1, -1)
        rates = np.array([sum_user_rate, sum_secrecy_rate]).reshape(1, -1)
        rates = np.hstack((rates, sum_leaked_rate))[0]

        # action (beamforming + phase shifts (q)) at (t -1)
        action = self.action

        state = np.hstack((comprehensive_channels, action, rates)) # I removed rate from state_dim
        return np.array(state)

    ### Other helper methods (Guo2021)
    def _comprehensive_channel_user(self, k):
        H_BR = self.H_BR.channel_matrix
        h_BU_k = self.h_BU[k].channel_matrix
        h_RU_k = self.h_RU[k].channel_matrix
        Phi = self.RIS.Phi
        return h_RU_k @ Phi @ H_BR + h_BU_k

    def _comprehensive_channel_attacker(self, p):
        H_BR = self.H_BR.channel_matrix
        h_BE_p = self.h_BE[p].channel_matrix
        h_RE_p = self.h_RE[p].channel_matrix
        Phi = self.RIS.Phi
        return h_RE_p @ Phi @ H_BR + h_BE_p

    def _update_channel_rate(self): # calculating rate / channel capacity of ue/eve

        # 1 calculate eavesdrop rate
        for attacker in self.attacker_list:
            attacker.rate = self._calculate_attacker_rate_array(attacker.index)
            self.eavesdrop_rate_array[attacker.index, :] = attacker.rate
            # remember to update channel of attacker
            attacker.channel = self._comprehensive_channel_attacker(attacker.index)

        #2 calculate user rate
        for user in self.user_list:
            user.rate, _ = self._calculate_user_rate(user.index)
            # 3 calculate user secrecy rate
            user.secrecy_rate = self._secrecy_rate_of_user_k(user.index)
            # remember to update channel of user
            user.channel = self._comprehensive_channel_user(user.index)

    def _calculate_user_rate(self, k):
        """
        returns user rate with/without interference
        """
        noise_power = 10**(self.user_list[k].noise_power/10)
        H_BR = self.H_BR.channel_matrix
        h_BU_k = self.h_BU[k].channel_matrix
        h_RU_k = self.h_RU[k].channel_matrix
        Phi = self.RIS.Phi
        w_k = self.BS.W[:, k]

        # W = np.hstack((self.BS.W[:, :k], self.BS.W[:, k+1:]))
        W =  np.delete(self.BS.W, k, axis=1) #W without w_k

        # la somme des taux sauf user k (interférence)
        interference = np.sum(np.array(np.abs((h_RU_k @ Phi @ H_BR + h_BU_k) @ W))[0]**2)

        alpha_k = np.abs((h_RU_k @ Phi @ H_BR + h_BU_k) @ w_k) ** 2
        beta_k = interference + noise_power
        rate_SINR = np.sum( np.log2(1 + alpha_k/beta_k) ) #to remove additional singletons

        # CASE with no interference
        rate_SNR = np.sum( np.log2(1 + alpha_k/noise_power) ) #to remove additional singletons


        return rate_SINR , rate_SNR

    def _calculate_attacker_rate_array(self, p):
        """
        returns a rate array of eavesdropper p listening on each channel of k users
        """
        eve_rate = []

        noise_power = 10**(self.attacker_list[p].noise_power/10)
        H_BR = self.H_BR.channel_matrix
        h_BE_p = self.h_BE[p].channel_matrix
        h_RE_p = self.h_RE[p].channel_matrix
        Phi = self.RIS.Phi
        W = self.BS.W

        # loop for each k in K
        for i , w_k in enumerate(W.T):
            alpha_p = abs((h_RE_p @ Phi @ H_BR + h_BE_p) @ w_k.T) ** 2
            W_removed = np.delete(W, i, axis=1)
            interference = np.sum(np.array(np.abs((h_RE_p @ Phi @ H_BR + h_BE_p) @ W_removed))[0]**2)
            beta_p = interference + noise_power
            eve_rate.append(np.sum(np.log2(1 + abs(alpha_p/beta_p))))

        return np.array(eve_rate)

    def _secrecy_rate_of_user_k(self, k):
        user = self.user_list[k]
        R_k = user.rate
        R_k_maxeavesdrop = max(self.eavesdrop_rate_array[:, k])
        return max(0, R_k - R_k_maxeavesdrop)

    def _get_action_dim(self):
        # TODO : calculate dimension (Re+Im)
        # beamforming + phase shift dim
        action_dim = 2 * (self.M * self.K) +  self.N * 2
        return action_dim

    def _get_state_dim(self):
        # TODO : calculate dimension (Re+Im)
        # users & attackers comprehensive channel
        channel_size = 2 * (self.K + self.P) * self.M

        # sum_rate , sum_secrecy_rate, sum_rate of each user k eavesdropped on by P eavesdroppers (leaked message from each user)? ?
        rate_size = 1 + 1 + 1 # self.K # doubtful ida self.P/self.K (for mean time leave it to one) or (+1)
        #rate_size = 0 # decided to remove it cuz i can calculate it based on CSI & action ?
        # current rate size = self.K + self.P * self.K (a lot)

        state_dim = channel_size + rate_size + self._get_action_dim()
        return state_dim

    def _update_channels_data(self,channels, episode_num):
        #loaded channels from .mat file
        #single user case
        t = episode_num
        self.H_BR.channel_matrix = channels["H_BR"][t]
        if (self.K == 1 and self.P == 1):
            self.h_BU[0].channel_matrix = channels["h_BU"][t]
            self.h_BE[0].channel_matrix = channels["h_BE"][t]
            self.h_RU[0].channel_matrix = channels["h_RU"][t]
            self.h_RE[0].channel_matrix = channels["h_RE"][t]
        else :
            for i in range(self.K):
                self.h_BU[i].channel_matrix = channels["h_BU"][t][f"user_{i}"]
                self.h_RU[i].channel_matrix = channels["h_RU"][t][f"user_{i}"]
            for i in range(self.P):
                self.h_BE[i].channel_matrix = channels["h_BE"][t][f"eve_{i}"]
                self.h_RE[i].channel_matrix = channels["h_RE"][t][f"eve_{i}"]
    
    def _upload_channels_mat(self,channels, episode_num):
        #loaded channels from .mat file
        #single user case
        t = episode_num
        self.H_BR.channel_matrix = channels["H_BR"][:,:,t]
        self.h_BU[0].channel_matrix = channels["h_BU"][:,:,t]
        self.h_BE[0].channel_matrix = channels["h_BE"][:,:,t]
        self.h_RU[0].channel_matrix = channels["h_RU"][:,:,t]
        self.h_RE[0].channel_matrix = channels["h_RE"][:,:,t]