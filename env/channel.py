import numpy as np
import math, cmath

def smooth_channel(previous_channel, current_channel, alpha=0.2):
    # Apply smoothing to the current channel using the previous channel
    smoothed_channel = alpha * current_channel + (1 - alpha) * previous_channel
    return smoothed_channel

# Rayleigh / Rician / mmWave
def rayleigh(N_t, N_r):
    h = np.random.normal(0, np.sqrt(0.5), (N_r, N_t)) + 1j * np.random.normal(0, np.sqrt(0.5),(N_r, N_t))
    return h

def rician(N_t, N_r, K):
    #aod = np.random.uniform(0, 2*np.pi)
    #aoa = np.random.uniform(0, 2*np.pi)

    #transmitter
    #a_t = np.mat(np.ones(shape=(N_t,1)), dtype=complex)
    #for i in range(N_t):
        #a_t[i,0] =  cmath.exp(1j*np.pi*i*np.sin(aod))
    #reciever
    #a_r = np.mat(np.ones(shape=(N_r,1)), dtype=complex)
    #for i in range(N_r):
        #a_r[i,0] = cmath.exp(1j*np.pi*i*np.sin(aoa))

    #for multiple antennas berk / for 1 user one angle ça suffit
    #Los = a_r * a_t.H #if N_r > 1 else a_t
    #Nlos = np.mat((np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t)), dtype=complex) / np.sqrt(2)

    mean = np.sqrt(K / (1 + K))
    sigma = np.sqrt(1 / 2*(1 + K))

    channel_matrix = (mean + sigma * np.random.randn(N_r, N_t)) + 1j * (mean + sigma * np.random.randn(N_r, N_t))

    return channel_matrix


class Channel(object):
    def __init__(self, transmitter, reciever, PL0=-30, path_loss_exp=2.0):
        self.transmitter = transmitter
        self.reciever = reciever

        # init & update path loss
        self.pl = path_loss_exp
        self.PL0 = PL0
        self.path_loss_normal = self.get_channel_path_loss() 

        # init & update channel CSI matrix
        self.channel_matrix = self.get_estimated_channel_matrix()

    def get_channel_path_loss(self):
        pl = self.pl # pathloss exponent
        distance = np.linalg.norm(self.transmitter.coordinate - self.reciever.coordinate)
        PL0 = self.PL0 
        path_loss = PL0 - 10*pl*np.log10(distance)
        path_loss = 10**(path_loss/10)

        return path_loss

    def get_estimated_channel_matrix(self):
        N_t = self.transmitter.num_ant
        N_r = self.reciever.num_ant
        PL = self.get_channel_path_loss()
        #channel_matrix = math.pow(PL, 0.5) * array_response
        channel_matrix = np.sqrt(PL) * rician(N_t, N_r,K=1)

        return rician(N_t, N_r,K=1)#channel_matrix

    def update_CSI(self):
        #init & update path loss
        self.path_loss_normal = self.get_channel_path_loss()
        #self.path_loss_dB = normal_to_dB(self.path_loss_normal)
        
        # init & update channel CSI matrix
        #old channel
        prev_channel = self.channel_matrix
        self.channel_matrix = smooth_channel(previous_channel=prev_channel, 
                                             current_channel=self.get_estimated_channel_matrix())

