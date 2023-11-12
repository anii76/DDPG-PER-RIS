import numpy as np
import math as m

class Entity(object):
    def __init__(self, coordinate, index, num_ant):
        self.coordinate = coordinate
        self.num_ant = num_ant
        self.index = index

    def reset(self, coordinate):
        self.coordinate = coordinate


class User(Entity):
    def __init__(self, coordinate, index, noise_power, num_ant = 1):
        super().__init__(coordinate, index ,num_ant)
        self.noise_power = noise_power 
        self.rate = 0 # capacity
        self.channel = 0
        self.secrcy_rate = 0



class Attacker(Entity):
    def __init__(self, coordinate, index, noise_power, num_ant = 1):
        super().__init__(coordinate, index ,num_ant)
        self.noise_power = noise_power 
        self.rate = 0 # capacity array
        self.channel = 0

class BS(Entity):
    def __init__(self, coordinate, num_recv_ant = 1, index = 0, num_ant = 4):
        super().__init__(coordinate, index ,num_ant)
        #self.max_power = 20 

        #init beamforming matrix
        self.W = np.mat(np.zeros((num_ant, 1))) #1 for one user
        self.W_Pmax = 0 # max power

class RIS(Entity):
    def __init__(self, coordinate, index = 0, num_elements = 25): 
        super().__init__(coordinate, index ,num_elements)

        #init reflecting phase shift
        self.q = np.ones(self.num_ant, dtype=complex)
        self.Phi = np.mat(np.diag(self.q), dtype = complex) # q = ONES(num_elements)
