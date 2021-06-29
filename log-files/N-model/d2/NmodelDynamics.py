import random as r
import numpy as np
from network_dict import network_dictionary
import itertools
import copy



class ProcessingNetwork:
    # N-model network class
    def __init__(self, A, D, alpha, mu, holding_cost,  name):
        self.alpha = np.asarray(alpha)  # arrival rates
        self.mu = np.asarray(mu)  # service rates
        self.uniform_rate = np.sum(alpha)+np.sum(mu)  # uniform rate for uniformization
        self.p_arriving = np.divide(self.alpha, self.uniform_rate)# normalized arrival rates
        self.p_compl = np.divide(self.mu, self.uniform_rate) #normalized service rates
        self.cumsum_rates = np.unique(np.cumsum(np.concatenate([self.p_arriving, self.p_compl])))

        self.A = np.asarray(A)  # each row represents activity: -1 means job is departing, +1 means job is arriving
        self.routing_matrix = 1 * (self.A > 0)
        self.D = np.asarray(D)  # ith row represents buffers that associated to the ith stations



        self.action_size = np.prod(np.sum(D, axis=1))  # total number of possible actions
        self.action_size_per_buffer = [sum(D[i]) for i in range(len(D))]  # number of possible actions for each station
        self.stations_num = np.shape(D)[0]  # number of stations
        self.buffers_num = 2  # number of buffers

        self.holding_cost = holding_cost
        self.network_name = name




        self.dict_absolute_to_binary_action, self.dict_absolute_to_per_server_action = self.absolute_to_binary()
        self.actions = list(self.dict_absolute_to_binary_action.values())  # list of all actions


    @classmethod
    def Nmodel_from_load(cls, load: float):
        # another constructor for the standard queuing networks
        # based on a queuing network name, find the queuing network info in the 'network_dictionary.py'
        return cls(A=network_dictionary['Nmodel']['A'],
                   D=network_dictionary['Nmodel']['D'],
                   alpha=np.asarray([1.3 * load, 0.4 * load]),
                   mu=network_dictionary['Nmodel']['mu'],
                   holding_cost=network_dictionary['Nmodel']['holding_cost'],
                   name=network_dictionary['Nmodel']['name'])



    def absolute_to_binary(self):
        """
        :return:
        dict_absolute_to_binary_action: Python dictionary where keys are 'act_ind' action representation,
                                        values are 'action_full' representation
        dict_absolute_to_per_server_action: Python dictionary where keys are 'act_ind' action representation,
                                            values are 'action_for_server' representation
        act_ind - all possible actions are numerated GLOBALLY as 0, 1, 2, ...
        action_full - buffers that have priority are equal to 1, otherwise 0
        action_for_server - all possible actions FOR EACH STATIONS are numerated as 0, 1, 2, ...
        For the simple reentrant line.
        If priority is given to the first class:
            act_ind = [0]
            action_full = [1, 1, 0]
            action_for_server = [0, 0]
        If priority is given to the third class:
            act_ind = [1]
            action_full = [0, 1, 1]
            action_for_server = [1, 0]
        """

        dict_absolute_to_binary_action = {}
        dict_absolute_to_per_server_action = {}

        actions_buffers = [[a] for a in range(self.action_size_per_buffer[0])]

        for ar_i in range(1, self.stations_num):
            a =[]
            for c in actions_buffers:
                for b in range(self.action_size_per_buffer[ar_i]):
                    a.append(c+[b])
            actions_buffers = a

        assert len(actions_buffers) == self.action_size
        for i, k in enumerate(actions_buffers):
            dict_absolute_to_binary_action[i] = self.action_to_binary(k)
            dict_absolute_to_per_server_action[i] = k



        return dict_absolute_to_binary_action, dict_absolute_to_per_server_action




    def action_to_binary(self, act_ind):
        """
        change action representation
        :param act_ind: all possible actions are numerated GLOBALLY as 0, 1, 2, ...
        :return: buffers that have priority are equal to 1, otherwise 0
        For the simple reentrant line.
        If priority is given to the first class:
        act_ind = [0]
        action_full = [1, 1, 0]
        If priority is given to the third class:
        act_ind = [1]
        action_full = [0, 1, 1]
        """

        action_full = np.zeros(self.buffers_num)
        for i in range(len(self.D)): #number of stations 
            res_act = act_ind[i] 
            k = -1

            for act in range(len(self.D[0])): #check each class to see if being served by current station
                if self.D[i][act] == 1: #current class served by current station
                    k += 1
                    if res_act == k: # found the one priority is given to for current station
                        break
            action_full[act] = 1

        return action_full




    def next_state_N1(self, state, action):
        """
        :param state: current state
        :param action: action
        :return: next state
        """

        w = np.random.random()
        wi = 0 
        while w > self.cumsum_rates[wi]: #randomly choose which state to go to 
            wi += 1
        if wi == 0:
            state_next = state + np.asarray([1, 0]) #station 1 new arrival 
        elif wi == 1:
            state_next = state + np.asarray([0, 1]) #station 2 new arrival
        elif wi == 2 and (state[0] > 0):
            state_next = state - np.asarray([1, 0]) #station 1 departure 
        elif wi == 3 and ((action[0] == 1 or state[1] == 0) and state[0] > 1):
            state_next = state - np.asarray([1, 0]) #station 1 departure
        elif wi == 4 and ((action[0] == 0 or state[0] < 2) and state[1] > 0):
            state_next = state - np.asarray([0, 1]) #station 2 departure 
        else:
            state_next = state #no change 
        return state_next



    def next_state_probN(self, states_array):
        """
        Compute probability of each transition for each action for the criss-cross network
        :param states_array: array of states
        :return: probability of each transition
        """

        num = len(states_array)

        prod_for_actions_list = []

        prob = np.asarray([self.p_arriving[0]* np.ones(num), self.p_arriving[1]* np.ones(num), self.p_compl[0]*(states_array[:,0]>0) +
                           self.p_compl[1]*(states_array[:,0]>1), self.p_compl[2]* (states_array[:,0]<2) *  (states_array[:,1]>0)   ])
        prob_fake_transition = 1 - np.sum(prob, axis=0)  # probability of a fake transition
        prod_for_actions = np.hstack([prob.T, prob_fake_transition[:, np.newaxis]])
        prod_for_actions_list.append(prod_for_actions)



        prob2 = np.asarray([self.p_arriving[0]* np.ones(num), self.p_arriving[1]* np.ones(num), self.p_compl[0]*(states_array[:,0]>0) +
                           self.p_compl[1]*( 1*(states_array[:,0]>1) * 1*(states_array[:,1]==0) ), self.p_compl[2]*  (states_array[:,1]>0)  ])

        prob_fake_transition2 = 1 - np.sum(prob2, axis=0)  # probability of a fake transition
        prod_for_actions2 = np.hstack([prob2.T, prob_fake_transition2[:, np.newaxis]])
        prod_for_actions_list.append(prod_for_actions2)
        return prod_for_actions_list


    def next_state_list(self):
        """
        :return:
        list: Python dictionary s.t. keys are ( state > 0, action, activity), values are jobs transitions
        list_next_states: set of all possible jobs transitions
        """
        list = {}
        s_D = np.shape(self.D)


        adjoint_buffers = {} # Python dictionary: key is a buffer, value is a list of buffers associated to the same station
        for i in range(0, s_D[0]): #each station
            for j in range(0, s_D[1]): #each buffer for current station
                if self.D[i][j] ==1: #able to serve the buffer 
                    d = copy.copy(self.D[i])  # TODO: np.copy?
                    d[j] = 0
                    adjoint_buffers[j] = copy.copy(d)
        self.adjoint_buffers = adjoint_buffers

        for a in [False, True]:#self.actions:  # indicator that activity 'w' is legitimate
            for s in itertools.product([0, 1], repeat=self.buffers_num):  # combination of non-empty, empty buffers
                for w in range(0, int(np.sum(self.alpha>0)+np.sum(self.mu[1:]>0))):  # activity

                        ar = np.asarray(s, 'int8')
                        if w < np.sum(self.alpha>0):  # arrival activity
                            el = np.nonzero(self.alpha)[0][w]
                            arriving = np.zeros(self.buffers_num, 'int8')
                            arriving[el] = 1
                            list[(tuple(ar), a, w)] = arriving
                        elif ar[w - np.sum(self.alpha>0)]>0 and \
                                (a or np.sum(np.dot(ar, adjoint_buffers[w - np.sum(self.alpha>0)]))==0):# service activity is possible
                            list[(tuple(ar), a, w)] = self.A[w - np.sum(self.alpha>0)]

                        else:  # service activity is not possible. Fake transition
                            list[(tuple(ar), a, w)] = np.zeros(self.buffers_num, 'int8')


        list_next_states = np.asarray([ list[(tuple(np.ones(self.buffers_num)), 1, w)]
                                        for w in range(0, int(np.sum(self.alpha>0)+np.sum(self.mu[1:]>0)))])
        return list_next_states#, list

