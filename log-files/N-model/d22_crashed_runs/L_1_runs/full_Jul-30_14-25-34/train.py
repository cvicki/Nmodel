import numpy as np
from actor_utils import Policy
from value_function import NNValueFunction
from utils import Logger, Scaler
import os
import argparse
import NmodelDynamics as pn
import datetime
import copy
import random

from simulation import run_policy, run_weights




def diag_dot(A, B):
    # returns np.diag(np.dot(A, B))
    return np.einsum("ij,ji->i", A, B) #element by element multiplication, then sum each row?

def get_vals(values, observes , state_dict, label):
    #returns the average value estimates for each state in state dict 
    for i in range(len(observes)):
        state = str(observes[i])
        if state_dict.get(state) != None:
            state_dict[state] += [values[i]]
    lst = []
    for key in state_dict.keys():
        val = np.array(state_dict[key]).mean()
        lst.append((key + label , val))
        # logger.log({label + key : val})
    # print(label, ': ', lst)
    return lst 

def log_vals(val, label, index, logger, tolist = False):
    """
    Helper for logging values.
    :param val: values to be logged (list or can be converted to a list)
    :param label: description of val
    :param index: index of values used 
    :param tolist: true if val needs to be converted to a list  
    returns: list of values 
    """
    lst = [] # to hold the values
    for i in index:
        if tolist:
            v = list(val[i])[0]
            logger.log({label + str(i): v})
            lst.append(v)
        else:
            logger.log({label + str(i): val[i]})
            lst.append(val[i])
    return lst 

# def add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration, state1_dict, logger, states = None):
def add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration, L):
    """
    compute value function for further training of Value Neural Network
    :param trajectory: simulated data
    :param network: queuing network
    :param policy: current policy
    :param gamma: discount factor
    :param lam: lambda parameter in GAE
    :param scaler: normalization values
    """
    # counter = 0
    start_time = datetime.datetime.now()
    for trajectory in trajectories:

        if iteration!=1:

            values = trajectory['values_NN']
            observes = trajectory['observes']
            unscaled_obs = trajectory['unscaled_obs']


            ###### compute expectation of the value function of the next state ###########
            probab_of_actions = policy.sample(observes) # probability of choosing actions according to a NN policy

            action_array = network.next_state_probN(unscaled_obs) # transition probabilities for fixed actions

            # expectation of the value function for fixed actions
            value_for_each_action_list = []
            for act in action_array:
                value_for_each_action_list.append(diag_dot(act, trajectory['values_set'].T))
            value_for_each_action = np.vstack(value_for_each_action_list)

            P_pi = diag_dot(probab_of_actions, value_for_each_action)  # expectation of the value function
            ##############################################################################################################

            # td-error computing
            tds_pi = trajectory['rewards'] - values + gamma*P_pi[:, np.newaxis]#gamma * np.append(values[1:], values[-1]), axis=0)#

            # value function computing for futher neural network training
            disc_sum_rew = discount(x=tds_pi,   gamma= gamma * lam, v_last = tds_pi[-1]) + values
            # lst = get_vals(disc_sum_rew, trajectory['unscaled_obs'],state1_dict, 'val1_',  logger)

        else:
            disc_sum_rew = discount(x=trajectory['rewards'],   gamma= gamma, v_last = trajectory['rewards'][-1])
            # lst = get_vals(disc_sum_rew, trajectory['unscaled_obs'],state1_dict, 'val1_',  logger)

        trajectory['disc_sum_rew'] = disc_sum_rew
        # lst = get_vals(disc_sum_rew, trajectory['unscaled_obs'], state1_dict, 'v1_', logger)
    # for key, val in lst:
    #     logger.log({'val1_' + key : val})

    # logger.log({'val1_first' : list(disc_sum_rew[0])[0]})

    
    
    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('add_disc_sum_rew time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

    burn = L # cut the last 'burn' points of the generated trajectories

    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])
    if iteration == 1:
        scaler.update(np.hstack((unscaled_obs, disc_sum_rew))) # scaler updates just once
    scale, offset = scaler.get()
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    disc_sum_rew_norm = (disc_sum_rew - offset[-1]) * scale[-1]
    # lst1_norm  = get_vals(disc_sum_rew, unscaled_obs, state1_dict, 'v1n', logger)
    if iteration == 1:
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]

    return observes, disc_sum_rew_norm #, lst, lst1_norm 
    # return observes, disc_sum_rew_norm


def adv_logger(adv, trajectory, adv_log, iteration, act_adv, state = np.array([2,2])):
    """
    Advantage function logger.
    :param adv: np array of calculated advantage function estimates
    :param trajectory: the trajectory from which advantages were calculated 
    :param adv_log: the np array holding all the estimates 
                rows: advantage estimates
                columns: policy iteration
    :param state: the state chosen to log the adv estimates 
    :param length: the number of adv estimates to log, will log the first [length]
            of them. If state visited less than [length] times, will have None value
    """
    observes = trajectory['unscaled_obs']
    length, _ = adv_log.shape
    counter = 1
    adv_list = ['None']* length
    action_list = ['None']* length

    action_list[0] = str(iteration) +' algo1 action:'
    adv_list[0] = str(iteration) +' algo1 adv:'

    for ac, ad in act_adv:
        action_list[counter] = str(ac)
        adv_list[counter] = str(ad)
        counter +=1 

    action_list[counter] = ' algo2 action:'
    adv_list[counter] = ' algo2 adv:'
    counter +=1
    mod = 0


    for i in range(0, len(observes)):
        if counter >= length:
            break
        curr_state = observes[i]
        # if curr_state == state:
        if np.array_equal(curr_state, state):
            if np.array_equal(curr_state, state) and mod % 5 ==0:
                adv_list[counter] = str(adv[i][0])
                action_list[counter] = str(trajectory['actions'][i][0])
                counter +=1 
            mod +=1
    act_adv = np.vstack((action_list, adv_list)).T
    # print(act_adv.shape)
    adv_log = np.append(adv_log, act_adv, axis = 1)
    # print(adv_log.shape)
    return adv_log

# def advantage_fun(trajectories, policy, network, gamma, lam, scaler, iteration, state_dict, logger, states= None):
def advantage_fun(trajectories, gamma, lam, scaler, iteration, adv_log, val_func, logger, L, act_adv1):
    """
    for algo 2, computes advantage function, very similar to value function of algo 1 
    compute value function for further training of Value Neural Network
    :param trajectory: simulated data
    :param network: queuing network
    :param policy: current policy
    :param gamma: discount factor
    :param lam: lambda parameter in GAE
    :param scaler: normalization values
    """
    start_time = datetime.datetime.now()
    for trajectory in trajectories:

        if iteration!=1:

            values = trajectory['values_NN'] # from the NN
            # observes = trajectory['observes'] #normalized states
            # unscaled_obs = trajectory['unscaled_obs'] #original states 

            ###### compute value function of the next state ###########
            obs_next = np.append(trajectory['observes'], [[0,0]], axis=0)
            obs_next = np.delete(obs_next, 0, axis = 0)
            values_next = val_func.predict(obs_next) #use value NN predict val from value fun 
            scale, offset = scaler.get()
            values_next = values_next / scale[-1] + offset[-1]

            summed_vals = trajectory['rewards'] - values + gamma * values_next
            advantages = discount_2(x=summed_vals, gamma= lam*gamma, L = L ) #+ values  #new advantage function


        else:
            advantages = discount_2(x=trajectory['rewards'],   gamma= gamma*lam, L= L)

        trajectory['advantages'] = advantages
    # lst1 = get_vals(advantages, trajectory['unscaled_obs'], state_dict, 'adv2_')
    
    # for key, val in lst1:
    #     logger.log({key:val})
    adv_log = adv_logger(advantages, trajectory, adv_log, iteration, act_adv1) 

    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('add_disc_sum_rew time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

    burn = L # cut the last 'burn' points of the generated trajectories
    advantages = np.concatenate([t['advantages'][:-burn] for t in trajectories])
    actions = np.concatenate([t['actions'][:-burn] for t in trajectories])
    # unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    # scale, offset = scaler.get()
    # observes = (unscaled_obs - offset[:-1]) * scale[:-1])
    advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages
    # if iteration == 1:
    #     for t in trajectories:
    #         t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]

    return advantages, actions, adv_log

def add_disc_sum_rew_2(trajectories, policy, network, gamma, lam, scaler, iteration):
    """
    for algo 2 method 2 
    compute value function for further training of Value Neural Network
    :param trajectory: simulated data
    :param network: queuing network
    :param policy: current policy
    :param gamma: discount factor
    :param lam: lambda parameter in GAE
    :param scaler: normalization values
    """
    start_time = datetime.datetime.now()
    for trajectory in trajectories:

        if iteration!=1:

            values = trajectory['values_NN'] # from the NN 
            observes = trajectory['observes'] #normalized states
            unscaled_obs = trajectory['unscaled_obs'] #original states 

            ###### compute expectation of the value function of the next state ###########
            probab_of_actions = policy.sample(observes) # probability of choosing actions according to a NN policy

            action_array = network.next_state_probN(unscaled_obs) # transition probabilities for fixed actions

            # expectation of the value function for fixed actions
            value_for_each_action_list = []
            for act in action_array:
                value_for_each_action_list.append(diag_dot(act, trajectory['values_set'].T))
            value_for_each_action = np.vstack(value_for_each_action_list)

            P_pi = diag_dot(probab_of_actions, value_for_each_action)  # expectation of the value function
            ##############################################################################################################

            # td-error computing
            tds_pi = trajectory['rewards'] - values + gamma*P_pi[:, np.newaxis]#gamma * np.append(values[1:], values[-1]), axis=0)#

            # algo 1 value function computing for futher neural network training
            disc_sum_rew = discount(x=tds_pi,   gamma= lam*gamma, v_last = tds_pi[-1]) + values  #uncomment for algo 1 value function.
           
            # # algo 2 advantage function for futher neural network training
            # advantages = discount(x=tds_pi,   gamma= lam*gamma, v_last = tds_pi[-1]) #new advantage function

        else:
            # advantages = discount(x=trajectory['rewards'],   gamma= gamma * lam, v_last = trajectory['rewards'][-1])
            disc_sum_rew = discount(x=trajectory['rewards'],   gamma= gamma, v_last = trajectory['rewards'][-1]) #algo 1 value function
        
        trajectory['disc_sum_rew'] = disc_sum_rew
        # trajectory['advantages'] = advantages


    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('add_disc_sum_rew time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

    burn = 1 # cut the last 'burn' points of the generated trajectories

    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    # advantages = np.concatenate([t['advantages'][:-burn] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])

    if iteration == 1:
        scaler.update(np.hstack((unscaled_obs, disc_sum_rew))) # scaler updates just once

    scale, offset = scaler.get()
    # advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages
    disc_sum_rew_norm = (disc_sum_rew - offset[-1]) * scale[-1]
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    
    return  disc_sum_rew_norm, observes


# def val_fun_2(trajectories, gamma, iteration, scaler, state2_dict, logger, lam):
def val_fun_2(trajectories, gamma, iteration, scaler, lam, logger):
    """
    estimates the value function for algo 2
    :param trajectories: simulated data
    :param gamma: discount factor
    """
    # counter = 0
    for trajectory in trajectories:
        # counter +=1
        value = discount(x=trajectory['rewards'],   gamma= gamma*lam, v_last = trajectory['rewards'][-1])
        trajectory['values'] = value
        # log_vals(value, 'val2_', states, logger, True)
        # lst = get_vals(value, trajectory['unscaled_obs'], state2_dict, 'v2', logger)
        # print(value.shape)
        
    
    burn = 1
    values = np.concatenate([t['values'][:-burn] for t in trajectories])
    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    if iteration == 1:
        scaler.update(np.hstack((unscaled_obs, values))) # scaler updates just once
        scale, offset = scaler.get()
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]
    
    scale, offset = scaler.get()
    values_norm = (values - offset[-1]) * scale[-1] 
    # print(values_norm.shape)
    # print(values_norm[0].shape)


    # lst_norm = get_vals(values_norm, unscaled_obs, state2_dict, 'v2n', logger)
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    return values_norm, observes
    # return lst, lst_norm

def discount(x, gamma, v_last):
    """ Calculate discounted forward sum of a sequence at each point """
    disc_array = np.zeros((len(x), 1))
    disc_array[-1] = v_last
    for i in range(len(x) - 2, -1, -1):
        if x[i+1]!=0:
            disc_array[i] = x[i] + gamma * disc_array[i + 1]

    return disc_array

def discount_2(x, gamma, L):
    """ 
    Calculate discounted forward sum of a sequence at each point with set future interval (l) 
    :param x: the values to sum, burn the last L states. 
    """
    # disc_array = np.zeros((len(x), 1))
    # disc_array[-1] = x[-1]
    # for i in range(2, L+1): #-2 to -L, second last to L last 
    #     disc_array[-i] = x[-i] + gamma *disc_array[-i+1] 
        
    # for i in range(len(x) - L -2, -1, -1): #start at end of original values, iterate to beginning 
    #     disc_array[i] = x[i] + (gamma * disc_array[i + 1]) - (gamma**(L+1))*x[i+L+1]
    # return disc_array[:-L]
    disc_array = np.zeros((len(x), 1))
    disc_array[-1] = x[-1]

    for i in range(len(x) - 2, -1, -1): #start at end of original values, iterate to beginning 
        if i < len(x)-L-2: #can sum full L values 
            disc_array[i] = x[i] + (gamma * disc_array[i + 1]) - (gamma**(L+1))*x[i+L+1]
        else: 
            disc_array[i] = x[i] + (gamma * disc_array[i + 1])

    return disc_array




def add_value(trajectories, val_func, scaler, possible_states):
    """
    # compute value function from the Value Neural Network
    :param trajectory_whole: simulated data
    :param val_func: Value Neural Network
    :param scaler: normalization values
    :param possible_states: transitions that are possible for the queuing network
    """
    start_time = datetime.datetime.now()
    scale, offset = scaler.get()


    # get value NN values for generated states and all possible 'next' states
    for trajectory in trajectories:
        values = val_func.predict(trajectory['observes']) #use value NN predict val from value fun 
        trajectory['values_NN'] = values / scale[-1] + offset[-1]

        # approximate value function of the states where transitions are possible from generated states
        values_set = np.zeros(( len(possible_states)+1, len(trajectory['observes'])))

        new_obs = (trajectory['unscaled_last'] - offset[:-1]) * scale[:-1] #normalized prev state values 
        values = val_func.predict(new_obs) #predict value from prev state 
        values = values / scale[-1] + offset[-1] #unscaled
        values_set[-1] = np.squeeze(values) #add to set when dont transition 

        for count, trans in enumerate(possible_states):
            new_obs =(trajectory['unscaled_last'] + trans - offset[:-1]) * scale[:-1] #trans, how many people enter/leave buffer 
            values = val_func.predict(new_obs)
            values = values / scale[-1] + offset[-1]
            values_set[count] = np.squeeze(values)


        trajectory['values_set'] = values_set.T


    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('add_value time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

def build_train_set(trajectories, gamma, scaler, adv_log, L):
    """
    # data pre-processing for training, computation of advantage function estimates
    :param trajectory_whole:  simulated data
    :param scaler: normalization values
    :return: data for further Policy and Value neural networks training
    """

    for trajectory in trajectories:
        values = trajectory['values_NN']

        unscaled_obs = trajectory['unscaled_obs']


        ###### compute expectation of the value function of the next state ###########
        action_array = network.next_state_probN(unscaled_obs) # transition probabilities for fixed actions

        # expectation of the value function for fixed actions
        value_for_each_action_list = []
        for act in action_array:
            value_for_each_action_list.append(diag_dot(act, trajectory['values_set'].T))
        value_for_each_action = np.vstack(value_for_each_action_list)
        ##############################################################################################################

        # # expectation of the value function w.r.t the actual actions in data
        distr_fir = np.eye(len(network.actions))[trajectory['actions_glob']]

        P_a = diag_dot(distr_fir, value_for_each_action)

        advantages = trajectory['rewards'] - values +gamma*P_a[:, np.newaxis]# gamma * np.append(values[1:], values[-1]), axis=0)  #
        trajectory['advantages_1'] = np.asarray(advantages)

    action_taken = []
    act_adv = []
    for i in range(len(advantages)):
        if len(act_adv) == 2:
            break
        act = trajectory['actions'][i][0]
        if act not in action_taken:
            action_taken.append(act)
            adv = trajectory['advantages_1'][i][0]
            act_adv.append((act, adv))

    act_adv = sorted(act_adv, key=lambda x:x[0])
    # print('act_adv: ', act_adv) 

    # adv_logger(advantages, trajectory, state_dict, logger) 
    # adv_log = adv_logger(advantages, trajectory, adv_log, iteration)

    start_time = datetime.datetime.now()
    burn = L

    # merge datapoints from all trajectories
    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    # disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])

    """
    scale, offset = scaler.get()
    actions = np.concatenate([t['actions'][:-burn] for t in trajectories])
    advantages = np.concatenate([t['advantages_1'][:-burn] for t in trajectories])
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages
    """


    # uncomment if need to average over estimates for each state
    # ########## averaging value function estimations over all data ##########################
    # states_sum = {}
    # states_number = {}
    # states_positions = {}
    #
    # for i in range(len(unscaled_obs)):
    #     if tuple(unscaled_obs[i]) not in states_sum:
    #         states_sum[tuple(unscaled_obs[i])] = disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] = 1
    #         states_positions[tuple(unscaled_obs[i])] = [i]
    #
    #     else:
    #         states_sum[tuple(unscaled_obs[i])] +=  disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] += 1
    #         states_positions[tuple(unscaled_obs[i])].append(i)
    #
    # for key in states_sum:
    #     av = states_sum[key] / states_number[key]
    #     for i in states_positions[key]:
    #         disc_sum_rew[i] = av
    # ########################################################################################

    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('build_train_set time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')
    # return observes,  actions, advantages, disc_sum_rew
    return act_adv
    # return actions, advantages



def build_train_set2(trajectories, gamma, scaler, state_dict, logger, states = None):
    """
    # advantage function computation for algo 2 method 1, returns less values 
    (advantage function using algo 1)
    # data pre-processing for training, computation of advantage function estimates
    :param trajectory_whole:  simulated data
    :param scaler: normalization values
    :return: data for further Policy and Value neural networks training
    """
    # counter = 0
    for trajectory in trajectories:
        # counter +=1
        values = trajectory['values_NN']

        unscaled_obs = trajectory['unscaled_obs']

        ###### compute expectation of the value function of the next state ###########
        action_array = network.next_state_probN(unscaled_obs) # transition probabilities for fixed actions

        # expectation of the value function for fixed actions
        value_for_each_action_list = []
        for act in action_array:
            value_for_each_action_list.append(diag_dot(act, trajectory['values_set'].T))
        value_for_each_action = np.vstack(value_for_each_action_list)
        ##############################################################################################################

        # # expectation of the value function w.r.t the actual actions in data
        distr_fir = np.eye(len(network.actions))[trajectory['actions_glob']]

        P_a = diag_dot(distr_fir, value_for_each_action)

        advantages = trajectory['rewards'] - values +gamma*P_a[:, np.newaxis]# gamma * np.append(values[1:], values[-1]), axis=0)  #
        # lst = []
        # for i in range(1,4):
        #     logger.log({str(counter) + 'adv1_' + str(-i): list(advantages[-i])[0] })
        #     lst.append(list(advantages[-i])[0])

        # logger.log({str(counter) + 'adv1_first' : list(advantages[0])[0] })
        # lst.append(list(advantages[0])[0])
        # lst = log_vals(advantages, 'adv1_', states, logger, True)
        lst = get_vals(advantages, trajectory['unscaled_obs'], state_dict, 'adv1_', logger)

        
        trajectory['advantages'] = np.asarray(advantages)

    start_time = datetime.datetime.now()
    burn = 1

    # merge datapoints from all trajectories
    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    # disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])

    # scale, offset = scaler.get()
    # actions = np.concatenate([t['actions'][:-burn] for t in trajectories])
    advantages = np.concatenate([t['advantages'][:-burn] for t in trajectories])
    # observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages


    # uncomment if need to average over estimates for each state
    # ########## averaging value function estimations over all data ##########################
    # states_sum = {}
    # states_number = {}
    # states_positions = {}
    #
    # for i in range(len(unscaled_obs)):
    #     if tuple(unscaled_obs[i]) not in states_sum:
    #         states_sum[tuple(unscaled_obs[i])] = disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] = 1
    #         states_positions[tuple(unscaled_obs[i])] = [i]
    #
    #     else:
    #         states_sum[tuple(unscaled_obs[i])] +=  disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] += 1
    #         states_positions[tuple(unscaled_obs[i])].append(i)
    #
    # for key in states_sum:
    #     av = states_sum[key] / states_number[key]
    #     for i in states_positions[key]:
    #         disc_sum_rew[i] = av
    # ########################################################################################

    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('build_train_set time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')
    return advantages, lst
    



def log_batch_stats(observes, actions, advantages, logger, episode):
# def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    # metadata tracking

    time_total = datetime.datetime.now() - logger.time_start
    logger.log({'_mean_act': np.mean(actions),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                # '_mean_discrew': np.mean(disc_sum_rew),
                # '_min_discrew': np.min(disc_sum_rew),
                # '_max_discrew': np.max(disc_sum_rew),
                # '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode,
                '_time_from_beginning_in_minutes': int((time_total.total_seconds() / 60) * 100) / 100.
                })

def most_common(trajectories, n, nonzero):
    """
    :param trajectories: results from actors
    :param n: number of states to return
    :param nonzero: True: all states do not include zero, False: allowed to include zero.
    returns a dictionary of the n most common states (unscaled_obs) visited in
    trajectories as keys and empty list as values (to hold values estimates calculated)

    """
    # count instances of each state 
    states = {} #key: state, value: count of state 
    for trajectory in trajectories:
        for state in trajectory['unscaled_obs']:
            state = str(state)
            if states.get(state) == None:
                states[state] = 1
            else:
                states[state] +=1

    print('[2,2]', states.get('[2 2]'))
    states_dict = {}
    #filter on nonzero states and occuring more than 300 times.
    if nonzero:
        f_dict = {k:v for k,v in states.items() 
                        if (k.find('0') == -1) and (v > 300)}

        sampled = random.sample(f_dict.items(), n)
        for k, _ in sampled:
            states_dict[k] = []
    
    else:
        #find n most common states 
        for _ in range(n):
            key = max(states, key= states.get) #get state 
            states_dict[key] = []
            states[key] = -1 #remove from top
    print(states_dict.keys())
    return states_dict

def log_diff(lst1, lst2, logger, avg_only, label):
    #lst1 and lst2 are the estimated value functions of the ten most common states in the first trajectory
    #if
    diff_unscaled = np.zeros(len(lst1))

    for i in range(len(lst1)):
        k1,v1 = lst1[i]
        k2,v2 = lst2[i]
        diff_unscaled[i] = v1 - v2 
        if not avg_only: 
            logger.log({k1: v1,
                        k2: v2})
    logger.log({label+'avg_diff' : diff_unscaled.mean()})
    return 

# TODO: check shadow name
def main(network, num_policy_iterations, no_of_actors, episode_duration, no_arrivals, gamma, lam, clipping_parameter,
         ep_v, bs_v, lr_v, ep_p, bs_p, lr_p, kl_targ, hid1_mult):
    """
    # Main training loop
    :param: see ArgumentParser below
    """

    obs_dim = network.buffers_num # 2
    act_dim = network.action_size_per_buffer # number of possible actions for each station
    now = datetime.datetime.utcnow().strftime("%b-%d_%H-%M-%S")  # create unique directories
    time_start= datetime.datetime.now()
    logger = Logger(logname=network.network_name, now=now, time_start=time_start)


    scaler = Scaler(obs_dim + 1) # object that keeps statistics needed for normalization before NNs training
    # scaler2 = Scaler(obs_dim + 1)  #comment out when not comparing
    val_func = NNValueFunction(obs_dim, hid1_mult, ep_v, bs_v, lr_v) # Value Neural Network initialization
    policy = Policy(obs_dim, act_dim, hid1_mult, kl_targ,  ep_p, bs_p, lr_p, clipping_parameter) # Policy Neural Network initialization


    ############## creating set of initial states for episodes in simulations##########################
    run_policy(network, policy, scaler, logger, gamma, policy_iter_num=0, no_episodes=1, time_steps=episode_duration)
    ###########################################################################

    iteration = 0  # count of policy iterations
    weights_set = []
    scaler_set = []
    state1_dict = {} # for algo 1 val fun estimates
    # state2_dict = {} # for algo 2 val fun estimates
    
    # states = random.sample(range(1, episode_duration), k = 5)
    # states.sort()
    # print(states)

    num = 200
    adv_log = np.zeros((num + 1,1))

    while iteration < num_policy_iterations:
        # decrease clipping_range and learning rate each iteration
        iteration += 1
        alpha = 1. - iteration / num_policy_iterations
        policy.clipping_range = max(0.01, alpha*clipping_parameter)
        policy.lr = max(0.05, alpha)*lr_p
        # save policy NN parameters each 10th iteration
        if iteration % 10 == 1:
            weights_set.append(policy.get_weights())
            scaler_set.append(copy.copy(scaler))

        # simulate trajectories
        trajectories = run_policy(network, policy, scaler, logger, gamma, iteration,
                                      no_episodes=no_of_actors, time_steps=episode_duration)

        ### algo 1
        # for debugging algo 2, change network parameter '-n' (# policy iterations) to 5 
        """
        # find 10 most visited states 
        if iteration == 1:
            state1_dict = most_common(trajectories, 10) # for algo 1 val fun estimates
            state2_dict = {k:[] for k in state1_dict.keys()} # for algo 2 val fun estimates, deep copy of state1_dict 
        
        # compute value NN for each visited state
        add_value(trajectories, val_func, scaler, network.next_state_list())
        # compute estimated of the value function
        observes, disc_sum_rew_norm, lst1, lst1_norm = add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration, state1_dict, logger) #algo 1
        # observes, disc_sum_rew_norm = add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration, state1_dict, logger) #algo 1
        lst2, lst2_norm =val_fun_2(trajectories, gamma, iteration, scaler, state2_dict, logger) #algo 2
        log_diff(lst1, lst2, logger, False, 'u')
        log_diff(lst1_norm, lst2_norm, logger, True, 'n')
        # update value function
        val_func.fit(observes, disc_sum_rew_norm, logger)
        # recompute value NN for each visited state
        add_value(trajectories, val_func, scaler, network.next_state_list())
        # compute advantage function estimates
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories, gamma, scaler)
        # add various stats
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, iteration)
        # update policy
        policy.update(observes, actions, np.squeeze(advantages), logger)

        logger.write(display=True)  # write logger results to file and stdout
        # """

        ### algo 1 val, unsummed algo 2 adv fun 
        """
        # find 10 most visited states 
        if iteration == 1:
            state1_dict = most_common(trajectories, 10) # for algo 1 val fun estimates
            state2_dict = {k:[] for k in state1_dict.keys()} # for algo 2 val fun estimates, deep copy of state1_dict 
        # compute value NN for each visited state
        add_value(trajectories, val_func, scaler, network.next_state_list())
        # compute advantage function estimates and differences 
        lst1, lst2 = advantage_fun(trajectories, policy, network, gamma, lam, scaler, iteration, state1_dict, logger)
        advantages, lst = build_train_set2(trajectories, gamma, scaler, state2_dict, logger) #use algo 1 advantage function 

        # j = 0 #counter 
        # for i in state1_dict.keys():
        #     logger.log({'diff_full' + str(i) : lst[j] - lst1[j],
        #                 'diff_' + str(i) : lst[j] - lst2[j]})
        #     j +=1
        
        # compute value function estimates and update scaler 
        observes, disc_sum_rew_norm = add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration, state1_dict, logger) #algo 1
        # values_norm, observes = val_fun_2(trajectories, gamma, iteration, scaler, lam, logger, states)
        #compute differences 
        log_diff(lst, lst1, logger, True, 'ad_diff_')
        log_diff(lst, lst2, logger, False, 'ad_diff_full_')
        # update value function
        val_func.fit(observes, disc_sum_rew_norm, logger)
        # compute actions
        burn = 1
        actions = np.concatenate([t['actions'][:-burn] for t in trajectories])
        # add various stats
        log_batch_stats(observes, actions, advantages, logger, iteration)
        # log_batch_stats(observes, actions, tds_pi, logger, iteration)

        # update policy
        policy.update(observes, actions, np.squeeze(advantages), logger)
        # policy.update(observes, actions, np.squeeze(tds_pi), logger)


        logger.write(display=True)  # write logger results to file and stdout
        # """

        ### algo 2.1: value as alg2 but lam*gamma and adv as algo 1 (training ~61-63)
        """
        # compute estimated of the value function
        values_norm= val_fun_2(trajectories, gamma, iteration, scaler, lam)
        # compute value NN for each visited state
        add_value(trajectories, val_func, scaler, network.next_state_list())
        # compute advantage function estimates
        observes,  actions, advantages = build_train_set(trajectories, gamma, scaler)
        # update value function
        val_func.fit(observes, values_norm, logger)
        # add various stats
        log_batch_stats(observes, actions, advantages,logger, iteration)
        # update policy
        policy.update(observes, actions, np.squeeze(advantages), logger)

        logger.write(display=True)  # write logger results to file and stdout
        # """

        ## algo 1: with new advantage function and algo 1 val function 
        # """
        L = 2 #rollback amount

        # compute value NN for each visited state
        add_value(trajectories, val_func, scaler, network.next_state_list())
        #compute value function estimates and update scaler 
        observes, disc_sum_rew_norm = add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration, L) #algo 1
        # update value function NN 
        val_func.fit(observes, disc_sum_rew_norm, logger)# add various stats
        #recompute value NN for each visited state 
        add_value(trajectories, val_func, scaler, network.next_state_list())
        # compute advantage function estimates  
        act_adv1 = build_train_set(trajectories, gamma, scaler, adv_log, L)
        advantages, actions, adv_log = advantage_fun(trajectories, gamma, lam, scaler, iteration, adv_log, val_func, logger, L, act_adv1) #new advantage function  
        log_batch_stats(observes, actions, advantages, logger, iteration)
        val_func.fit(observes, disc_sum_rew_norm, logger)# add various stats

        # update policy
        policy.update(observes, actions, np.squeeze(advantages), logger)

        logger.write(display=True)  # write logger results to file and stdout
        # """


        # ### algo 2: with value as alg 1, advantage as alg 2 
        """
        # find 10 most visited states 
        # if iteration == 1:
        #     state1_dict = most_common(trajectories, 3) # for algo 1 val fun estimates
        #     state2_dict = {k:[] for k in state1_dict.keys()} # for algo 2 val fun estimates, deep copy of state1_dict 
        # compute value NN for each visited state
        add_value(trajectories, val_func, scaler, network.next_state_list())
        # compute advantage function estimates 
        lst1 = advantage_fun(trajectories, policy, network, gamma, lam, scaler, iteration, state1_dict, logger, states) # algo 2 
        lst, advantages = build_train_set2(trajectories, gamma, scaler, state2_dict, logger, states) #use algo 1 advantage function 

        j = 0 #counter 
        for i in states:
            logger.log({'diff' + str(i) : lst[j] - lst1[j]})
            j +=1
        
        # compute value function estimates 
        observes, disc_sum_rew_norm = add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration, state1_dict, logger, states) #algo 1
        values_norm, observes2= val_fun_2(trajectories, gamma, iteration, scaler, lam,logger, states )
        val_func.fit(observes, disc_sum_rew_norm, logger)
        # # compute actions
        burn = 1
        actions = np.concatenate([t['actions'][:-burn] for t in trajectories])
        # add various stats
        log_batch_stats(observes, actions, advantages, logger, iteration)
        # update policy
        policy.update(observes, actions, np.squeeze(advantages), logger)

        logger.write(display=True)  # write logger results to file and stdout
        # """

    ########## save policy NN parameters of the final policy and normalization parameters ##############################
    weights = policy.get_weights()
    file_weights = os.path.join(logger.path_weights, 'weights_' + str(iteration) + '.npy')
    np.save(file_weights, np.array(weights, dtype=object))

    file_scaler = os.path.join(logger.path_weights, 'scaler_' + str(iteration) + '.npy')
    scale, offset = scaler.get()
    np.save(file_scaler, np.asarray([scale, offset]))
    weights_set.append(policy.get_weights())
    scaler_set.append(copy.copy(scaler))
    #####################################################################
    
    ###logging adv estimates###
    np.savetxt(str(logger.path_adv) + "/adv_estimates_5.csv", adv_log, delimiter=",", fmt='%s')


    ################## Policy performance #######################   
    # performance_evolution_all, ci_all = run_weights(network, weights_set, policy, scaler,
    #                                     time_steps=int(1. / sum(network.p_arriving) * no_arrivals))


    # file_res = os.path.join(logger.path_weights,
    #                         'average_' + str(performance_evolution_all[-1]) + '+-' +str(ci_all[-1]) + '.txt')
    # file = open(file_res, "w")
    # for i in range(len(ci_all)):
    #     file.write(str(performance_evolution_all[i])+'\n')
    # file.write('\n')
    # for i in range(len(ci_all)):
    #     file.write(str(ci_all[i])+'\n')
    ############################################333


    logger.close()





if __name__ == "__main__":

    network = pn.ProcessingNetwork.Nmodel_from_load(load=0.9)

    parser = argparse.ArgumentParser(description=('Train policy for a queueing network '
                                                  'using Proximal Policy Optimizer'))

    parser.add_argument('-n', '--num_policy_iterations', type=int, help='Number of policy iterations to run',
                        default=80) #default=50, use 5 for val fun comp.
    parser.add_argument('-b', '--no_of_actors', type=int, help='Number of episodes per training batch',
                        default=2)
    parser.add_argument('-t', '--episode_duration', type=int, help='Number of time-steps per an episode',
                        default=50*10**3) # default=20*10**3, algo 2: 50*10**3
    parser.add_argument('-x', '--no_arrivals', type=int, help='Number of arrivals to evaluate policies',
                        default=5*10**6)

    parser.add_argument('-g', '--gamma', type=float, help='Discount factor',
                        default=0.998)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.99)
    parser.add_argument('-c', '--clipping_parameter', type=float, help='Initial clipping parameter',
                        default=0.2)

    parser.add_argument('-e', '--ep_v', type=float, help='number of epochs for value NN training',
                        default=10) #default 10 
    parser.add_argument('-s', '--bs_v', type=float, help='minibatch size for value NN training',
                        default=256)
    parser.add_argument('-r', '--lr_v', type=float, help='learning rate for value NN training',
                        default=2.5 * 10**(-4))

    parser.add_argument('-p', '--ep_p', type=float, help='number of epochs for policy NN training',
                        default=3)
    parser.add_argument('-w', '--bs_p', type=float, help='minibatch size for policy NN training',
                        default=2048) #default = 2048, algo 2: 4096
    parser.add_argument('-q', '--lr_p', type=float, help='learning rate for policy NN training',
                        default=2.5 * 10 ** (-4)) # default=2.5 * 10 ** (-4), algo 2: 5 * 10 ** (-5)

    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-m', '--hid1_mult', type=int, help='Size of first hidden layer for value and policy NNs',
                        default=10)


    args = parser.parse_args()
    main(network,  **vars(args))