##### Log File Descriptions:

###### d1:
Description: Simplifying algo 1. 
Kept algo 1's formulas for estimating value and advantage functions. Modified code so that the Value NN values were computed only once in the beginning and used for both advantage and value estimates and then updated at the end. The original computed the val NN values to estimate the value function, then updated the NN and recomputed new values to estimate the advantage. Used algo 1's network parameters. (Following logs also used the same parameters)

Result: At around the 10-12 iteration, output showed signs of learning and improvement (judged by % of optimal actions increasing). Final iteration (50) had % of optimal actions at ~69. This is comparable to some runs of the original algorithm. 

##### Modifying the formula for estimating the Value function 
###### d2, d2_2:
Description: Modified value fun to be the adv. fun of algo 2 and used algo 1's adv fun (excluded 'values' (&zeta; (x^(t)) from disc_sum_rev calculation). Used algo 2's order of execution. 

Result: Did not train well, % of optimal actions (%opt) stayed at around 50 for most iterations but would drop to around 30 for some then increase back to 50. Reran a second time (d2_2) and saw similar results.

###### d3_2:
Description: Simplifying value function, adv function estimate still the same. Remove the expectation of the value function (P_Pi in add_disc_sum_rew) and gamma. Therefore only reward and previous policy iteration (zeta). 

Result: Trained poorly, %opt dropped to single digits and very high average costs before the code crashed. Crashed 3/3 times. 
###### d3_3:
Description: Simplifying value function by having the discount factor be only &gamma; (gamma) (as opposed to &lambda; &gamma;, lambda * gamma). Adv. fun same as algo 1. 

Result: Algorithm trained, at around 10~12 iteration, began to have steadier increase in %opt, however variable performance in average cost. Finshed with final iteration %opt of 61 and 19 average cost (20, 23 in prev iteration). 

###### d3_4, d3_42:
Description: Modified algo 2 value function. Changed discount factor to &lambda; &gamma; (previously &gamma;). Adv. function same as algo 1. 

Results: Trained well in the sense the %opt increased somewhat steadily after the 15-20th iteration. ValFuncLoss much larger than algo 1 (0.036 vs 5.01e-06). %opt was 63, 58.5 on final policy iteration. 

##### Modifying the formula for estimating the Advantage function 
###### d4:
Description: Value function is modified algo 2 with &gamma; &lambda; as discount factor. Used algo 2 for advantage function estimate. 

Result: Did not train. Average cost fluctuated between 19 and 40, generally mid-20's. %opt stayed between 48 and 50. ValFuncLoss comparable to d3_4.

###### d5:
Description: Algo 2's adv fun, Algo 1's val fun. 

Result: Much lower ValFuncLoss (9.190e-0.6), otherwise results comparable to d4. Average costs flucuate between 20 and 30. 

###### d6:
Description: Modify algo 2's adv function by adding prev policy iteration results (identical to algo 1's val fun estimate). Use algo 1's val fun. 

Result: Results comparable to d5. ValFuncLoss is slightly larger (3.43e-05), otherwise similarly not training.

###### d7:
Description: Comparing algo 1 and algo 2's adv function. Policy updates according to algo 2's adv function however the advantage estimate is also calculated using algo 1's formula. The 10 most common states of the first trajectory are chosen and the average (unscaled) advantage for each of those 10 states are calculated for each policy iteration. In addition the average difference (find difference for each of the averaged states and take overall average)of all the ten states is found (u_advavg_diff). The average scaled advantage is also calculated (n_advavg_diff), each using its own respective scaling method. v1\[1,0\] refers to the average advantage estimate for state \[1,0\] using algo 1. 

Result: Since the value function for algo 1 returns large negative values, the advantage function estimate using algo 2 is subsequently larger than the estimates from algo 1. Since the value function increases in magnitude thoughout the iterations, the difference between the average values increases. At the first iteration there is around a difference of 400, whereas in the final 50th iteration the difference is ~4500. Overall trained poorly, the %opt flucuated between 49 and 50. The average cost flucuated around as well between 20-30.

###### d8:
Description: Comparing algo 1 and algo 2's adv function. Done similarly to d7, except used algo 2's val function estimate since the values returned have a lower variance. 

Result: The difference between the averages is smaller however there is an overall increase (more negative) in the differences between the algorithms. The differences were around 600 near the start of the training and around 875 at the end. In regards to %opt, near the beginning there was a sharp drop to 4 before increasing back to 50 and staying around there. Average cost was between 24-30 except for the seven iterations where %opt dropped then increased back to 50.  

###### d9:
Description: Comparing the advantage functions. Printing algo 2's advantage function estimates without summing future reward. 'adv2_1' represents eq 4.7's value when k = 1 and the remaining values are not summed. 'adv1_1' represents algo 1's adv function (4.5) estimate for k = 1.'adv_1_diff' = adv1_1 - adv2_1. Use algo 2's value function as zeta in algo 2's adv function estimate. 

Result: The most significant difference is with the later advantage function estimates. The first few advantage values are quite similar. The averages calculated are the magnitude of the difference (absolute value of each of the differences)

###### d10:
Description: Same as d9 but with algo 1 value function. 

Result: The same trend is seen as d9, except the differences are slightly larger. 

###### d10_2, d10_3:
Description: Same as d10, except reduced the number of iterations to 30 and printing out last 3 advantage function estimates (unsummed). 'adv_-2_diff' = 'adv1_-2' - 'adv2_-2' where -2 is the second to last item. 

Result: The difference between estimates for the first iteration is higher (adv_first_diff:5.20) compared to d10 (2.69) and the last value average difference off is much lower compared to d10. Potentially an outlier with d10. 

###### d11, d11_2, d11_3:
Description: Further reduced the number of iterations to 20. In addition to those parameters in D10, included the full advantage estimate, d11_3 includes the percentage of the advantage estimate was captured by the first value in the summation. 

Result: Overall small difference between algo 1 and algo 2 estimates except a few times and difference is quite significant. 

###### d12:
Description: Prints same parameters as d11, except use algo 1's val and adv estimates to update policy. Therefore the model should train and see how algo 2's estimate differs. 

###### d13: 
Description: In addition to d12 parameters, also value function estimates and value NN outputs for the same states. 

###### d14, d14_2:
Description: Uses algo 1's value and advantage function to update policy and run code. Randomly chose 5 episodes from the second trajectory and output the estimates of the value and advantage functions for each policy update. d14 runs for 15 iterations, d14_2 runs for 30 iterations. Other parameters are the default parameters. 

Parameters: 
- Advantage parameters:
  - algo 2:
  - 'adv2_4911': unsummed advantage estimate using algo 2 for the 4911th episode of the second (last) trajectory. Unsummed meaning, only the first element of the summation (eq. 4.7) is returned. 
  - 'adv2_full_4911': the advantage estimate using algo 2 (all the elements summed).  
  - algo 1:
  - 'adv1_4911': advantage function estimate using algo 1's eq. 
  - misc:
  - 'state1_4911': buffer 1's state at the 4911th episode
  - 'state2_4911': buffer 2's state at the 4911th episode
  - 'diff4911': 'adv1_4911' - 'adv2_full_4911'
  - note: same naming scheme follows for the other 4 states. 
- Value parameters: 
  - 'val_NN4911': output of the value function neural net. The NN is updated using algo 1's val. function estimate. 
  - 'val1_4911': estimate of value function using algo 1 of the 4911th epsiode of trajectory 2. 
  - 'val2_4911': estimate of value function using algo 2.

Results: 

###### d15:
Description: Value function with algo 1 and modified algo 2 advantage function. Using only the first unsummed value instead of entire summation. (will run later)

###### d16:
Description: State specific advantage function logging. Chose the 10 most common state and averaged the advantages at each state for the trajectory. Values are for the second trajectory. Used algo 1's value and advantage function to update policy.

###### d17, d17_2: 
Description: Used new advantage function and algo 2's value function. New value (zeta(x^(t+1))) found using the next value of value_NN. d17_2, is the same except prints out the advantage function of a few states. 

###### d18, d18_2, d18_3:
Description: Used new advanatage function and algo 2's value function. New value (zeta(x^(t+1))) found by shifting trajectory['observes'] down one and repredicting the values using the val NN. 
d18_2, d18_3 used algo 1's value function. d18_2: 50 iterations instead of 30 with d18. d18_3 used algo 2 parameters for network (crashed at iteration 48)

###### d19_10, d19_100, d_19_1000:
Description: Used new advantage function (4.7) and summed a constant amount forward (N+L -> k+L, where L = 10,100, or 1000 ) and algo 1 val function (4.3). &zeta;(x<sup>(t+1)</sup>) was calculated by using the value function NN to predict the value of the next state of the trajectory. 

Results: 
Comparable Val NN loss across all three L values with L=1000 slightly higher (1.5e-05 vs 5.6e-5). The advantage function is still only negative across all three with the larger L values having larger negative values (with follows since there are more values to sum). The advantage function increases (becomes less negative) across the policy iterations however appears to slow down and perhaps decrease again at around the 50th iteration. None of the values appear to be training, L=1000 almost appears to be training in the opposite (favoring worse options) since the %opt appears to be decreasing in one run (d19_1000_1) but just fluctated around 50 the next time (d19_1000_2). In regards to policy update, there was a larger difference in min and max ratio with smaller L values (10,100) than L=1000 in the first ~10 iterations, ratios at the end were comparable.  

Note: At each L = 10,100,1000 the code crashed from poor performance (avg cost >100 and low %optimal steps). 

###### d20_10, d20_10_2:
Description: Used new advantage function (4.7) and L=10. Logs the first instance (index) of the chosen state, the action taken at that state, and the advantage estimate assoicated with that action. Used algo 1's order of execution and value function. The states were randomly sampled from states with both nonzero buffers and were visited at least 300 times in the first trajectory. Increased the number of timesteps per episode from 20000 to 50000 to avoid the program crashing from poor performance (divide by zero error)

Results: 
In d20_10_2, two of the states chosen were skewed in the number of people per buffer ([12,1] and [11,3]) and overall the policy favored choosing action 1 and would be reflected by a larger advantage function estimate. However there would be instances were choosing action 0 also resulted in a very large estimate. In regards to more similar states (ex [2,2]) the advantage estimates were a lot more varied and tended to change largely across policy iterations even when the same action was taken. Overall the %opt did not train. d20_10 had 25 policy updates, d20_10_2 had 50.

###### d21_10, d21_100, d21_algo1:
Description: Same parameters as d20 but with additional log file: 'd21_10_adv_estimates'. Logs the advantage estimates and action taken (0 or 1) for everytime the second(final) trajectory visits the chosen state ([2,2]). 
Each column represents a trajectory and the number of rows corresponds to the number of times [2,2] was visited. Each column was set to a length of 500 since it appeared that the majority of trajectories did not visit [2,2] more than 500 times and most only ~300. d21_10 has L=10 whereas d21_100 has L=100.
d21_algo1 logs the same values except uses algo 1's adv fun. 

Result: %opt did not improve. The advantage function would change between actions (decrease or increase) but would also change rapidly while the same action was taken. The sign and magnitude would also flucuate a lot. 


###### d22 folder:
Description: Algo 1 value function and algo 2 advantage function with L = 1, 2 rollout (sorted into L_1_runs and L_2_runs folders). Changed network parameters to 50,000 timesteps. Runs that did not crash are indicated with 'full_' prefix in name. For 'adv_estimate' algo 1's adv function estimate for both actions 0 and 1 at state [2,2]are printed on top. Below, every 3 (or 5 if have '_5' suffix) times [2,2] is visited, the action chosen and the adv estimate for action is logged. If there are None values, the trajectory did not visit at least 588 (980) times ((201-5)*3). Otherwise the estimates are truncated. 


Issues: Program would crash at 48th iteration (of 50) due to 'Control-c event', potentially cause by ray package being unstable for Windows when running 50,000 timesteps per trajectory and L=1. Due to crash only the first 32 iterations would be logged onto log file and advantage estimates are not logged (since advantage estimates are saved only at very end). 
Program also crashed at 48th iteration (of 60) due to 'control-c event' for 50,000 timesteps and L=2, however was sucessful in running another time. 
Solution: Program has been successful at running at 47 iterations. 

Results: 