##### Log File Descriptions:

###### d1:
Description: Simplifying algo 1. 
Kept algo 1's formulas for estimating value and advantage functions. Modified code so that the Value NN was only updated once at the end(add_value called only once) compared to when called twice in original algorithm. Used algo 1's network parameters. (Following logs also used the same parameters)

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
