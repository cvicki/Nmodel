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

##### d12:
Description: Prints same parameters as d11, except use algo 1's val and adv estimates to update policy. Therefore the model should train and see how algo 2's estimate differs. 

