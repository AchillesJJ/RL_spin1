# RL_spin1

This is the complete code for the question in StackOverflow (https://stackoverflow.com/questions/47478641/slow-down-when-using-openmp-and-calling-subroutine-in-a-loop?noredirect=1#comment81913388_47478641)

The description for each files :

main.f90 : main file in which it calls Q_learning() function multiple times
main_mod.f90 : define some parameters
function.f90 : some self-defined functions
ED.f90 : matrix diagonalization calling MKL
Watkins_Q_learning.f90 : implement Q-learning algorithm


most of the time is costed in evaluating Q_learning function, especially for line 339 in "Qatkins_Q_learning.f90" : theta = theta+delta_te. In fact, it cost almost 90% of total time.
