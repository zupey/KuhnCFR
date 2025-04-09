# Linus' Implementation of Regret Matching and CFR
The purpose of implementing these algorithms is to better understand the intuition by implementation.

# Regret Matching
The basic regret matching algorithm is run on simple one shot normal-form games.  In essence, in each iteration, the regret matching algorithm tries to calculate the regret of playing each action by comparing what the player got vs what they could have got by playing that action.  Then, in the next iteration, the player plays each action proportional to the regret.

I ran regret matching on a few games:
1. Rock-Paper-Scissors
2. Battle of the Sexes (this is not a zero-sum game so regret matching does not work in practice but surprisingly, it still converged to mixed nash equilibrium).
3. Blotto
4. Penalty shootout (shooter is crippled on one leg)

Curious to find the distribution of strategies that the regret matching algorithm plays in battle of the sexes to see how mixed nash is reached because there are 2 pure strategies. And having pure strategies means if it is played, nobody would deviate.

# KuhnCFR
Trying to Solve Kuhn Poker using Counterfactual Regret Minimization
In my first iteration, I implemented KuhnPoker game in OOP manner to generate the game states.  Then, I tried to implement counterfactual regret minimization on this game.  However, due to my limited understanding, there were many bugs in the code.  After implementing regret matching, I re-implemented CFR for Kuhn Poker using the bare minimum with less overhead.  It correctly identified the value of the game to be -1/18 for P1.  
