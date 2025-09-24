# Pass the Pigs (Dice Game)

Quick introduction: Pass the Pigs is a version of the dice game Pig (created by David Moffatt in 1977) and uses asymmetrical throwing dice. 

--> Each turn, you throw 2 model pigs (each has a dot on its side) and the player will gain/lose points or be eliminated from the game. 
--> Winner is first player to reach a predetermined score (lets say 100).

### Rules
![alt text](https://github.com/apayne185/pass-the-pig-RL/blob/main/pass_the_pigs/game/game-rules.png?raw=true)




## stuff to modify 
- add 2+ players? make it collaborative? 
- **try cournot variant**
- **try sequential variant (stackelberg)** --> close to what we already have
- progress further with hog calls --> we already have a withhogcall flag
- multiagent (we train both agents to see if they converge to stable strategies)
    --> maybe try collusion or coalitions if 2+ players
- progressive training (curriculum learning)
    1. start with dumb opponents - roller 50% 
    2. then smarter opponents (baseline threshold 15, 20,...)
    3. then mirror match (agent v itself --> self play)


### Game State

## Directory 

## Project Overview
### Methods 



## Running the Code



*need to analyze the final results using the graph examples in the pdf slideshow*