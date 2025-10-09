# Pass the Pigs (Dice Game)

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.26.2+-green.svg)](https://gymnasium.farama.org/)
[![Pygame](https://img.shields.io/badge/pygame-2.5.2+-orange.svg)](https://www.pygame.org/news)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![TensorBoard](https://img.shields.io/badge/tensorboard-2.0+-ff6f00.svg)](https://www.tensorflow.org/tensorboard)


Quick introduction: Pass the Pigs is a version of the dice game Pig (created by David Moffatt in 1977) and uses asymmetrical throwing dice. 

--> Each turn, players throw 2 model pigs (each has a dot on its side) and the player will gain/lose points or be eliminated from the game. 
--> Players can choose to "bank" points or risk losing them on a bad roll
--> Winner is first player to reach a predetermined score (at the moment, we say 100).

### Rules
![alt text](https://github.com/apayne185/pass-the-pig-RL/blob/main/pass_the_pigs/game/game-rules.png?raw=true)


### Scoring System

| Roll Result | Points | Description |
|-------------|---------|-------------|
| Double Trotter | 40 | Both pigs on side with legs showing |
| Double Snouter | 30 | Both pigs on snout |
| Double Razorback | 20 | Both pigs on back |
| Double Leaning Jowler | 20 | Both pigs leaning on side |
| Leaning Jowler | 15 | Single pig leaning on side |
| Snouter | 10 | Single pig on snout |
| Razorback | 5 | Single pig on back |
| Trotter | 5 | Single pig with legs showing | 
| Sider | 1 | Single pig on side | roll again |
| Piggyback | 0 | One pig on top of another | lose turn |
| Pig Out | 0 | One pig on side, one on back | lose turn |
| Oinker | 0 | Both pigs on different sides | lose turn |


## stuff to modify 
- **try cournot variant**  --> anna
- **try sequential variant (stackelberg)** --> close to what we already have --> tomas
- progressive training (curriculum learning)
    1. start with dumb opponents - roller 50% 
    2. then smarter opponents (baseline threshold 15, 20,...)
    3. then mirror match (agent v itself --> self play) 

- progress further with hog calls --> we already have a withhogcall flag 
    1. intermediate rewards integrated for agent to learn immediate rewards after correct/incorrect calls

- multiagent (we train both agents to see if they converge to stable strategies)
    --> maybe try collusion or coalitions if 2+ players
    --> add 2+ players? make it collaborative? 


## Directory 
```
pass-the-pig/
├── env.py              # Main environment and agent classes
├── environment.yml    # Conda environment setup
├── README.md          # This file
├── output/            # Model results saved here 
├── runs/              # 
└── pass_the_pigs/     # Game assets (images, sounds)
```

## Project Overview

### Environment Design
- **State**: 
- **Actions**: 
- **Rewards**:


### Agent Architecture 

The Q-learning agent features:

- **Curriculum Training**: Progressive opponent difficulty
- **Efficient Q-tables**: defaultdict implementation for large state spaces
- **Epsilon-greedy Exploration**: Balanced exploration vs exploitation
- **Threshold-based Strategies**: Intelligent banking decisions




## Running the Code
### Packages/Libraries

- Python 3.7+
- pip package manager 

**Activate the conda environment**

```bash 
conda env create -f environment.yml
conda activate ptp-env
conda list
```



## License

This project is available for educational and research purposes.

## Acknowledgments

- Original Pass the Pigs game by David Moffatt and Nigel Hirst
- Gymnasium framework by Farama Foundation
- Reinforcement learning algorithms from Sutton & Barto



*need to analyze the final results using the graph examples in the pdf slideshow*


