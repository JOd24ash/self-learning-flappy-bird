# Flappy Bird AI (NEAT + Pygame)

A Flappy Bird clone that teaches itself to play using **NEAT** (NeuroEvolution of Augmenting Topologies). Many birds play at the same time, each controlled by its own small neural network. The birds that survive longest are bred and mutated to form the next generation, and over time they learn to fly through the pipes without dying.

## How it works

1. A population of birds is created, each with a random neural network.
2. Every frame, each network receives 3 inputs and decides whether to jump:
   - the bird's `y` position
   - the distance from the bird to the top pipe opening
   - the distance from the bird to the bottom pipe opening
3. Fitness is awarded as follows:
   - `+0.1` for every frame the bird stays alive
   - `+5` for passing a pipe
   - `-1` for hitting a pipe
4. Birds die when they hit a pipe, the ground, or the top of the screen.
5. When all birds are dead, NEAT selects the best genomes, breeds and mutates them, and starts the next generation.
6. Training stops when a bird reaches the `fitness_threshold` set in the config, or after 50 generations.

## Project structure

```
flappy-bird-ai/
├── flapy_bird_tutorial.py            # game + NEAT training code
├── config_feedforward_68c3f3d4a7.txt # NEAT configuration
├── requirements.txt
├── README.md
└── imgs/
    ├── bird1.png
    ├── bird2.png
    ├── bird3.png
    ├── pipe.png
    ├── base.png
    └── bg.png
```

## Requirements

- Python 3.8+
- [pygame](https://www.pygame.org/)
- [neat-python](https://neat-python.readthedocs.io/)

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

`requirements.txt`:

```
pygame
neat-python
```

## Usage

```bash
python flapy_bird_tutorial.py
```

A window opens and you will see the birds learning generation by generation. Progress is also printed in the terminal.

## Configuration

NEAT settings live in `config_feedforward_68c3f3d4a7.txt`. Important values:

| Setting | Value | Meaning |
|---|---|---|
| `pop_size` | 100 | Birds per generation |
| `fitness_threshold` | 100 | Training stops when a bird reaches this fitness |
| `num_inputs` | 3 | Network inputs (see above) |
| `num_outputs` | 1 | One output: jump or not |
| `activation_default` | tanh | Output is between -1 and 1; the bird jumps if it is above 0.5 |
| `no_fitness_termination` | False | Required by newer versions of neat-python |

## Troubleshooting

- **`Missing required configuration item: 'no_fitness_termination'`**: add `no_fitness_termination = False` to the `[NEAT]` section of the config file.
- **Images not found**: make sure the `imgs` folder is in the same directory as the script and contains all six images.
- **Path with spaces on Windows**: run from the terminal with `cd` into the project folder and then `python flapy_bird_tutorial.py`.

## Possible improvements

- Save the winning genome with `pickle` and replay it
- Plot fitness over generations with `matplotlib`
- Let a human play against the AI
- Tune the config to train faster

## Acknowledgements

- [NEAT-Python](https://neat-python.readthedocs.io/) documentation
- Original NEAT paper: Stanley & Miikkulainen, *Evolving Neural Networks through Augmenting Topologies* (2002)

