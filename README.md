# Effect of Different Behaviours On Survival During Earthquake Evacuation
Authors: Leah Heil, Nadezhda Dobreva, Megha Soni

This repository contains the code associated with our paper titled "Effect of Different Behaviours On Survival During Earthquake Evacuation." Our goal is to answer how different behaviours of people affect their survival rate during an earthquake evacuation. In order to do this, we investigate how survival rates are influenced by different strategies, and the boldness or scaredness displayed when choosing a path to follow. This is done in an environment that is modeled with an initial earthquake and the subsequent collapsing of small parts of the building of Mercator I from the campus of Radboud University, Nijmegen.

You can see our implementation in action here.

This repository is forked from Aalok Sate's fire evacuation simulator and our code builds on their existing codebase. We would like to express our gratitude to the original authors!

### Model
---
We model the floor plan as a CA. It is a 2D grid, the cells of which can have one of 7 types: normal (N), wall (W), door/bottleneck (B), safe zone (S), risky (R), dangerous (D), or grade (G). Example floor plans (i.e. the ones used in our simulations) are included in the `\in` folder. Rules for how the different cells get updated are defined in the `evacuate.py` file. The people are modeled using an agent-based model. The agents have different attributes (speed, strategy and boldness) and their goal is to get to the safe zone by moving between adjacent cells. The behaviour of the agents is defined in the `person.py` file.

The simulation can run for a specified max time or until everyone escapes, which allows us to study the survival rate, percent of people out of danger, mean escape time, etc.

### Usage
---
```
usage: evacuate.py [-h] [-i INPUT] [-n NUMPEOPLE] [-r RANDOM_STATE]
                   [-t MAX_TIME] [-f] [-g] [-v] [-d FIRE_RATE]
                   [-b BOTTLENECK_DELAY] [-a ANIMATION_DELAY]

optional arguments:
  -h, --help            show this help message and exit
  
  -i INPUT, --input INPUT
                        input floor plan file (default: in/twoexitbottleneck.py)
                        
  -n NUMPEOPLE, --numpeople NUMPEOPLE
                        number of people in the simulation (default: 10)
                        
  -r RANDOM_STATE, --random_state RANDOM_STATE
                        aka. seed (default: 8675309)
                        
  -t MAX_TIME, --max_time MAX_TIME
                        the building collapses at this clock tick. people
                        beginning movement before this will be assumed to have
                        moved away sufficiently (no default argument)
                        
  -d FIRE_RATE, --fire_rate FIRE_RATE
                        exponent of spread of fire rate function exponentiator
                        fire grows exponentially. d determines how exponentially.
                        
  -b BOTTLENECK_DELAY, --bottleneck_delay BOTTLENECK_DELAY
                        how long until the next person may leave the B
                        
  -a ANIMATION_DELAY, --animation_delay ANIMATION_DELAY
                        delay per frame of animated visualization (s, default: 1)
                        
  -f, --no_spread_fire  disallow fire to spread around? (default: false)
  
  -g, --no_graphical_output
                        disallow graphics? (default: false)
                        
  -v, --verbose         show excessive output? (default: false)
  -S SCAREDNESS, --scaredness_rate SCAREDNESS
			number of people who are scared to walk through danger
  -F FOLLOWER, --follower_rate FOLLOWER
			number of people who have the follower strategy
  -I ID, --run_id ID
			keeps track of the run id
                         
```
