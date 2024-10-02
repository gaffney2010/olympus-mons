# olympus-mons

Olympus Mons (OM) is a continuous Markov Chain library, with modeled transitions and simulated inference.

Models are built with discrete states, actions, and variables.  Transitions and variable changes can be extensibly modeled as a function of state and variables.  At inference time, a set of events will be simulated with modeled behavior.

OM is designed for ease-of-use.  To that end, simulated data looks exactly like training data.

## Intro

## Example

We define a game called "24":  

24 - Version 1
> Players take turns rolling a 6-sided die.  Both players keep a running total of the values they roll.  The first person to reach 24 wins.

Let's model this.  Implementation 1:

```python
import olympusmons as om

twentyfour = om.Model()
twentyfour.add_variable("turn")
twentyfour.add_variable("player_1_score")
twentyfour.add_variable("player_2_score")
twentyfour.add_state("Turn")
twentyfour.add_action("Roll")

twentyfour.end_condition = om.om_or(
    om.cond("player_1_score", ">=", 24),
    om.cond("player_2_score", ">=", 24),
)

twentyfour.transition("Turn", om.transition_constant("Roll"))
twentyfour.update("Roll", om.update_branch([
    (om.cond("turn", "eq" "player_1"), {
        "turn": om.update_constant("player_2"),
        "player_1_score": os.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
    }),
    (om.cond("turn", "eq" "player_2"), {
        "turn": om.update_constant("player_1"),
        "player_2_score": os.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
    }),
]))
```

Alternatively, we can write the update rule using some syntactic sugar.  Implementation 2:

```python
twentyfour.add_pseudovariable("active_player_score", om.if_else(
    om.cond("turn", "==", "player_1"),
    "player_1_score",
    "player_2_score",
))

twentyfour.update("Roll", om.update_branch({
    "turn": om.rotate(["player_1", "player_2"]),
    "active_player_score": os.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
})
```

Now we can simulate a game.

```python
twentyfour.sim({
    "turn": "player_1",
    "player_1_score": 0,
    "player_2_score": 0,
})
```

This will return the ending state.  If we want to predict the probability of player_1 winning (given that they go first), we can count ourselves.

```python
twentyfour.set_default_start_state({
    "player_1_score": 0,
    "player_2_score": 0,
})

player_1_wins = 0
for _ in range(N_SIMS):
    sim = twentyfour.sim({"turn": "player_1"})
    if sim["player_1_score"] > sim["player_2_score"]:
        player_1_wins += 1
print(player_1_wins * 1.0 / N_SIMS)
```

We could alternatively define this as a two-state Markov chain.  Implementation 3:

```python
import olympusmons as om

twentyfour = om.Model()
twentyfour.add_variable("player_1_score")
twentyfour.add_variable("player_2_score")
twentyfour.add_state("PlayerOneTurn")
twentyfour.add_action("PlayerOneRoll")
twentyfour.add_state("PlayerTwoTurn")
twentyfour.add_action("PlayerTwoRoll")

twentyfour.end_condition = om.om_or(
    om.cond("player_1_score", ">=", 24),
    om.cond("player_2_score", ">=", 24),
)

twentyfour.transition("PlayerOneTurn", om.transition_constant("PlayerOneRoll"))
twentyfour.update("PlayerOneRoll", {
    "player_1_score": os.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
})
twentyfour.transition("PlayerTwoTurn", om.transition_constant("PlayerTwoRoll"))
twentyfour.update("PlayerTwoRoll", {
    "player_2_score": os.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
})
```

Notice that actions have to be globally unique.  So different states necessitate different actions.

Although we CAN define this as two states, should we?  No.  It will be a repeated concern:  When should two states of the game be the same state?  We could, for example, define different nodes for late game vs early game.  OM has enough flexibility to allow this, but we do have an opinion:  **Each state should correspond to a model.**  If it doesn't make sense to do different models, then it doesn't make sense to do different states.  Here the "model" is an untrained uniform-random selection, which makes it easier to copy.  But when you have to train models, the different states will provide different training data.

## Comparison to other packages

In the above description, I've preferred to be more practical than theoretical.  Where it sits in the mathematical ecosystem could be described as:

- The OM framework is a continuous stochastic process with the Markov property.  Meaning that it has a continuous state space due to continuous variables being allowed.  But it's memoryless because the current state is needed to predict the next state.  The separation of state into "state" and "variable" is a convenience only.
- No state is hidden in the training, though states can be transformed before being used to determine actions.
- The framework resembles an MCMC.  It uses a Monte Carlo for inference, but not for training.

This library tries to optimize for easy-of-use and interpretability for its use case.  It will only report the end-state of simulations.  It does not try to answer statistical questions, like "How much time is spent in each node?" or "Is the chain transitive?"

Some comparabale libraries are:
- [PyDTMC](https://pypi.org/project/PyDTMC/)
- [pgmpy Markov Chain](https://pgmpy.org/models/markovchain.html)
- [markovify](https://github.com/jsvine/markovify)
- [mchmm](https://pypi.org/project/mchmm/)
- [PyMC](https://www.pymc.io/welcome.html)
- [stochastic](https://stochastic.readthedocs.io/en/latest/)

|                    | Continuous<br>State Space | Graphing | Generates<br>Samples | Models Trans-<br>ition Matrix | Stats | HMM |
|:------------------:|:-------------------------:|:--------:|:--------------------:|:-----------------------------:|:-----:|:---:|
| Olympus Mons       |             X             |     X    |           X          |               X               |       |     |
| PyDTMC             |                           |     X    |                      |                               |   X   |  X  |
| pgmpy Markov Chain |                           |          |           X          |                               |       |     |
| markovify          |                           |          |           X          |               X               |       |     |
| mchmm              |                           |     X    |           X          |               X               |       |  X  |
| PyMC               |             X             |     X    |           X          |               X               |   X   |     |
| stochastic         |             X             |          |           X          |                               |   X   |     |


