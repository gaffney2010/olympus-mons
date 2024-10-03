# olympus-mons

// TODO Redo this part
<!-- Olympus Mons (OM) is a continuous Markov Chain library, with modeled transitions and simulated inference. -->

<!-- Models are built with discrete states, actions, and variables.  Transitions and variable changes can be extensibly modeled as a function of state and variables.  At inference time, a set of events will be simulated with modeled behavior. -->

<!-- OM is designed for ease-of-use.  To that end, simulated data looks exactly like training data. -->

## Intro

OM is technically a Mealy machine.  What that means for us is that we have to define a graph by States (nodes) and Actions (edges).

States use Models to decide to which Action to take, and Actions decide which state comes next.

As a starting example, let's look at a simple 3-state setup.

```python
import olympusmons as om


graph = (
    om.GraphBuilder("Rotate")
    .State("A", model=om.ConstModel("to_b")).Action("to_b", next_state="B")
    .State("B", model=om.ConstModel("to_c")).Action("to_c", next_state="C")
    .State("C", model=om.ConstModel("to_a")).Action("to_a", next_state="A")
    .set_starting_state("A")
    .set_end_condition("step >= 5")
    .build()
)
```

This is pretty self-explanatory.  It uses a [builder pattern](https://refactoring.guru/design-patterns/builder).  The most unusual thing here is the end condition.  An end condition is always required.  `step` is a built-in counter that starts at 0 and counts up.  The end condition is encoded as a string rather than a lambda.  The internals of OM uses [SymPy](https://www.sympy.org/en/index.html) to translate into logic.  Handling it this way lets us do certain validity checks, and save/load the graph.

Once we have a graph, we can then run a simulation.

```python
output = graph.sim(debug="screen")
```

`output` is a dict that, for now, only contains `{"step": 5}`.  By setting `debug="screen"`, this will print to screen all the intermediate states and actions as a CSV:

```
state,action,step
A,to_b,0
B,to_c,1
C,to_a,2
A,to_b,3
B,to_c,4
```

Note that `state` and `step` are the values at the begining of the step.

To make this more interesting, we can add some randomness.  We do this with a non-constant Model.

```python
import random

import olympusmons as om


class SplitA(om.Model):
    def sim(input):
        if random.random() < 0.5:
            return "to_b"
        return "to_c"


graph = (
    om.GraphBuilder("Out and back")
    .State("A", model=SplitA([])).Action("to_b", next_state="B").Action("to_c", next_state="C")
    .State("B", model=om.ConstModel("to_a_from_b")).Action("to_a_from_b", next_state="A")
    .State("C", model=om.ConstModel("to_a_from_c")).Action("to_a_from_c", next_state="A")
    .set_starting_state("A")
    .set_end_condition("step >= 5")
    .build()
)
```

A few things to notice:

- The action names must be globally unique, so we added the "from" part this time.
- Model __init__ takes a list of variables that it's allow to access.  In this example, we don't have any variables, except the `step`.
- Model has a `sim` function that does simulations.  It takes the parameter `input` which is a dictionary of all the variable values.
- `sim` returns an Action name

We could have instead made a function that depens on `step` like this:

```python
class SplitA(om.Model):
    def sim(input):
        if random.random() < 0.5 / input["step"]:
            return "B"
        return "C"


graph = (
    om.GraphBuilder("Out and back")
    .State("A", model=SplitA(["step"])).Action("to_b", next_state="B").Action("to_c", next_state="C")
    ....
)
```

om.ConstModel is a Model factory that builds a model like `SplitA` but that always returns a fixed State.  That factory needs to know what the target

## Example

We define a game called "24":  

24 - Version 1
> Players take turns rolling a 6-sided die.  Both players keep a running total of the values they roll.  The first person to reach 24 wins.

Let's model this.  Implementation 1:

```python
import olympusmons as om

twentyfour = om.Graph()
twentyfour.add_variable("turn")
twentyfour.add_variable("player_1_score")
twentyfour.add_variable("player_2_score")
twentyfour.add_state("Turn", starting=True)
twentyfour.add_action("Roll")

twentyfour.end_condition = om.om_or(
    om.cond("player_1_score", ">=", 24),
    om.cond("player_2_score", ">=", 24),
)

twentyfour.transition("Turn", om.transition_constant("Roll"))
twentyfour.update("Roll", om.update_branch([
    (om.cond("turn", "eq" "'player_1'"), {
        "turn": om.update_constant("'player_2'"),
        "player_1_score": om.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
    }),
    (om.cond("turn", "eq" "'player_2'"), {
        "turn": om.update_constant("'player_1'"),
        "player_2_score": om.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
    }),
]))
```

Alternatively, we can write the update rule using some syntactic sugar.  Implementation 2:

```python
twentyfour.add_pseudovariable("active_player_score", om.if_else(
    om.cond("turn", "==", "'player_1'"),
    "player_1_score",
    "player_2_score",
))

twentyfour.update("Roll", om.update_branch({
    "turn": om.rotate(["'player_1'", "'player_2'"]),
    "active_player_score": om.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
})
```

Now we can simulate a game.

```python
twentyfour.sim({
    "turn": "'player_1'",
    "player_1_score": 0,
    "player_2_score": 0,
})
```

This will return the ending state.  If we want to predict the probability of player_1 winning (given that they go first), we can count ourselves.

If you wish to print the whole simulated game to screen, you can pass the debug argument.

```python
twentyfour.sim({
    "turn": "'player_1'",
    "player_1_score": 0,
    "player_2_score": 0,
}, debug="screen")
```

```python
twentyfour.set_default_start_state({
    "player_1_score": 0,
    "player_2_score": 0,
})

player_1_wins = 0
for _ in range(N_SIMS):
    sim = twentyfour.sim({"turn": "'player_1'"})
    if sim["player_1_score"] > sim["player_2_score"]:
        player_1_wins += 1
print(player_1_wins * 1.0 / N_SIMS)
```

We could alternatively define this as a two-state Markov chain.  Implementation 3:

```python
import olympusmons as om

twentyfour = om.Graph()
twentyfour.add_variable("player_1_score")
twentyfour.add_variable("player_2_score")
twentyfour.add_state("PlayerOneTurn", starting=True)
twentyfour.add_action("PlayerOneRoll")
twentyfour.add_state("PlayerTwoTurn")
twentyfour.add_action("PlayerTwoRoll")

twentyfour.end_condition = om.om_or(
    om.cond("player_1_score", ">=", 24),
    om.cond("player_2_score", ">=", 24),
)

twentyfour.transition("PlayerOneTurn", om.transition_constant("PlayerOneRoll"))
twentyfour.update("PlayerOneRoll", {
    "player_1_score": om.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
})
twentyfour.transition("PlayerTwoTurn", om.transition_constant("PlayerTwoRoll"))
twentyfour.update("PlayerTwoRoll", {
    "player_2_score": om.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
})
```

Notice that actions have to be globally unique.  So different states necessitate different actions.

Although we CAN define this as two states, should we?  No.  It will be a repeated concern:  When should two states of the game be the same state?  We could, for example, define different nodes for late game vs early game.  OM has enough flexibility to allow this, but we do have an opinion:  **Each state should correspond to a model.**  If it doesn't make sense to do different models, then it doesn't make sense to do different states.  Here the "model" is an untrained uniform-random selection, which makes it easier to copy.  But when you have to train models, the different states will provide different training data.

### Context variables

Context variables are game-level constants.  Though it would be easy to make a different `om.Graph` with a different constant, we sometimes want the same Model with different game-level constants to get trained together.

24 - Version 2
> Players take turns rolling a 6-sided die.  Both players keep a running total of the values they roll.  The first person to reach `goal` wins, where `goal` is a context variable.

```python
import olympusmons as om

twentyfour = om.Graph()
twentyfour.add_context("goal")
twentyfour.add_variable("turn")
twentyfour.add_variable("player_1_score")
twentyfour.add_variable("player_2_score")
twentyfour.add_pseudovariable("active_player_score", om.if_else(
    om.cond("turn", "==", "'player_1'"),
    "player_1_score",
    "player_2_score",
))
twentyfour.add_state("Turn", starting=True)
twentyfour.add_action("Roll")

twentyfour.end_condition = om.om_or(
    om.cond("player_1_score", ">=", "goal"),
    om.cond("player_2_score", ">=", "goal"),
)

twentyfour.transition("Turn", om.transition_constant("Roll"))
twentyfour.update("Roll", om.update_branch({
    "turn": om.rotate(["'player_1'", "'player_2'"]),
    "active_player_score": om.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
})
```

### Modeled behavior

So far, there's been no fitting to data.

24 - Version 3
> Players take turns.  Each turn they can either roll a 6-sided die, taking their score from the top face, or they can flip 6 coins, taking their score from the number of heads flipped.  Both players keep a running total of the values they roll.  The first person to reach `goal` wins, where `goal` is a context variable.

These are similar choices, but the coin flips are more likely to score around the average value.  It may be a good strategy to choose coins when you're ahead, and dice rolls when you're behind.  However, we don't know a priori if the players will use good strategy.  We will leave it up to the model to learn.

For our first implementation, we'll assume that the player chooses at random.

```python
import random

import olympusmons as om

twentyfour = om.Graph()
twentyfour.add_context("goal")
twentyfour.add_variable("turn")
twentyfour.add_variable("player_1_score")
twentyfour.add_variable("player_2_score")
twentyfour.add_pseudovariable("active_player_score", om.if_else(
    om.cond("turn", "==", "'player_1'"),
    "player_1_score",
    "player_2_score",
))
twentyfour.add_state("Turn", starting=True)
twentyfour.add_action("Flip")
twentyfour.add_action("Roll")

twentyfour.end_condition = om.om_or(
    om.cond("player_1_score", ">=", "goal"),
    om.cond("player_2_score", ">=", "goal"),
)

class FlipModel(om.Model):
    def sim(self, _):
        score = 0
        for _ in range(6):
            if random.random() < 0.5:
                score += 1
        return score

twentyfour.transition("Turn", om.transition_uniform(["Flip", "Roll"]))
twentyfour.update("Flip", om.update_branch({
    "turn": om.rotate(["'player_1'", "'player_2'"]),
    "active_player_score": om.plus_equal(FlipModel(inputs=[], outputs=["update"])),
})
twentyfour.update("Roll", om.update_branch({
    "turn": om.rotate(["'player_1'", "'player_2'"]),
    "active_player_score": om.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
})
```

`sim` takes an argument, but it still doesn't use anything.  `FlipModel` is an `om.Model`, and so inputs and outputs needs to get specified.  `inputs=[]` means that this doesn't depend on any variables, and `outputs=["update"]` is a special keyword for update models.  The other Updates (`om.uniform` and `om.rotate`) are also Models, but built in.

Next we assume that the decision to Flip or Roll depends on the player's current score and the context.

```python
import random

import olympusmons as om

twentyfour = om.Graph()
twentyfour.add_context("goal")
twentyfour.add_variable("turn")
twentyfour.add_variable("player_1_score")
twentyfour.add_variable("player_2_score")
twentyfour.add_pseudovariable("active_player_score", om.if_else(
    om.cond("turn", "==", "'player_1'"),
    "player_1_score",
    "player_2_score",
))
twentyfour.add_state("Turn", starting=True)
twentyfour.add_action("Flip")
twentyfour.add_action("Roll")

twentyfour.end_condition = om.om_or(
    om.cond("player_1_score", ">=", "goal"),
    om.cond("player_2_score", ">=", "goal"),
)

class TurnModel(om.Model):
    def __init__(self, **kwargs):
        self.model = None
        super().__init__(**kwargs)

    def sim(self, X):
        assert(self.model is not None)
        proba = self.model.predict_proba(X)[0]
        if random.random() < proba:
            return self.y_order[0]
        return self.y_order[1]

    def train(self, X, y)
        self.model = LogisticRegression().fit(X, y)

class FlipModel(om.Model):
    def predict(self, _):
        score = 0
        for _ in range(6):
            if random.random() < 0.5:
                score += 1
        return score

twentyfour.transition("Turn", TurnModel(inputs=["active_player_score", "goal"], outputs=["Flip", "Roll"]))
twentyfour.update("Flip", om.update_branch({
    "turn": om.rotate(["'player_1'", "'player_2'"]),
    "active_player_score": om.plus_equal(FlipModel()),
})
twentyfour.update("Roll", om.update_branch({
    "turn": om.rotate(["'player_1'", "'player_2'"]),
    "active_player_score": om.plus_equal(om.uniform([1, 2, 3, 4, 5, 6])),
})
```

Notice that `sim` uses self.y_order to determine which Action is in position 0 vs 1.  This is one of many variables that are set in the `__init__` of `om.Graph`; so it's important to still call this, even as we overwrite the child `__init__`.

Now, before we can run simulations, we need to train the model on the data.  For this, we'll use `PandasBulkTrainer`.  This assumes that the data is in a pandas DataFrame.  The "bulk" part means that the `om.Graph` needs to hold all game data in memory at once before training.  This is the only approach that's supported today.

```python
class TwentyFourGameGenerator(om.PandasBulkTrainer):
    def get_game(self):
        all_games = pd.read_sql("select game_id, goal from games;", con=_engine())
        for _, game in all_games.iterrows():
            df = pd.read_sql(f"select state, action, player_1_score, player_2_score from data where game_id={game_id};", con=_engine())
            yield ({"goal": game["goal"]}, df)

twentyfour.train(TwentyFourGameGenerator())
```

All we need to pass to train is a class with a function that yields (context_dict, df).  That df should represent a full game.  We expect this to have each row be a "turn": state, action-taken, variables at the beginning of the turn.

At this point, we should explain all that's going on.

Train does the following step:

- Check validity of the `om.Graph`.  (See Constraints below.)
- For each game in the dataset, start at the starting state, and walk through the "turns" (state, action, variables).
  - Check that the action agrees with the state, and that the exit condition has not been met.
  - Look at the Transition for the State; check that the Action is in the Transition's outputs.  Record row's inputs with the row's Action

///////
// Bedtime note:  I need better naming.  "The data", "the row", "input", are all kinda vague.

## Technical details

Constraints:

- Should be exactly 1 starting state and exit condition.
- Each Action should belong to exactly 1 State.
- State:Transition is 1:1
- Action:Update is 1:1
- [Optional] Graph should be connected, with all states reachable from starting node

Transitions and Updates are all Models, but Update Models use a special "update" output.

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


