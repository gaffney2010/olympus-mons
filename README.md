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
    .set_starting_state("A")
    .set_end_condition("step >= 5")
    .State("A", model=om.ConstModel("to_b")).Action("to_b", next_state="B")
    .State("B", model=om.ConstModel("to_c")).Action("to_c", next_state="C")
    .State("C", model=om.ConstModel("to_a")).Action("to_a", next_state="A")
    .Build()
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
    def sim(self, input):
        if random.random() < 0.5:
            return "to_b"
        return "to_c"


graph = (
    om.GraphBuilder("Out and back")
    .set_starting_state("A")
    .set_end_condition("step >= 5")
    .RegisterModel("SplitA", SplitA)
    .State("A", model=om.UDM("SplitA", input=[])).Action("to_b", next_state="B").Action("to_c", next_state="C")
    .State("B", model=om.ConstModel("to_a_from_b")).Action("to_a_from_b", next_state="A")
    .State("C", model=om.ConstModel("to_a_from_c")).Action("to_a_from_c", next_state="A")
    .Build()
)
```

A few things to notice:

- The action names must be globally unique, so we added the "from" part this time.
- The SplitA model gets referenced with `UDM` (User-Defined Model), along with input variables.  In this example, we don't have any variables, except the `step`, which isn't used in the model logic.
- User models need to be "registered", then referenced by name.  Everytime the Model is referenced, a new instance of the model will be created.
- Model has a `sim` function that does simulations.  It takes the parameter `input` which is a dictionary of all the variable values.
- `sim` returns an Action name

We could have instead made a function that depens on `step` like this:

```python
class SplitA(om.Model):
    def sim(self, input):
        if random.random() < 0.5 / (input["step"] + 1):  # step is 0-indexed
            return "B"
        return "C"


graph = (
    om.GraphBuilder("Out and back")
    ....
    .RegisterModel("SplitA", SplitA)
    .State("A", model=om.UDM("SplitA", input=["step"])).Action("to_b", next_state="B").Action("to_c", next_state="C")
    ....
)
```

om.ConstModel is a Model factory that builds a model like `SplitA` but that always returns a fixed State.  That factory needs to know what the target is.

The power of OM is its ability to encode states as variables.  For this next example, let's assume that the probability of visiting state B from state A is equal to 1/2 raised to number of times B has been previously visited.  We can do this by introducing a variable, `num_b_visits`.

```python
import random

import olympusmons as om


class SplitA(om.Model):
    def sim(self, input):
        if random.random() < (0.5)**input["num_b_visits"]:
            return "to_b"
        return "to_c"


class IncNumBVisits(om.Model):
    def sim(self, input):
        return input["num_b_visits"] + 1


graph = (
    om.GraphBuilder("Out and back")
    .set_starting_state("A")
    .set_end_condition("step >= 5")
    .RegisterModel("SplitA", SplitA)
    .RegisterModel("IncNumBVisits", IncNumBVisits)
    .Variable("num_b_visits", initially=0)
    .State("A", model=om.UDM("SplitA", input=["num_b_visits"]))
        .Action("to_b", next_state="B")
        .Action("to_c", next_state="C")
    .State("B", model=om.ConstModel("to_a_from_b"))
        .Action("to_a_from_b", next_state="A").update("num_b_visits", om.UDM("IncNumBVisits", input=["num_b_visits"]))
    .State("C", model=om.ConstModel("to_a_from_c"))
        .Action("to_a_from_c", next_state="A")
    .Build()
)
```

A few things to note:

- Since the code got bigger, I used tabbing to help with readability.
- Now that `SplitA` uses "num_b_visits", that has to be passed into A's State setup.  This tells the graph to pass this as part of the `input` dict.
- Update models like `IncNumBVisits` are also Model types, but they return values rather than Actions.

`IncNumBVisits` is a common enough pattern that it has a built-in factory.

```python
graph = (
    om.GraphBuilder("Out and back")
    ....
    .RegisterModel("SplitA", SplitA)
    .Variable("num_b_visits", initially=0)
    ....
    .State("B", model=om.ConstModel("to_a_from_b"))
        .Action("to_a_from_b", next_state="A").update("num_b_visits", om.IncModel("num_b_visits"))
    ....
```

### Context and Journals

We may want to run different simulations for different setups.  Say we want to run with different number of steps or change the decay factor, 0.5.  We could create different graphs, but that's too noisy.  We could make variables that we never change, but actually that requires creating a new graph.  For this reason, we create game-level constants, called Contexts.

```python
import random

import olympusmons as om


class SplitA(om.Model):
    def sim(self, input):
        if random.random() < input["decay_factor"]**input["num_b_visits"]:
            return "to_b"
        return "to_c"


class IncNumBVisits(om.Model):
    def sim(self, input):
        return input["num_b_visits"] + 1


graph = (
    om.GraphBuilder("Out and back")
    .set_starting_state("A")
    .set_end_condition("step >= total_steps")
    .RegisterModel("SplitA", SplitA)
    .RegisterModel("IncNumBVisits", IncNumBVisits)
    .Context("total_steps", default=5)
    .Context("decay_factor", default=0.5, validator="0.0 < decay_factor < 1.0")
    .Variable("num_b_visits", initially=0)
    .State("A", model=om.UDM("SplitA", input=["num_b_visits", "decay_factor"]))
        .Action("to_b", next_state="B").Action("to_c", next_state="C")
    .State("B", model=om.ConstModel("to_a_from_b"))
        .Action("to_a_from_b", next_state="A").update("num_b_visits", om.UDM("IncNumBVisits", input=["num_b_visits"]))
    .State("C", model=om.ConstModel("to_a_from_c"))
        .Action("to_a_from_c", next_state="A")
    .Build()
)
```

Now we can change Context in the sim function.

```python
# Run with ten steps
graph.sim(debug="screen", context={"total_steps": 10})
# Run with twenty steps
graph.sim(debug="screen", context={"total_steps": 20})
```

So far we've only written to screen, but you can also write to a Journal.  After passing a Journal to debug, the Journal will contain a field call `df` with the simulation written as a pandas DataFrame.

Let's say that you want to answer the question:  "What's the expected number of steps spent in State "B" out of the first 100 steps?"

```python
N_SIMS = 1_000
num = 0
for _ in range(N_SIMS):
    journal = om.Journal()
    graph.sim(debug=journal, context={"total_steps": 100})
    num += len(journal.df[journal.df["State"] == "B"])
print(f"Avg steps in State B: {num / N_SIMS}")
```

### Training

Next we'll assume that we don't know what decay_factor is, and fit it.

```python
import random

import olympusmons as om


class SplitA(om.Model):
    def __init__(self):
        self.is_fit = False
        self.decay_factor = None
        super().__init__(**kwargs)

    def sim(self, input):
        assert(self.is_fit)
        if random.random() < self.decay_factor**input["num_b_visits"]:
            return "to_b"
        return "to_c"

    def train(self, input, output):
        count_by_num_b_visits = dict()
        goto_b_by_num_b_visits = dict()
        for i, o in zip(input, output):
            count_by_num_b_visits[i["num_b_visits"]] += 1
            if "B" == o:
                goto_b_by_num_b_visits[i["num_b_visits"]] += 1

        self.decay_factor = 0
        for k, v in goto_b_by_num_b_visits.items():
            tot = count_by_num_b_visits[k]
            weight = tot / len(output)
            self.decay_factor += weight * (v / tot) ** (1/k)
        self.is_fit = True


class IncNumBVisits(om.Model):
    def sim(self, input):
        return input["num_b_visits"] + 1


trainable_graph = (
    om.GraphBuilder("Out and back")
    .set_starting_state("A")
    .set_end_condition("step >= total_steps")
    .RegisterModel("SplitA", SplitA)
    .RegisterModel("IncNumBVisits", IncNumBVisits)
    .Context("total_steps", default=5)
    .Variable("num_b_visits", initially=0)
    .State("A", model=om.UDM("SplitA", input=["num_b_visits"]))
        .Action("to_b", next_state="B").Action("to_c", next_state="C")
    ....
)
```

Don't worry too much about the math of the `train` function; it's not even a very good model.  But a few things to notice:

- If we add an `__init__` to an `om.Model`, then we _must_ initialize the parent class with super.
- `train` takes as arguments two equal length lists.  The first, `input`, is a list of dicts that look identical to the input of `sim`.  The second, `output`, is a list of values that look identical to the output of `sim`.
- We check ourselves if the model is trained before using it.  We saw above that sims can be done without training.  It's up to the application side to determine if that should be allow.
- Note that `IncNumBVisits` gets "trained" too, but the default `train` function on `om.Model` does nothing.

OM uses a Bring-Your-Own-Model approach.  So any modeling you want to do you have to build yourself.  Often this will just mean wrapping models from sklearn, but we didn't want to depend on that complex, ever-evolving library directly.

Now that you have this graph setup with a trainable model in it, you can provide it data in a training step.  In the example below, we will provide data from DataFrames.  Each DataFrame represents one full simulation.  These DataFrames generally can be obtained from a database or built however you want.  But to make this concrete, we'll provide data from `graph` (our earlier, non-trainable graph) using Journals; don't confuse with `trainable_graph` (our new graph).

```python
class MyGraphGenerator(om.PandasBulkTrainer):
    def get_game(self):
        global graph
        for _ in range(100):
            journal = om.Journal()
            graph.sim(debug=journal, context={"total_steps": 100})
            yield om.PandasDatum(journal.df, context={"total_steps": 100})

trainable_graph.train(MyGraphGenerator())
```

A few notes:

- The name PandasBulkTrainer means that the graph must be trained in bulk.  That is all data must be held in memory at the same time.  This is the only supported method right now.  But if your training set is quite large, this could pose problems.
- Trainers must be generators, meaning that they `yield` simulations as they go.  `MyGraphGenerator` generates 100 total games.
- Note that you have to specify any non-default context.

### Observation and Saving

After training, we can observe what was learned by looking at `trainable_graph["SplitA"].decay_factor`.  Observing models on updates may be a little more complicated because updates are not typically named.  What we recommend instead is to implement the `describe` function on the Model.

```python
class SplitA(om.Model):
    def __init__(self):
        self.is_fit = False
        self.decay_factor = None
        super().__init__(**kwargs)

    ...

    def describe(self):
        return f"SplitA Model: {self.decay_factor=}"
```

This may return any string you want, and will be used for various Graph descriptions.

Last you may want to save and load your graph.  This requires that all models have `save` and `load` functions, which write and read jsons.  You must be able to rebuild your model from the `load` function for this to work.

```python
import json


class SplitA(om.Model):
    def __init__(self):
        self.is_fit = False
        self.decay_factor = None
        super().__init__(**kwargs)

    ...

    def save(self):
        assert(self.is_fit)
        pre_json = {"decay_factor": self.decay_factor}
        return json.dumps(pre_json)

    @staticmethod
    def load(from_json):
        post_json = json.loads(from_json)
        result = SplitA()
        result.decay_factor = post_json["decay_factor"]
        result.is_fit = True
        return result
```

Note that this doesn't actually save your model, only the json.  If in the loading stage you register a different or newer version of the model with the same name, then this may break your saves.  Compatibility is the application code's concern.

## NFL

```python
graph = (
    om.GraphBuilder("NFL")
    .set_starting_state("Coin Flip")
    .set_end_condition("time <= 0")
    .Context("away_team")
    .Context("home_team")
    .Context("away_team_starting_score")
    .Context("home_team_starting_score")
    .RegisterModel...
    .Variable("down", initially=None)
    .Variable("yards_to_td", initially=None)
    .Variable("first_down_line", initialy=None)
    .Variable("time", initially=60*30)
    .Variable("offense", initally=None)
    .Variable("away_team_score")
    .Variable("home_team_score")
    .PseudoVariable("offense_is_home", "1 if offense == home_team else 0")
    .PseudoVariable("defence", "away_team if offense == home_team else home_team")
    .PseudoVariable("offense_score", "away_team_score if offense == away_team else home_team")
    .PseudoVariable("defence_score", "away_team_score if defence == away_team else home_team")
    .PseudoVariable("offense_net_score", "offense_score - defence_score")
    .PseudoVariable("yards_to_first", "yards_to_td - first_down_line")
    .PseudoVariable("-2", "-2").PseudoVariable("1", "1").PseudoVariable("3", "3").PseudoVariable("6", "6")
    .State("Coin Flip", model=om.Constant("Coin Flip Winner"))
        .Action("Coin Flip Winner", next_state="Kick-Off")
            .update("offense", om.UDF("CoinFlipResult", input=[])
            .update("away_team_score", om.Equal("away_team_starting_score"))
            .update("home_team_score", om.Equal("home_team_starting_score"))
    .State("Kick-Off", model=om.Constant("Kick-Off Return"))
        .pre_update(["offense", "yards_to_td"], om.UDF("TurnOverLogic", input=["away_team", "home_team", "yards_to_td"]))
        .Action("Kick-Off Return", next_state="Down")
            .update("down", om.Constant(0))
            .update("yards_to_td", om.UDF("Regression", input=["offense", "defence", "offense_is_home"]))
            .update("first_down_line", om.UDF("YardsPlusTen", input=["yards_to_td"])
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
    .State("Pre-Down", model=om.UDF("DetermineWhichDown", input=["down", "yard"]))
        # Adds 1 down, calculates if first down, fourth down, or a turnover
        .pre_update(["down", "first_down_line", "offense"], om.UDF("DownCalculator", input=["down", "yards_to_td", "offense"]))
        .Action("Down from Pre-Down", next_state="Down")
        .Action("Fourth Down Decision from Pre-Down", next_state="Fourth Down Decision")
        .Action("Touchdown from Pre-Down", next_state="Point After Decision")
    .State("Down", model=om.UDF("RandomForest", input=["offense", "defence", "down", "yards_to_td", "yards_to_first", "offense_net_score", "offense_is_home"])
        .Action("Pass", next_state="In the air")
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("Run", next_state="Pre-Down")
            .update("yards_to_td", om.Plus("yards_to_td", om.UDF("Non-Parametric", input=["offense", "defence"])))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("Sack", next_state="Pre-Down")
            .update("yards_to_td", om.Plus("yards_to_td", om.UDF("Non-Parametric", input=["offense", "defence"])))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("Fumble", next_state="Live Ball")
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        # We will combine these into the previous play.
        # .Action("PenaltyRedoDown", next_state="Pre-Down")
        #     .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("Safety", next_state="Kick-Off")
            .update(["player_1_score", "player_2_score"], om.UDF("UpdateScore", input=["offense", "-2"])
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
    .State("In the air", model=om.UDF("Regression", input=["offense", "defence", "offense_is_home"])
        .Action("Completion", next_state="Pre-Down")
            .update("yards_to_td", om.Plus("yards_to_td", om.UDF("Non-Parametric", input=["offense", "defence"])))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("Incomplete", next_state="Pre-Down")
            .update("yards_to_td", om.Plus("yards_to_td", om.UDF("Non-Parametric", input=["offense", "defence"])))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("Interception", next_state="Pre-Down")
            .update("yards_to_td", om.Plus("yards_to_td", om.UDF("Non-Parametric", input=["offense", "defence"])))
            .update(["offense", "yards_to_td"], om.UDF("TurnOverLogic", input=["away_team", "home_team", "yards_to_td"]))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("Fumble After Completion", next_state="Live Ball")
            # We will sum the net yards before an afte the completion
            .update("yards_to_td", om.Plus("yards_to_td", om.UDF("Non-Parametric", input=["offense", "defence"])))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
    .State("Live Ball", model=om.UDF("Regression", input=["offense", "defence", "offense_is_home"])
        .Action("Recovery", next_state="Pre-Down")
            .update("yards_to_td", om.Plus("yards_to_td", om.UDF("Non-Parametric", input=["offense", "defence"])))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("Turn-Over", next_state="Pre-Down")
            .update("yards_to_td", om.Plus("yards_to_td", om.UDF("Non-Parametric", input=["offense", "defence"])))
            .update(["offense", "yards_to_td"], om.UDF("TurnOverLogic", input=["away_team", "home_team", "yards_to_td"]))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        # If there are multiple fumbles, just count them as 1.
    .State("Fourth Down Decision", model=om.UDF("RandomForest", input=["offense", "yards_to_first", "offense_net_score", "time"])
        .Action("Punt", next_state="Down")
            .update("yards_to_td", om.Plus("yards_to_td", om.UDF("Non-Parametric", input=["offense", "defence"])))
            .update(["offense", "yards_to_td"], om.UDF("TurnOverLogic", input=["away_team", "home_team", "yards_to_td"]))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("Field Goal", next_state="Field Goal Attempt")
        .Action("Go for it", next_state="Fourth Down")
    .State("Field Goal Attempt", model=om.UDF("Non-Parametric", input=["yards_to_td", "offense"])
        .Action("FG Success", next_state="Kick-Off")
            .update(["player_1_score", "player_2_score"], om.UDF("UpdateScore", input=["offense", "3"]))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("FG Miss", next_state="Down")
            .update("down", om.Constant(0)))
            .update("yards_to_td", om.UDF("KickOffReturnYard", input=["offense", "defence"]))
            .update(["offense", "yards_to_td"], om.UDF("TurnOverLogic", input=["away_team", "home_team", "yards_to_td"]))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
    .State("Fourth Down", model=om.UDF("RandomForest", input=["offense", "defence", "down", "offense_net_score"]))
        .Action("Make it", next_state="Down") 
            .update("down", om.Constant(0)))
            .update("yards_to_td", om.UDF("KickOffReturnYard", input=["offense", "defence"]))
            .update("first_down_line", om.UDF("YardsPlusTen", input=["yards_to_td"]))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("FD Touchdown", next_state="Point After Decision") 
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
        .Action("Turn-Over on Downs", next_state="Down")
            .update("down", om.Plus(om.Constant(1), "1"))
            .update("yards_to_td", om.UDF("KickOffReturnYard", input=["offense", "defence"]))
            .update(["offense", "yards_to_td"], om.UDF("TurnOverLogic", input=["away_team", "home_team", "yards_to_td"]))
            .update("time", om.Plus("time", om.UDF("Non-Parametric", input=[])))
    .State("Point After Decision", model=om.UDF("Regression", input=["offense", "offense_net_score", "time"]))
        .pre_update(["player_1_score", "player_2_score"], om.UDF("UpdateScore", input=["offense", "6"])
        .Action("Decide PAT", next_state="PAT")
        .Action("Decide PAT-2", next_state="PAT-2")
    .State("PAT", model=om.UDF("Regression", input=["offense", "offense_is_home"]))
        # We have to throw away PAT safeties, it's just too much, I'm sorry.
        .Action("PAT Success", next_state="Kick-off")
            .update(["player_1_score", "player_2_score"], om.UDF("UpdateScore", input=["offense", "1"]))
        .Action("PAT Failure", next_state="Kick-off")
    .State("PAT-2", model=om.UDF("Regression", input=["offense", "defence", "offense_is_home"]))
        .Action("PAT-2 Success", next_state="Kick-off")
            .update(["player_1_score", "player_2_score"], om.UDF("UpdateScore", input=["offense", "2"]))
        .Action("PAT-2 Failure", next_state="Kick-off")
    .Build()
)
```

The code here is complex, because the graph we designed is complex.  The GraphBuilder is [DAMP](https://testing.googleblog.com/2019/12/testing-on-toilet-tests-too-dry-make.html) by design.

Some notes:

- Because football has two half, we simulate the halves each as their own game with the use of the starting_score Context variables, and use the coin flip to load these in.
- PseudoVariables may depend on other PseudoVariables.  The order they're written here determines evaluation order.
- I need forking logic to decide if I should travel to the `Down` or the `FourthDown` state.  I implement a special state for this.  I can also check for touchdowns and first downs with this logic step.  The tricky thing about pretend states like this is that we have to record these into the training data, but an error will be raised if you forget.
- Observe how different features are used for different models and how these are kinda intuitive.
- Notice that we're able to update two variables at once in some places.  The `sim` function must return a tuple, and these Models should not be trained.  "offense_score" as a pseudo-variable is convenient, but recall that we can't set the value of a pseudo-variable.
- Notice that we have to code literals like PseudoVariable, so that they can be used in `sim` functions.  This is a hack to avoid having to create and register different Models.
- Sometimes we reuse models like "Regression".  Because the graph makes a new instance of this every time it's called, we aren't saying that these different regressions have anything in common.
- This uses `pre_update` which is short-hand for attaching updates to every incoming action.

There are lots of decisions made here.  For example, I could make "Run" and "Sack" the same with a unified model to predict net yards.  I call a fumble-after-completion a "Fumble" without a "Completion" first.  Different decisions will lead to different models with different performance.  This is an art and a science.

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


