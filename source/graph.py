from collections import defaultdict
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import networkx as nx
import sympy


State = str
Action = str
VariableValue = Any
ActionOrVariable = Union[Action, VariableValue]


class OMError(AssertionError):
    pass


class Model(object):
    def __init__(self):
        self._om_model_args = {}
        self._om_model_id = None
        self.trainable = False
        self.class_name = "Default"

    def sim(self, input: Dict) -> ActionOrVariable:
        raise OMError(
            "Model {self.class_name} must implement sim().  (Set self.class_name for better debugging.)"
        )

    def train(self, inputs: List[Dict], outputs: List[ActionOrVariable]) -> None:
        if self.trainable:
            raise OMError(
                "Model {self.class_name} must implement train().  (Set self.class_name for better debugging.)"
            )


class ConstModel(Model):
    def __init__(self, action: Action = None):
        if not action:
            raise OMError("ConstModel must specify an action")
        self.action = action
        super().__init__()
        self.class_name = "ConstModel"
        self._om_model_id = "ConstModel"
        self._om_model_args = {"action": action}

    def sim(self, input: Dict) -> ActionOrVariable:
        return self.action


def evaluate(expr, values):
    """A layer of abstraction to SymPy's eval() function"""
    expr = sympy.simplify(expr)
    for key, value in values.items():
        expr = expr.subs(key, value)
    return expr


class Graph(object):
    def __init__(self, **kwargs):
        self.name = kwargs["name"]

        self.starting_state = kwargs["starting_state"]
        self.end_condition = kwargs["end_condition"]

        self.states = kwargs["states"]
        self.reachable_actions_from_state = kwargs["reachable_actions_from_state"]
        self.model_registry = kwargs["model_registry"]
        self.models_by_state = kwargs["models_by_state"]
        self.model_args_by_state = kwargs["model_args_by_state"]
        self.next_state_by_action = kwargs["next_state_by_action"]

        self.materialized_models_by_state = dict()
        self._materialize_models()

        self.steps = None
    
    def _materialize_models(self) -> None:
        for state, model_name in self.models_by_state.items():
            model = self.model_registry[model_name]
            self.materialized_models_by_state[state] = model(**self.model_args_by_state.get(state, dict()))

    def draw(self) -> None:
        G = nx.DiGraph()
        for state in self.states:
            G.add_node(state)

        for state, actions in self.reachable_actions_from_state.items():
            for action in actions:
                G.add_edge(state, self.next_state_by_action[action], label=action)

        # Define position for nodes (using spring layout for better positioning)
        pos = nx.spring_layout(G)

        # Draw nodes
        nx.draw(G, pos, with_labels=True)

        # Draw edges
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

    def sim(self, **kwargs) -> Dict:
        working_state = self.starting_state
        variables = {"step": 0}
        while not evaluate(self.end_condition, variables):
            restricted_variables = {k: v for k, v in variables.items()}
            working_action = self.materialized_models_by_state[working_state].sim(input=restricted_variables)

            working_state = self.next_state_by_action[working_action]
            break

        return variables


class GraphBuilder(object):
    def __init__(self, name):
        self.name = name
        self.mode = "Initial"
        self.mode_detail = None

        self.starting_state = None
        self.end_condition = None

        self.states = list()
        self.reachable_actions_from_state = defaultdict(list)
        self.model_registry = dict()
        self.models_by_state = dict()
        self.model_args_by_state = dict()
        self.next_state_by_action = dict()

    def _mode(self, probe: str, probe_detail: str = "") -> None:
        needed_mode = {
            "set_starting_state": "Initial",
            "set_end_condition": "Initial",
            "Action": "State",
        }
        if probe in needed_mode and needed_mode[probe] != self.mode:
            raise OMError(f"Can't run function {probe} in {self.mode} mode.")

        if probe == "State":
            self.mode = probe
            self.mode_detail = probe_detail

        if probe == "Action":
            self.mode = probe
            self.mode_detail = [self.mode_detail, probe_detail]

    def _set_state_model(self, state: State, model: Model) -> None:
        if not isinstance(model, str) and not isinstance(model, Model):
            raise OMError(f"Model for State {state} must be a string or a Model")

        if isinstance(model, Model):
            if model._om_model_id:
                self.model_args_by_state[state] = model._om_model_args
                if model not in self.model_registry:
                    self.model_registry[model._om_model_id] = model.__class__
                model = model._om_model_id
            else:
                raise OMError(
                    f"UDM {model.name} must be registered and referred to by name"
                )

        if model not in self.model_registry:
            raise OMError(f"UDM {model} is not registered")

        self.models_by_state[state] = model

    def set_starting_state(self, starting_state: State) -> "GraphBuilder":
        self._mode("set_starting_state")
        self.starting_state = starting_state
        return self

    def set_end_condition(self, end_condition: str) -> "GraphBuilder":
        self._mode("set_end_condition")
        self.end_condition = end_condition
        return self

    def State(self, state: State, **kwargs) -> "GraphBuilder":
        self._mode("State", state)
        self.states.append(state)

        if "model" not in kwargs:
            raise OMError(f"State {state} doesn't specify a model")
        self._set_state_model(state, kwargs["model"])
        if "model_args" in kwargs:
            self.model_args_by_state[state] = kwargs["model_args"]

        return self

    def Action(self, action: Action, **kwargs) -> "GraphBuilder":
        self._mode("Action", action)

        self.reachable_actions_from_state[self.mode_detail[0]].append(action)

        if "next_state" not in kwargs:
            raise OMError(f"Action {action} doesn't specify a next_state")
        self.next_state_by_action[action] = kwargs["next_state"]

        return self

    def Build(self, **kwargs) -> Graph:
        if self.starting_state not in self.states:
            raise OMError(
                f"Starting state {self.starting_state} is not in states: {self.states}"
            )
        for k, v in self.next_state_by_action.items():
            if v not in self.states:
                raise OMError(
                    f"Next state {v} specified by Action {k} is not in states: {self.states}"
                )

        # We want to make sure that the models all return the correct values
        n_sims = kwargs.get("n_sims", 100)
        for _ in range(n_sims):
            for state, model_name in self.models_by_state.items():
                model = self.model_registry[model_name](
                    **self.model_args_by_state.get(state, dict())
                )
                if model.trainable:
                    result_action = model.sim(input=dict())
                    if result_action not in self.reachable_actions_from_state[state]:
                        raise OMError(
                            f"Model for State {state} returns Action {result_action} that is not reachable"
                        )

        return Graph(
            name = self.name, 
            starting_state = self.starting_state, 
            end_condition = self.end_condition, 
            states = self.states, 
            reachable_actions_from_state = self.reachable_actions_from_state, 
            model_registry = self.model_registry, 
            models_by_state = self.models_by_state, 
            model_args_by_state = self.model_args_by_state, 
            next_state_by_action = self.next_state_by_action, 
        )
