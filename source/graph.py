from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import sympy


State = str
Action = str
Variable = str
VariableValue = Any
ActionOrVariable = Union[Action, VariableValue]


class OMError(AssertionError):
    pass


class Journal(object):
    def __init__(self):
        self.df = None
        self.raw = None


class Model(object):
    def __init__(self, name: str = "Default", input: Optional[List] = None, **kwargs):
        self.input = input or dict()
        self.name = name
        self.trainable = kwargs.get("trainable", False)

    def sim(self, input: Dict) -> ActionOrVariable:
        raise OMError("Model {self.name} must implement sim().")

    def train(self, inputs: List[Dict], outputs: List[ActionOrVariable]) -> None:
        if self.trainable:
            raise OMError("Model {self.name} must implement train().")


class ModelMetadata(object):
    def __init__(self, name: str):
        self.name = name
        self.trainable = False
        self._om_model_id = None
        self.model_args = dict()


class ConstModelImpl(Model):
    def __init__(self, name, input: Optional[List] = None, **kwargs):
        if "action" not in kwargs:
            raise OMError("ConstModel must specify an action")
        self.action = kwargs["action"]
        super().__init__(name=name, input=input, **kwargs)

    def sim(self, input: Dict) -> ActionOrVariable:
        return self.action


class ConstModel(ModelMetadata):
    def __init__(self, action: Action = None, **kwargs):
        if not action:
            raise OMError("ConstModel must specify an action")
        self.action = action
        self.input = list()
        super().__init__("ConstModel")
        self._om_model_id = "ConstModel"
        self._om_class = ConstModelImpl
        self.model_args = {"action": action}


class UDM(ModelMetadata):
    def __init__(self, model_name: str, **kwargs):
        if "input" not in kwargs:
            raise OMError(f"UDM {model_name} must specify an input")
        self.model_name = model_name
        self.input = input
        super().__init__(model_name)
        self.model_args.update(kwargs.get("model_args", {}))


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
        self.model_metadata_by_state = kwargs["model_metadata_by_state"]
        self.next_state_by_action = kwargs["next_state_by_action"]

        self.materialized_models_by_state = kwargs["materialized_models_by_state"]

        # self.steps = None

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
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

    @staticmethod
    def _variable_belongs_to_model_input(variable: Variable, model: Model) -> bool:
        if variable in ("step",):
            # Special variable, always include
            return True
        return variable in model.inputs

    def sim(self, **kwargs) -> Dict:
        data_cols = ["State", "Action", "step"] + []
        body = {k: [] for k in data_cols}

        working_state = self.starting_state
        variables = {"step": 0}
        while not evaluate(self.end_condition, variables):
            # Run sim on the current model.  Need to pass the right variables
            this_model = self.materialized_models_by_state[working_state]
            restricted_variables = dict()
            for k, v in variables.items():
                if Graph._variable_belongs_to_model_input(k, this_model):
                    restricted_variables[k] = v
            working_action = self.materialized_models_by_state[working_state].sim(
                input=restricted_variables
            )

            # Record data before we change state or variables
            body["State"].append(working_state)
            body["Action"].append(working_action)
            body["step"].append(variables["step"])

            # Change state
            working_state = self.next_state_by_action[working_action]

            # Change variables
            variables["step"] += 1

        if kwargs.get("debug"):
            if kwargs.get("debug") == "screen":
                print(",".join(data_cols))
                for i in range(len(body["State"])):
                    print(",".join([str(body[k][i]) for k in data_cols]))

            if isinstance(kwargs.get("debug"), Journal):
                kwargs.get("debug").df = pd.DataFrame(body, columns=data_cols)
                kwargs.get("debug").raw = body
                kwargs.get("debug").csv = kwargs.get("debug").df.to_csv()

        return variables


class GraphBuilder(object):
    def __init__(self, name):
        self.name = name
        self.mode = "Initial"
        self.mode_detail = None
        self.body_turnstile = False

        self.starting_state = None
        self.end_condition = None

        self.states = list()
        self.reachable_actions_from_state = defaultdict(list)
        self.model_registry = dict()
        self.model_metadata_by_state = dict()
        self.materialized_models_by_state = dict()
        self.next_state_by_action = dict()

    def _materialize_models(self) -> None:
        for state, model_metadata in self.model_metadata_by_state.items():
            model = self.model_registry[model_metadata.name]
            self.materialized_models_by_state[state] = model(
                model_metadata.name,
                model_metadata.input,
                **model_metadata.model_args,
            )

    def _mode(self, probe: str, probe_detail: str = "") -> None:
        needed_mode = {
            "set_starting_state": ["Initial"],
            "set_end_condition": ["Initial"],
            "Action": ["State", "Action"],
        }
        if probe in needed_mode and self.mode not in needed_mode[probe]:
            raise OMError(f"Can't run function {probe} in {self.mode} mode.")

        header_only = {"set_starting_state", "set_end_condition", "RegisterModel"}
        if probe in header_only and self.body_turnstile:
            raise OMError(f"Cannot run function {probe} in body mode.")
        if "State" == probe:
            self.body_turnstile = True

        if probe == "RegisterModel":
            self.mode = probe
            self.mode_detail = probe_detail

        if probe == "State":
            self.mode = probe
            self.mode_detail = probe_detail

        if probe == "Action":
            self.mode = probe
            if isinstance(self.mode_detail, list):
                # In format [State, Action]
                self.mode_detail = self.mode_detail[0]
            self.mode_detail = [self.mode_detail, probe_detail]

    def _set_state_model(self, state: State, model_metadata: Model) -> None:
        if not isinstance(model_metadata, ModelMetadata):
            raise OMError(f"Model for State {state} must be a Model")

        if model_metadata._om_model_id:
            if model_metadata._om_model_id not in self.model_registry:
                self.model_registry[
                    model_metadata._om_model_id
                ] = model_metadata._om_class

        if model_metadata.name not in self.model_registry:
            raise OMError(f"UDM {model_metadata.name} is not registered")

        self.model_metadata_by_state[state] = model_metadata

    def set_starting_state(self, starting_state: State) -> "GraphBuilder":
        self._mode("set_starting_state")
        self.starting_state = starting_state
        return self

    def set_end_condition(self, end_condition: str) -> "GraphBuilder":
        self._mode("set_end_condition")
        self.end_condition = end_condition
        return self

    def RegisterModel(self, model_name: str, model: Model, **kwargs) -> "GraphBuilder":
        self._mode("RegisterModel", model_name)
        self.model_registry[model_name] = model
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

        self._materialize_models()

        # We want to make sure that the models all return the correct values
        n_sims = kwargs.get("n_sims", 100)
        for _ in range(n_sims):
            for state, model_metadata in self.model_metadata_by_state.items():
                if model_metadata.trainable:
                    # Can't really check these ones
                    continue

                model = self.materialized_models_by_state[state]
                # TODO: Make some sampling logic
                result_action = model.sim(input={"step": 1})
                if result_action not in self.reachable_actions_from_state[state]:
                    raise OMError(
                        f"Model for State {state} returns Action {result_action} that is not reachable"
                    )

        return Graph(
            name=self.name,
            starting_state=self.starting_state,
            end_condition=self.end_condition,
            states=self.states,
            reachable_actions_from_state=self.reachable_actions_from_state,
            model_registry=self.model_registry,
            model_metadata_by_state=self.model_metadata_by_state,
            next_state_by_action=self.next_state_by_action,
            materialized_models_by_state=self.materialized_models_by_state,
        )
