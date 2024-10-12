from collections import defaultdict
import random
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

NOT_TRAINABLE_SENTINEL = "OM_NOT_TRAINABLE"
MAX_GAME_LENGTH = 10_000


class OMError(AssertionError):
    pass


class Journal(object):
    def __init__(self):
        self.df = None
        self.raw = defaultdict(list)

    def _add(self, action: ActionOrVariable, state: State, variables: Dict):
        self.raw["State"].append(state)
        self.raw["Action"].append(action)
        for k, v in variables.items():
            self.raw[k].append(v)

    def _build_df(self):
        column_order = ["State", "Action", "step"]
        column_order += [k for k in self.raw.keys() if k not in column_order]
        self.df = pd.DataFrame(self.raw, columns=column_order)


class Model(object):
    def __init__(self, name: str = "Default", input: Optional[List] = None, **kwargs):
        self.input = input or dict()
        self.name = name

    def sim(self, input: Dict) -> ActionOrVariable:
        raise OMError("Model {self.name} must implement sim().")

    def train(self, inputs: List[Dict], outputs: List[ActionOrVariable]) -> Any:
        return NOT_TRAINABLE_SENTINEL

    @property
    def trainable(self) -> bool:
        try:
            value = self.train([], [])
            return value != NOT_TRAINABLE_SENTINEL
        except:
            return False


class ModelMetadata(object):
    def __init__(self, name: str):
        self.name = name
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


class IncModelImpl(Model):
    def __init__(self, name, input: Optional[List] = None, **kwargs):
        if "target_variable" not in kwargs:
            raise OMError("IncModel must specify a target variable")
        self.delta = kwargs["delta"] or 1
        super().__init__(name=name, input=input, **kwargs)
        self.target_variable = kwargs["target_variable"]

    def sim(self, input: Dict) -> ActionOrVariable:
        return input[self.target_variable] + self.delta


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


class IncModel(ModelMetadata):
    def __init__(self, target_variable: str, delta: int = 1, **kwargs):
        self.target_variable = target_variable
        self.delta = delta
        self.input = [target_variable]
        super().__init__("IncModel")
        self._om_model_id = "IncModel"
        self._om_class = IncModelImpl
        self.model_args = {"target_variable": target_variable, "delta": delta}


class UDM(ModelMetadata):
    def __init__(self, model_name: str, **kwargs):
        if "input" not in kwargs:
            raise OMError(f"UDM {model_name} must specify an input")
        self.model_name = model_name
        self.input = kwargs["input"]
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
        pass

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
        return variable in model.input

    def _sim_model_by_name(
        self, name: str, variables: Dict, **kwargs
    ) -> ActionOrVariable:
        model = self.materialized_models_by_name[name]
        restricted_variables = {k: v for k, v in variables.items() if k in model.input}
        if kwargs.get("untrained_mode") and model.trainable:
            # This is so we can test correctness without training
            return NOT_TRAINABLE_SENTINEL
        return model.sim(restricted_variables)

    def sim(self, **kwargs) -> Dict:
        """
        Simulates the behavior of this Graph from start to finish.

        :param kwargs: If 'untrained_mode' is True, then untrained models will be skipped over
        :return: A dictionary of the final variables.
        """

        # Set up the data collector
        journal = Journal()

        # Start the simulation
        state = self.starting_state
        variables = {k: v for k, v in self.variables_initially.items()}
        variables["step"] = 0
        while not evaluate(self.end_condition, variables):
            if variables["step"] > MAX_GAME_LENGTH:
                raise OMError(
                    f"Game has exceeded maximum length {MAX_GAME_LENGTH}.  Perhaps you have an infinite loop?"
                )

            # Run the current model
            if (
                action := self._sim_model_by_name(state, variables, **kwargs)
            ) == NOT_TRAINABLE_SENTINEL:
                action = random.choice(self.reachable_actions_from_state[state])

            if action not in self.reachable_actions_from_state[state]:
                raise OMError(
                    f"Model for State {state} returns Action {action} that is not reachable"
                )

            # Record data before we change state or variables
            journal._add(action, state, variables)

            # Change state
            state = self.next_state_by_action[action]

            # Change variables
            for update in self.updates_by_action[action]:
                if (
                    new_variables := self._sim_model_by_name(
                        update, variables, **kwargs
                    )
                ) == NOT_TRAINABLE_SENTINEL:
                    continue
                if not isinstance(new_variables, list):
                    new_variables = [new_variables]
                if len(new_variables) != len(self.targets_by_update[update]):
                    raise OMError(
                        f"Model for State {state} returns {len(new_variables)} variables, but {len(self.targets_by_update[update])} variables were specified"
                    )
                for target, source in zip(
                    self.targets_by_update[update], new_variables
                ):
                    variables[target] = source
            variables["step"] += 1

        if kwargs.get("debug") == "screen":
            for data in journal.df.to_dict(orient="records"):
                print(data)

        if isinstance(kwargs.get("debug"), Journal):
            journal._build_df()
            kwargs.get("debug").df = journal.df
            kwargs.get("debug").raw = journal.raw
            kwargs.get("debug").csv = journal.df.to_csv()

        return variables


class GraphBuilder(object):
    def __init__(self, name):
        self.graph = Graph()
        self.graph.name = name

        self.mode = "Initial"
        self.mode_detail = None
        self.body_turnstile = False

        self.graph.starting_state = None
        self.graph.end_condition = None

        self.graph.states = list()
        self.graph.reachable_actions_from_state = defaultdict(list)
        self.graph.model_registry = dict()

        self.graph.next_state_by_action = dict()
        self.graph.updates_by_action = defaultdict(list)
        self.graph.targets_by_update = dict()

        self.graph.model_metadata_by_name = dict()
        self.graph.materialized_models_by_name = dict()

        self.graph.variables_initially = dict()

    def _materialize_models(self) -> None:
        for state, model_metadata in self.graph.model_metadata_by_name.items():
            model = self.graph.model_registry[model_metadata.name]
            self.graph.materialized_models_by_name[state] = model(
                model_metadata.name,
                model_metadata.input,
                **model_metadata.model_args,
            )

    def _mode(self, probe: str, probe_detail: str = "") -> None:
        needed_mode = {
            "set_starting_state": ["Initial"],
            "set_end_condition": ["Initial"],
            "Action": ["State", "Action", "update"],
            "update": ["Action"],
        }
        if probe in needed_mode and self.mode not in needed_mode[probe]:
            raise OMError(f"Can't run function {probe} in {self.mode} mode.")

        header_only = {
            "set_starting_state",
            "set_end_condition",
            "RegisterModel",
            "Variable",
        }
        if probe in header_only and self.body_turnstile:
            raise OMError(f"Cannot run function {probe} in body mode.")
        if "State" == probe:
            self.body_turnstile = True

        if probe == "RegisterModel":
            self.mode = probe
            self.mode_detail = probe_detail

        if probe == "Variable":
            self.mode = probe
            self.mode_detail = probe_detail

        if probe == "State":
            self.mode = probe
            self.mode_detail = probe_detail

        if probe == "Action":
            self.mode = probe
            if isinstance(self.mode_detail, list):
                # In format [State, Action]
                self.mode_detail = self.mode_detail[:1]
            else:
                self.mode_detail = [self.mode_detail]
            self.mode_detail = self.mode_detail + [probe_detail]

        if probe == "update":
            self.mode = probe
            assert isinstance(self.mode_detail, list)
            if len(self.mode_detail) == 3:
                self.mode_detail = self.mode_detail[:2]
            if isinstance(probe_detail, list):
                probe_detail = ",".join(probe_detail)
            self.mode_detail = self.mode_detail + [probe_detail]

    def _set_general_model(
        self,
        state: State,
        model_metadata: ModelMetadata,
        **kwargs,
    ) -> None:
        if not isinstance(model_metadata, ModelMetadata):
            raise OMError(f"Model for State {state} must be a Model")

        # Check against registry
        if model_metadata._om_model_id:
            if model_metadata._om_model_id not in self.graph.model_registry:
                self.graph.model_registry[
                    model_metadata._om_model_id
                ] = model_metadata._om_class
        if model_metadata.name not in self.graph.model_registry:
            raise OMError(f"UDM {model_metadata.name} is not registered")

        # Set metadata
        self.graph.model_metadata_by_name[state] = model_metadata

    def set_starting_state(self, starting_state: State) -> "GraphBuilder":
        self._mode("set_starting_state")
        self.graph.starting_state = starting_state
        return self

    def set_end_condition(self, end_condition: str) -> "GraphBuilder":
        self._mode("set_end_condition")
        self.graph.end_condition = end_condition
        return self

    def RegisterModel(self, model_name: str, model: Model, **kwargs) -> "GraphBuilder":
        self._mode("RegisterModel", model_name)
        self.graph.model_registry[model_name] = model
        return self

    def Variable(self, variable_name: str, initially: Any = None) -> "GraphBuilder":
        self._mode("Variable", variable_name)
        self.graph.variables_initially[variable_name] = initially
        return self

    def State(self, state: State, **kwargs) -> "GraphBuilder":
        self._mode("State", state)
        self.graph.states.append(state)

        if "model" not in kwargs:
            raise OMError(f"State {state} doesn't specify a model")
        self._set_general_model(state, kwargs["model"], **kwargs)

        return self

    def Action(self, action: Action, **kwargs) -> "GraphBuilder":
        self._mode("Action", action)

        self.graph.reachable_actions_from_state[self.mode_detail[0]].append(action)

        if "next_state" not in kwargs:
            raise OMError(f"Action {action} doesn't specify a next_state")
        self.graph.next_state_by_action[action] = kwargs["next_state"]

        return self

    def _get_update_name(self, _: List[str]) -> str:
        # This has to be universally unique
        return "::".join(self.mode_detail) + ":" + str(random.randint(0, 1000000))

    def update(
        self, variable_names: Union[str, List[str]], model: ModelMetadata, **kwargs
    ) -> "GraphBuilder":
        self._mode("update", variable_names)
        if isinstance(variable_names, str):
            variable_names = [variable_names]

        for variable_name in variable_names:
            if variable_name not in self.graph.variables_initially:
                raise OMError(f"Variable {variable_name} is not declared")

        update_name = self._get_update_name(variable_names)
        self.graph.updates_by_action[self.mode_detail[1]].append(update_name)
        self.graph.targets_by_update[update_name] = variable_names

        self._set_general_model(update_name, model, **kwargs)

        return self

    def Build(self, **kwargs) -> Graph:
        if not self.graph.starting_state:
            raise OMError("No starting state specified")
        if not self.graph.end_condition:
            raise OMError("No end condition specified")
        if self.graph.starting_state not in self.graph.states:
            raise OMError(
                f"Starting state {self.graph.starting_state} is not in states: {self.graph.states}"
            )

        for k, v in self.graph.next_state_by_action.items():
            if v not in self.graph.states:
                raise OMError(
                    f"Next state {v} specified by Action {k} is not in states: {self.graph.states}"
                )

        self._materialize_models()

        # We want to make sure that the models all return the correct values
        n_sims = kwargs.get("n_sims", 100)
        for _ in range(n_sims):
            self.graph.sim(untrained_mode=True)

        return self.graph
