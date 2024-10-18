from collections import defaultdict
import copy
import random
from typing import Any, Dict, List, Optional, Union

import attr
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import sympy


ActionOrVariable = Any

NOT_TRAINABLE_SENTINEL = "OM_NOT_TRAINABLE"
MAX_GAME_LENGTH = 5_000


class OMError(AssertionError):
    pass


def _print_dict(d: Dict) -> None:
    """Alphabetizes dicts for predictable error messages."""
    result = list()
    for k in sorted(list(d.keys())):
        result.append(f"{k}: {d[k]}")
    return "{" + ", ".join(result) + "}"


class Journal(object):
    def __init__(self):
        self.df = None
        self.raw = defaultdict(list)
        self.errors = list()

    def _record_error(self, game_num: int, row_num: int, error: Dict, msg: str) -> None:
        self.errors.append(
            {
                "game_num": game_num,
                "row_num": row_num,
                "errant row": error,
                "error": msg,
            }
        )

    def _add(self, action: ActionOrVariable, state: str, variables: Dict) -> None:
        self.raw["State"].append(state)
        self.raw["Action"].append(action)
        for k, v in variables.items():
            self.raw[k].append(v)

    def _build_df(self) -> None:
        column_order = ["State", "Action", "step"]
        column_order += [k for k in self.raw.keys() if k not in column_order]
        self.df = pd.DataFrame(self.raw, columns=column_order)


class BulkTrainer(object):
    pass


class PandasBulkTrainer(BulkTrainer):
    def get_game(self):
        raise OMError("PandasBulkTrainer must implement get_game()")


class PandasDatum(object):
    def __init__(self, df: pd.DataFrame, context: Dict = None):
        if not context:
            context = dict()
        self.df = df
        self.context = context


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
    def __init__(self, action: Optional[str] = None, **kwargs):
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


@attr.s()
class State(object):
    name: str = attr.ib()
    # TODO: Refactor ModelMetadata
    metadata: ModelMetadata = attr.ib()
    reachable_actions: List["Action"] = attr.ib(default=attr.Factory(list))


@attr.s()
class Action(object):
    name: str = attr.ib()
    next_state: str = attr.ib()
    updates: List["Update"] = attr.ib(default=attr.Factory(list))


@attr.s()
class Update(object):
    name: str = attr.ib()
    metadata: ModelMetadata = attr.ib()
    targets: str = attr.ib()


class Expr(object):
    def __init__(self, expr: str):
        self.expr = expr
        try:
            self.symp_expr = sympy.sympify(expr)
        except:
            raise OMError(f"Invalid validator {expr}")

    def to_dict(self):
        return {"expr": self.expr}

    def extract_variables(self) -> List[str]:
        return list(self.symp_expr.free_symbols)

    def evaluate(self, values, allow_errors: bool = True) -> bool:
        """A layer of abstraction to SymPy's eval() function"""
        # Need to strip out Nones otherwise this thing breaks for reasons I don't understand
        values = {k: v for k, v in values.items() if v is not None}

        expr = self.symp_expr
        try:
            for key, value in values.items():
                expr = expr.subs(key, value)
            if expr == True:
                return True
            if expr == False:
                return False
            raise Exception("Expected a boolean expression")
        except:
            if not allow_errors:
                raise OMError(f"Expression {expr} fails unexpectedly on {values}.")
            return False

    def __repr__(self):
        return self.expr


class Graph(object):
    def __init__(self, **kwargs):
        pass

    def draw(self) -> None:
        G = nx.DiGraph()
        for state in self.states:
            G.add_node(state)

        for _, state in self.states.items():
            for action in state.reachable_actions:
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
    def _variable_belongs_to_model_input(variable: str, model: Model) -> bool:
        if variable in ("step",):
            # Special variable, always include
            return True
        return variable in model.input

    def _sim_model_by_name(
        self, name: str, variables: Dict, **kwargs
    ) -> ActionOrVariable:
        model = self.materialized_models[name]
        restricted_variables = {k: v for k, v in variables.items() if k in model.input}
        if kwargs.get("untrained_mode") and model.trainable:
            # This is so we can test correctness without training
            return NOT_TRAINABLE_SENTINEL
        return model.sim(restricted_variables)

    def _validate_variables(self, variables: Dict) -> List[str]:
        result = list()
        for validator in self.validators:
            if not validator.evaluate(variables):
                result.append(validator)
        return result

    def _setup_substate(self, context: Dict) -> Dict:
        substate = copy.deepcopy(self.substate)
        substate["step"] = 0
        for k, v in context.items():
            if k not in self.context_names:
                raise OMError(f"Context {k} has not been declared")
            substate[k] = v
        if failed_validators := self._validate_variables(substate):
            raise OMError(
                f"Validators {failed_validators} failed for variables {_print_dict(substate)}"
            )
        return substate

    def _update_builtins(self, substate: Dict, old_vars: Dict) -> Dict:
        if "step" not in old_vars:
            old_vars["step"] = -1
        substate["step"] = old_vars["step"] + 1
        for k, v in old_vars.items():
            if k.find("delta") >= 0:
                try:
                    substate[k] = substate[k[:-6]] - old_vars[k[:-6]]
                except:
                    # This will fail on strings and stuff
                    pass

    def sim(self, **kwargs) -> Dict:
        """
        Simulates the behavior of this Graph from start to finish.

        :param kwargs: If 'untrained_mode' is True, then untrained models will be skipped over
        :return: A dictionary of the final variables.
        """

        # Set up the data collector
        journal = Journal()

        # Set up the variables
        substate = self._setup_substate(kwargs.get("context", {}))

        # Start the simulation
        state_name = self.starting_state
        while not self.end_condition.evaluate(substate, allow_errors=False):
            if substate["step"] > MAX_GAME_LENGTH:
                raise OMError(
                    f"Game has exceeded maximum length {MAX_GAME_LENGTH}.  Perhaps you have an infinite loop?"
                )

            # Run the current model
            if (
                action_name := self._sim_model_by_name(state_name, substate, **kwargs)
            ) == NOT_TRAINABLE_SENTINEL:
                action_name = random.choice(self.states[state_name].reachable_actions)

            if action_name not in self.states[state_name].reachable_actions:
                raise OMError(
                    f"Model for State {state_name} returns Action {action_name} that is not reachable"
                )

            # Record data before we change state or variables
            journal._add(action_name, state_name, substate)

            # Change state
            state_name = self.actions[action_name].next_state

            # Change variables
            old_vars = copy.deepcopy(substate)
            for update_name in self.actions[action_name].updates:
                if (
                    new_variables := self._sim_model_by_name(
                        update_name, substate, **kwargs
                    )
                ) == NOT_TRAINABLE_SENTINEL:
                    continue
                if not isinstance(new_variables, list):
                    new_variables = [new_variables]
                if len(new_variables) != len(self.updates[update_name].targets):
                    raise OMError(
                        f"Model for State {state_name} returns {len(new_variables)} variables, but {len(self.updates[update_name].targets)} variables were specified"
                    )
                for target, source in zip(
                    self.updates[update_name].targets, new_variables
                ):
                    substate[target] = source

            self._update_builtins(substate, old_vars)
            if failed_validators := self._validate_variables(substate):
                raise OMError(
                    f"Validators {failed_validators} failed for variables {_print_dict(substate)}"
                )

        if kwargs.get("debug") == "screen":
            for data in journal.df.to_dict(orient="records"):
                print(data)

        if isinstance(kwargs.get("debug"), Journal):
            journal._build_df()
            kwargs.get("debug").df = journal.df
            kwargs.get("debug").raw = journal.raw
            kwargs.get("debug").csv = journal.df.to_csv()

        return substate

    def train(self, trainer: BulkTrainer, **kwargs) -> None:
        journal = kwargs.get("debug", Journal())
        if not isinstance(journal, Journal):
            raise OMError("Debug mode must be a Journal")

        training_input = defaultdict(list)
        training_output = defaultdict(list)
        for game_num, game in enumerate(trainer.get_game()):
            ti, to = defaultdict(list), defaultdict(list)

            old_vars = None
            allowed_to_change = None
            invalid_rows = dict()
            for i, row in game.journal.df.iterrows():
                # Construct substate
                substate = copy.deepcopy(row)
                # Overwrite built-ins
                substate.update(self._update_builtins(substate, old_vars))
                old_vars = copy.deepcopy(substate)

                # state models training data
                state_name = row["State"]
                action_name = row["Action"]
                needed_variables = self.states[state_name].metadata.input
                restricted_vars = {
                    k: v for k, v in substate.items() if k in needed_variables
                }
                ti[state].append((i, restricted_vars))
                to[state].append((i, action_name))

                # update models training data
                for update in self.actions[action_name].updates:
                    needed_variables = update.metadata.input
                    restricted_vars = {
                        k: v for k, v in substate.items() if k in needed_variables
                    }
                    ti[update].append((i, restricted_vars))
                    to[update].append((i, [substate[t] for t in update.targets]))

                # Mark rows that are invalid
                if state not in self.states:
                    invalid_rows[i] = "INVALID_STATE"
                if self._validate_variables(substate):
                    invalid_rows[i] = "INVALID_VARIABLES"
                    invalid_rows[i - 1] = "INVALID_VARIABLES_NEXT_ROW"
                if action_name not in self.states[state].reachable_actions:
                    invalid_rows[i] = "UNREACHABLE_ACTION"
                    invalid_rows[i + 1] = "UNREACHABLE_ACTION_PREV_ROW"
                if allowed_to_change:
                    for k, v in substate.items():
                        if v != old_vars[k] and k not in allowed_to_change:
                            invalid_rows[i] = "NOT_ALLOWED_TO_CHANGE_VARIABLES"
                            invalid_rows[
                                i - 1
                            ] = "NOT_ALLOWED_TO_CHANGE_VARIABLES_NEXT_ROW"

                # Track which variables are allowed_to_change from here for next go-around
                allowed_to_change = list()
                for update in self.actions[action_name].updates:
                    allowed_to_change += update.targets

            # Copy valid rows
            any_error = len(invalid_rows) > 0
            on_error = kwargs.get("on_error", "skip_row")
            for k, v in ti.items():
                for i, vi in v:
                    is_error = i in invalid_rows
                    should_skip = (any_error and on_error == "skip_game") or (
                        is_error and on_error == "skip_row"
                    )
                    if i in invalid_rows:
                        journal._record_error(game_num, i, row, invalid_rows[i])
                        if should_skip:
                            continue
                    training_input[k].append(vi)
                    training_output[k].append(to[k][i])

        # Now it's time to train
        for state, model_name in self.states.items():
            materialized_model = self.materialized_models_by_name[model_name]
            materialized_model.train(training_input[state], training_output[state])
        for update, target in self.targets_by_update.items():
            materialized_model = self.materialized_models_by_name[update]
            materialized_model.train(training_input[target], training_output[target])


def model_factory(model_registry, metadata: ModelMetadata):
    model = model_registry[metadata.name]
    return model(
        name=metadata.name,
        input=metadata.input,
        **metadata.model_args,
    )


class GraphBuilder(object):
    def __init__(self, name):
        self.graph = Graph()
        self.graph.name = name

        # Mode variables
        self.mode = "Initial"
        self.mode_detail = None
        self.mode_body_turnstile = False

        # Headers
        self.graph.starting_state = None
        self.graph.end_condition = None

        # Models.  Every state and update will have exactly one model.
        self.graph.model_registry = dict()

        # Keyed by name, must be universally unique
        self.graph.states = dict()
        self.graph.actions = dict()
        self.graph.updates = dict()

        # We store built-ins, variables, pseudovariables, and contexts all in substate
        self.graph.substate = dict()  # Gets deep copied to `substate` at start of sim
        self.graph.variable_names = list()
        self.graph.pseudovariables_names = list()
        self.graph.context_names = list()

        # These all act on substate
        self.graph.validators = list()

        # Materialize the models, creating instances instead of metadata
        self.graph.materialized_models = dict()

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
            "global_validator",
            "RegisterModel",
            "Variable",
            "Context",
        }
        if probe in header_only and self.mode_body_turnstile:
            raise OMError(f"Cannot run function {probe} in body mode.")
        if "State" == probe:
            self.mode_body_turnstile = True

        if probe in ("RegisterModel", "Variable", "State", "Context"):
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

    def set_starting_state(self, starting_state: str) -> "GraphBuilder":
        self._mode("set_starting_state")
        self.graph.starting_state = starting_state
        return self

    def set_end_condition(self, end_condition: str) -> "GraphBuilder":
        self._mode("set_end_condition")
        self.graph.end_condition = Expr(end_condition)
        return self

    def global_validator(self, validators: Union[str, List[str]]) -> "GraphBuilder":
        self._mode("global_validator")
        if isinstance(validators, str):
            validators = [validators]
        self.graph.validators += [Expr(v) for v in validators]
        return self

    def RegisterModel(self, model_name: str, model: Model, **kwargs) -> "GraphBuilder":
        self._mode("RegisterModel", model_name)
        self.graph.model_registry[model_name] = model
        return self

    def _add_validator(self, validator: str, only_contains: str) -> None:
        validator = Expr(validator)
        for variable in validator.extract_variables():
            if (
                str(variable) != only_contains
                and str(variable) != f"{only_contains}_delta"
            ):
                raise OMError(
                    f"Validator {validator} uses variable {variable} which is not {only_contains}.  Use global_validator instead."
                )
        self.graph.validators.append(validator)

    def Context(
        self, context_name: str, default: Any = None, **kwargs
    ) -> "GraphBuilder":
        self._mode("Context", context_name)
        if default is None:
            raise OMError(f"Context {context_name} must have a default value")
        if context_name in self.graph.substate:
            raise OMError(f"{context_name} is redefined")
        self.graph.substate[context_name] = default
        self.graph.context_names.append(context_name)

        delta_var = f"{context_name}_delta"
        self.graph.substate[delta_var] = 0
        self.graph.context_names.append(delta_var)

        if validators := kwargs.get("validator"):
            if not isinstance(validators, list):
                validators = [validators]
            for v in validators:
                self._add_validator(v, context_name)

        return self

    def Variable(self, variable_name: str, **kwargs) -> "GraphBuilder":
        self._mode("Variable", variable_name)
        if variable_name.find("delta") != -1:
            raise OMError(f"Variable {variable_name} cannot contain delta")
        if variable_name in self.graph.substate:
            raise OMError(f"{variable_name} is redefined")
        self.graph.substate[variable_name] = kwargs.get("initially")

        delta_var = f"{variable_name}_delta"
        self.graph.substate[delta_var] = 0
        self.graph.variable_names.append(delta_var)

        if validators := kwargs.get("validator"):
            if not isinstance(validators, list):
                validators = [validators]
            for v in validators:
                self._add_validator(v, variable_name)

        return self

    def _valiadate_metadata(
        self,
        model_metadata: ModelMetadata,
    ) -> ModelMetadata:
        if not isinstance(model_metadata, ModelMetadata):
            raise OMError(f"Model {model_metadata} must be a Model")

        # Check against registry
        if model_metadata._om_model_id:
            if model_metadata._om_model_id not in self.graph.model_registry:
                self.graph.model_registry[
                    model_metadata._om_model_id
                ] = model_metadata._om_class
        if model_metadata.name not in self.graph.model_registry:
            raise OMError(f"UDM {model_metadata.name} is not registered")

        return model_metadata

    def State(self, state_name: str, **kwargs) -> "GraphBuilder":
        self._mode("State", state_name)
        if "model" not in kwargs:
            raise OMError(f"State {state_name} doesn't specify a model")

        self.graph.states[state_name] = State(
            name=state_name, metadata=self._valiadate_metadata(kwargs["model"])
        )

        return self

    def Action(self, action_name: Action, **kwargs) -> "GraphBuilder":
        self._mode("Action", action_name)
        if "next_state" not in kwargs:
            raise OMError(f"Action {action_name} doesn't specify a next_state")

        on_state = self.mode_detail[0]
        self.graph.states[on_state].reachable_actions.append(action_name)

        self.graph.actions[action_name] = Action(
            name=action_name, next_state=kwargs["next_state"]
        )

        return self

    def _get_update_name(self, var_names: List[str]) -> str:
        # This has to be universally unique
        return (
            "::".join(self.mode_detail)
            + ":"
            + ",".join(var_names)
            + "?"
            + str(random.randint(0, 1000000))
        )

    def update(
        self, variable_names: Union[str, List[str]], model: ModelMetadata, **kwargs
    ) -> "GraphBuilder":
        self._mode("update", variable_names)
        if isinstance(variable_names, str):
            variable_names = [variable_names]

        for variable_name in variable_names:
            if variable_name not in self.graph.substate:
                raise OMError(f"Variable {variable_name} is not declared")

        update_name = self._get_update_name(variable_names)
        self.graph.actions[self.mode_detail[1]].updates.append(update_name)

        self.graph.updates[update_name] = Update(
            name=update_name,
            metadata=self._valiadate_metadata(model),
            targets=variable_names,
        )

        return self

    def Build(self, **kwargs) -> Graph:
        if not self.graph.starting_state:
            raise OMError("No starting state specified")
        if not self.graph.end_condition:
            raise OMError("No end condition specified")
        if self.graph.starting_state not in self.graph.states:
            raise OMError(
                f"Starting state {self.graph.starting_state} is not in states: {list(self.graph.states.keys())}"
            )

        # Make sure that our variables are not redefined
        self.built_in = {"step"}
        if i1 := self.built_in & self.graph.substate.keys():
            raise OMError(
                f"Variables {i1} are built-in special variables and cannot be redefined"
            )

        # Make sure all of our validators make sense
        known_variables = self.built_in | self.graph.substate.keys()
        for v in self.graph.validators:
            for variable in v.extract_variables():
                if str(variable) not in known_variables:
                    raise OMError(
                        f"Validator {v} uses variable {variable} which is not defined"
                    )

        # Check state transitions
        for action_name, action in self.graph.actions.items():
            if action.next_state not in self.graph.states:
                raise OMError(
                    f"Next state {action.next_state} specified by Action {action_name} is not in states: {list(self.graph.states.keys())}"
                )

        # We want to make sure that the models all return the correct values
        n_sims = kwargs.get("n_sims", 100)
        for _ in range(n_sims):
            for name, state_or_update in dict(
                self.graph.states, **self.graph.updates
            ).items():
                self.graph.materialized_models[name] = model_factory(
                    self.graph.model_registry, state_or_update.metadata
                )
            self.graph.sim(untrained_mode=True)

        for name, state_or_update in dict(
            self.graph.states, **self.graph.updates
        ).items():
            self.graph.materialized_models[name] = model_factory(
                self.graph.model_registry, state_or_update.metadata
            )
        return self.graph
