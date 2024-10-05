State = str
Action = str


class OMError(AssertionError):
    pass


class Model(object):
    def __init__(self):
        self._om_model_id = None


class ConstModel(Model):
    def __init__(self, action: Action):
        self.action = action
        super().__init__()
        self._om_model_id = "ConstModel"


class Graph(object):
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.starting_state = kwargs["starting_state"]
        self.states = kwargs["states"]


class GraphBuilder(object):
    def __init__(self, name):
        self.name = name
        self.mode = "Initial"
        self.mode_detail = None

        self.starting_state = None
        self.states = list()
        self.model_registry = dict()
        self.models_by_state = dict()

    def _mode(self, probe: str, probe_detail: str = "") -> None:
        needed_mode = {
            "set_starting_state": "Initial",
        }
        if probe in needed_mode and needed_mode[probe] != self.mode:
            raise OMError(f"Can't run function {probe} in {self.mode} mode.")

        capital_letters = [
            "State",
        ]
        if probe in capital_letters:
            self.mode = probe
            self.mode_detail = probe_detail
    
    def _set_state_model(self, state: State, model: Model) -> None:
        if not isinstance(model, str) and not isinstance(model, Model):
            raise OMError(f"Model for State {state} must be a string or a Model")
        
        if isinstance(model, Model): 
            if model not in self.model_registry:
                if model._om_model_id:
                    self.model_registry[model._om_model_id] = model
                    model = model._om_model_id
                else:
                    raise OMError(f"UDM {model.name} must be registered and referred to by name")
        
        if model not in self.model_registry:
            raise OMError(f"UDM {model} is not registered")

        self.models_by_state[state] = model

    def set_starting_state(self, starting_state: State) -> "GraphBuilder":
        self._mode("set_starting_state")
        self.starting_state = starting_state
        return self

    def State(self, name: State, **kwargs) -> "GraphBuilder":
        self._mode("State")
        self.states.append(name)

        if "model" not in kwargs:
            raise OMError(f"State {name} doesn't specify a model")
        self._set_state_model(name, kwargs["model"])

        return self

    def Build(self) -> Graph:
        if self.starting_state not in self.states:
            raise OMError(f"Starting state {self.starting_state} is not in states: {self.states}")

        return Graph(
            name=self.name,
            starting_state=self.starting_state,
            states=self.states,
        )
