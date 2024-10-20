import io
import random
from typing import Any, Dict, List, Optional
import unittest

import pandas as pd

import graph as om


class MockGenerator(om.PandasBulkTrainer):
    def __init__(self, csv: List[str], **kwargs):
        self.context = kwargs.get("context", {})
        self.single_game = pd.read_csv(io.StringIO("\n".join(csv)))
        self._second_game = None

    def second_game(self, csv: List[str]) -> None:
        self._second_game = pd.read_csv(io.StringIO("\n".join(csv)))

    def get_game(self):
        yield om.PandasDatum(
            self.single_game,
            context=self.context,
        )
        if self._second_game is not None:
            yield om.PandasDatum(
                self._second_game,
                context=self.context,
            )


class MockTrainable(om.Model):
    def __init__(
        self,
        name: str,
        received_input: List,
        received_output: List,
        input: Optional[List] = None,
        **kwargs,
    ):
        self.received_input = received_input
        self.received_output = received_output
        super().__init__(name="MockModel", input=input, **kwargs)

    def sim(self, input: Dict) -> Any:
        raise om.UntrainedException("MockTrainable is not meant for running sim.")

    def train(self, inputs: List, outputs: List) -> None:
        self.received_input.extend(inputs)
        self.received_output.extend(outputs)


class TestGraphBuilder(unittest.TestCase):
    def test_happy_path_deterministic_no_variables(self):
        graph = (
            om.GraphBuilder("Rotate")
            .set_starting_state("A")
            .set_end_condition("step >= 5")
            .State("A", model=om.ConstModel("to_b"))
            .Action("to_b", next_state="B")
            .State("B", model=om.ConstModel("to_c"))
            .Action("to_c", next_state="C")
            .State("C", model=om.ConstModel("to_a"))
            .Action("to_a", next_state="A")
            .Build(n_sims=1)
        )

        journal = om.Journal()
        final_state = graph.sim(debug=journal)
        self.assertDictEqual(final_state, {"step": 5})
        self.assertListEqual(journal.raw["State"], ["A", "B", "C", "A", "B"])
        self.assertListEqual(
            journal.raw["Action"], ["to_b", "to_c", "to_a", "to_b", "to_c"]
        )
        self.assertListEqual(journal.raw["step"], [0, 1, 2, 3, 4])
        self.assertEqual(len(journal.raw), 3)

    def _journal_compare(self, actual: om.Journal, expected: List[str]) -> None:
        self.assertEqual(actual.csv, "\n".join(expected + [""]))

    def test_happy_path_stochastic_no_variables_with_udm(self):
        class SplitA(om.Model):
            def sim(self, input):
                if random.random() < 0.5:
                    return "to_b"
                return "to_c"

        graph = (
            om.GraphBuilder("Rotate")
            .set_starting_state("A")
            .set_end_condition("step >= 10")
            .RegisterModel("SplitA", SplitA)
            .State("A", model=om.UDM("SplitA", input=[]))
            .Action("to_b", next_state="B")
            .Action("to_c", next_state="C")
            .State("B", model=om.ConstModel("to_a_from_b"))
            .Action("to_a_from_b", next_state="A")
            .State("C", model=om.ConstModel("to_a_from_c"))
            .Action("to_a_from_c", next_state="A")
            .Build(n_sims=100)
        )

        random.seed(0)
        journal = om.Journal()
        _ = graph.sim(debug=journal)
        self._journal_compare(
            journal,
            [
                ",State,Action,step",
                "0,A,to_c,0",
                "1,C,to_a_from_c,1",
                "2,A,to_c,2",
                "3,C,to_a_from_c,3",
                "4,A,to_b,4",
                "5,B,to_a_from_b,5",
                "6,A,to_b,6",
                "7,B,to_a_from_b,7",
                "8,A,to_c,8",
                "9,C,to_a_from_c,9",
            ],
        )

    def test_happy_path_stochastic_with_variable_udm(self):
        class SplitA(om.Model):
            def sim(self, input):
                if random.random() < 0.9 / (input["step"] + 1):
                    return "to_b"
                return "to_c"

        graph = (
            om.GraphBuilder("Rotate")
            .set_starting_state("A")
            .set_end_condition("step >= 10")
            .RegisterModel("SplitA", SplitA)
            .State("A", model=om.UDM("SplitA", input=["step"]))
            .Action("to_b", next_state="B")
            .Action("to_c", next_state="C")
            .State("B", model=om.ConstModel("to_a_from_b"))
            .Action("to_a_from_b", next_state="A")
            .State("C", model=om.ConstModel("to_a_from_c"))
            .Action("to_a_from_c", next_state="A")
            .Build(n_sims=100)
        )

        random.seed(4)
        journal = om.Journal()
        _ = graph.sim(debug=journal)
        self._journal_compare(
            journal,
            [
                ",State,Action,step",
                "0,A,to_b,0",
                "1,B,to_a_from_b,1",
                "2,A,to_b,2",
                "3,B,to_a_from_b,3",
                "4,A,to_c,4",
                "5,C,to_a_from_c,5",
                "6,A,to_c,6",
                "7,C,to_a_from_c,7",
                "8,A,to_b,8",
                "9,B,to_a_from_b,9",
            ],
        )

    def test_happy_path_stochastic_with_custom_variable(self):
        self.maxDiff = None

        class SplitA(om.Model):
            def sim(self, input):
                if random.random() < (0.5) ** input["num_b_visits"]:
                    return "to_b"
                return "to_c"

        class IncNumBVisits(om.Model):
            def sim(self, input):
                return input["num_b_visits"] + 1

        graph = (
            om.GraphBuilder("Rotate")
            .set_starting_state("A")
            .set_end_condition("step >= 10")
            .RegisterModel("SplitA", SplitA)
            .RegisterModel("IncNumBVisits", IncNumBVisits)
            .Variable("num_b_visits", initially=0)
            .State("A", model=om.UDM("SplitA", input=["num_b_visits"]))
            .Action("to_b", next_state="B")
            .Action("to_c", next_state="C")
            .State("B", model=om.ConstModel("to_a_from_b"))
            .Action("to_a_from_b", next_state="A")
            .update("num_b_visits", om.UDM("IncNumBVisits", input=["num_b_visits"]))
            .State("C", model=om.ConstModel("to_a_from_c"))
            .Action("to_a_from_c", next_state="A")
            .Build()
        )

        random.seed(4)
        journal = om.Journal()
        final_vars = graph.sim(debug=journal)
        self._journal_compare(
            journal,
            [
                ",State,Action,step,num_b_visits,num_b_visits_delta",
                "0,A,to_b,0,0,0",
                "1,B,to_a_from_b,1,0,0",
                "2,A,to_b,2,1,1",
                "3,B,to_a_from_b,3,1,0",
                "4,A,to_c,4,2,1",
                "5,C,to_a_from_c,5,2,0",
                "6,A,to_b,6,2,0",
                "7,B,to_a_from_b,7,2,0",
                "8,A,to_b,8,3,1",
                "9,B,to_a_from_b,9,3,0",
            ],
        )
        self.assertDictEqual(
            final_vars, {"num_b_visits": 4, "num_b_visits_delta": 1, "step": 10}
        )

    def _happy_trainable(
        self, a_input, a_output, b_input, b_output, c_input, c_output, **kwargs
    ):
        trainable_graph = (
            om.GraphBuilder("Three Node")
            .set_starting_state("A")
            .set_end_condition(f"step >= {kwargs.get('num_steps', 10)}")
            .RegisterModel("MockModel", MockTrainable)
            .State(
                "A",
                model=om.UDM(
                    "MockModel",
                    input=["step"],
                    model_args={"received_input": a_input, "received_output": a_output},
                ),
            )
            .Action("to_b_from_a", next_state="B")
            .Action("to_c_from_a", next_state="C")
            .State(
                "B",
                model=om.UDM(
                    "MockModel",
                    input=["step"],
                    model_args={"received_input": b_input, "received_output": b_output},
                ),
            )
            .Action("to_a_from_b", next_state="A")
            .Action("to_c_from_b", next_state="C")
            .State(
                "C",
                model=om.UDM(
                    "MockModel",
                    input=["step"],
                    model_args={"received_input": c_input, "received_output": c_output},
                ),
            )
            .Action("to_a_from_c", next_state="A")
            .Action("to_b_from_c", next_state="B")
            .Build()
        )
        return trainable_graph

    def test_happy_path_training(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output
        )

        trainable_graph.train(
            MockGenerator(
                [
                    "State,Action,step",
                    "A,to_b_from_a,0",
                    "B,to_c_from_b,1",
                    "C,to_b_from_c,2",
                    "B,to_c_from_b,3",
                    "C,to_a_from_c,4",
                    "A,to_b_from_a,5",
                    "B,to_c_from_b,6",
                    "C,to_b_from_c,7",
                    "B,to_a_from_b,8",
                    "A,to_c_from_a,9",
                ]
            )
        )

        self.assertListEqual(
            a_input,
            [{"step": 0}, {"step": 5}, {"step": 9}],
        )
        self.assertListEqual(
            a_output,
            ["to_b_from_a", "to_b_from_a", "to_c_from_a"],
        )
        self.assertListEqual(
            b_input,
            [{"step": 1}, {"step": 3}, {"step": 6}, {"step": 8}],
        )
        self.assertListEqual(
            b_output,
            ["to_c_from_b", "to_c_from_b", "to_c_from_b", "to_a_from_b"],
        )
        self.assertListEqual(
            c_input,
            [{"step": 2}, {"step": 4}, {"step": 7}],
        )
        self.assertListEqual(
            c_output,
            ["to_b_from_c", "to_a_from_c", "to_b_from_c"],
        )

    def test_happy_path_training_with_update(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        x_input, x_output = [], []
        trainable_graph = (
            om.GraphBuilder("Three Node")
            .set_starting_state("A")
            .set_end_condition(f"step >= 5")
            .Variable("x", initially=0)
            .RegisterModel("MockModel", MockTrainable)
            .State(
                "A",
                model=om.UDM(
                    "MockModel",
                    input=["step"],
                    model_args={"received_input": a_input, "received_output": a_output},
                ),
            )
            .Action("to_a_from_a", next_state="A")
            .update("x", model=om.UDM("MockModel", input=["x"], model_args={"received_input": x_input, "received_output": x_output}))
            .Action("to_b_from_a", next_state="B")
            .State(
                "B",
                model=om.UDM(
                    "MockModel",
                    input=["step"],
                    model_args={"received_input": b_input, "received_output": b_output},
                ),
            )
            .Action("to_a_from_b", next_state="A")
            .Action("to_b_from_b", next_state="B")
            .Build()
        )
        
        trainable_graph.train(
            MockGenerator(
                [
                    "State,Action,step,x",
                    "A,to_a_from_a,0,0",
                    "A,to_b_from_a,1,1",
                    "B,to_a_from_b,2,1",
                    "A,to_a_from_a,3,1",
                    "A,to_b_from_a,4,2",
                ]
            )
        )

        self.assertListEqual(x_input, [{"x": 0}, {"x": 1}])
        self.assertListEqual(x_output, [[1], [2]])

    def test_happy_path_training_2_games(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output, num_steps=5
        )

        generator = MockGenerator(
            [
                "State,Action,step",
                "A,to_b_from_a,0",
                "B,to_c_from_b,1",
                "C,to_b_from_c,2",
                "B,to_c_from_b,3",
                "C,to_a_from_c,4",
            ]
        )
        generator.second_game(
            [
                "State,Action,step",
                "A,to_b_from_a,5",
                "B,to_c_from_b,6",
                "C,to_b_from_c,7",
                "B,to_a_from_b,8",
                "A,to_c_from_a,9",
            ]
        )
        trainable_graph.train(generator)

        self.assertListEqual(
            a_input,
            [{"step": 0}, {"step": 0}, {"step": 4}],
        )
        self.assertListEqual(
            a_output,
            ["to_b_from_a", "to_b_from_a", "to_c_from_a"],
        )
        self.assertListEqual(
            b_input,
            [{"step": 1}, {"step": 3}, {"step": 1}, {"step": 3}],
        )
        self.assertListEqual(
            b_output,
            ["to_c_from_b", "to_c_from_b", "to_c_from_b", "to_a_from_b"],
        )
        self.assertListEqual(
            c_input,
            [{"step": 2}, {"step": 4}, {"step": 2}],
        )
        self.assertListEqual(
            c_output,
            ["to_b_from_c", "to_a_from_c", "to_b_from_c"],
        )

    def test_starting_state_is_set(self):
        with self.assertRaisesRegex(om.OMError, "No starting state specified"):
            _ = om.GraphBuilder("TEST").Build(n_sims=1)

    def test_end_condition_is_set(self):
        with self.assertRaisesRegex(om.OMError, "No end condition specified"):
            _ = om.GraphBuilder("TEST").set_starting_state("A").Build(n_sims=1)

    def test_set_starting_state_in_initial_mode(self):
        with self.assertRaisesRegex(
            om.OMError, "Can't run function set_starting_state in State mode."
        ):
            _ = (
                om.GraphBuilder("TEST")
                .State(
                    "A", model=om.ConstModel("to_a")
                )  # .Action("to_a", next_state="A")
                .set_starting_state("A")
                .Build(n_sims=1)
            )

    def test_starting_state_valid(self):
        with self.assertRaisesRegex(
            om.OMError, "Starting state non-state is not in states: \[\]"
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("non-state")
                .set_end_condition("step >= 5")
                .Build(n_sims=1)
            )

    def test_state_specifies_model(self):
        with self.assertRaisesRegex(om.OMError, "State A doesn't specify a model"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .State("A")  # .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_state_model_is_string_or_model(self):
        with self.assertRaisesRegex(om.OMError, "Model 1 must be a Model"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .State("A", model=1)
                .Build(n_sims=1)
            )

    def test_state_model_is_registered(self):
        with self.assertRaisesRegex(om.OMError, "UDM SomeUnknownUDM is not registered"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .State("A", model=om.UDM("SomeUnknownUDM", input=[]))
                .Build(n_sims=1)
            )

    def test_action_specifies_next_state(self):
        with self.assertRaisesRegex(
            om.OMError, "Action A doesn't specify a next_state"
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .State("A", model=om.ConstModel("to_a"))
                .Action("A")
                .Build(n_sims=1)
            )

    def test_action_specifies_valid_next_state(self):
        with self.assertRaisesRegex(
            om.OMError,
            "Next state non-state specified by Action bad-action is not in states: \['A'\]",
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .State("A", model=om.ConstModel("to_a"))
                .Action("bad-action", next_state="non-state")
                .Build(n_sims=1)
            )

    def test_state_model_reaches_reachable_actions(self):
        with self.assertRaisesRegex(
            om.OMError, "Model for State A returns Action to_b that is not reachable"
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .State("A", model=om.ConstModel("to_b"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_udm_must_define_input(self):
        class Regression(om.Model):
            pass

        with self.assertRaisesRegex(om.OMError, "UDM Regression must specify an input"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .RegisterModel("Regression", Regression)
                .State("A", model=om.UDM("Regression"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_register_not_cannot_run_in_body(self):
        with self.assertRaisesRegex(
            om.OMError, "Cannot run function RegisterModel in body mode."
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .RegisterModel("Regression", om.ConstModel("to_a"))
                .Build(n_sims=1)
            )

    def test_update_variable_is_declared(self):
        with self.assertRaisesRegex(om.OMError, "Variable X is not declared"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .update("X", model=om.ConstModel(1))
                .Build(n_sims=1)
            )

    def test_infinite_loop(self):
        with self.assertRaisesRegex(
            om.OMError,
            "Game has exceeded maximum length 5000.  Perhaps you have an infinite loop?",
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step < 0")
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_variables_returned_matches_length_of_target_variables(self):
        with self.assertRaisesRegex(
            om.OMError,
            "Model for State A returns 2 variables, but 1 variables were specified",
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .Variable("X", initially=0)
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .update("X", model=om.ConstModel([1, 2]))
                .Build(n_sims=1)
            )

    def test_variables_returned_matches_length_of_target_variables_2(self):
        with self.assertRaisesRegex(
            om.OMError,
            "Model for State A returns 1 variables, but 2 variables were specified",
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .Variable("X", initially=0)
                .Variable("Y", initially=0)
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .update(["X", "Y"], model=om.ConstModel(1))
                .Build(n_sims=1)
            )

    def test_variable_length_match_happy_path(self):
        _ = (
            om.GraphBuilder("TEST")
            .set_starting_state("A")
            .set_end_condition("step >= 5")
            .Variable("X", initially=0)
            .Variable("Y", initially=0)
            .State("A", model=om.ConstModel("to_a"))
            .Action("to_a", next_state="A")
            .update(["X", "Y"], model=om.ConstModel([1, 2]))
            .Build(n_sims=1)
        )

    def test_variable_validator(self):
        with self.assertRaisesRegex(
            om.OMError,
            "Validators \[X < 3\] failed for variables {X: 3, X_delta: 1, step: 3}",
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .Variable("X", initially=0, validator="X < 3")
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .update("X", model=om.IncModel("X"))
                .Build(n_sims=1)
            )

    def test_context_validator(self):
        graph = (
            om.GraphBuilder("TEST")
            .set_starting_state("A")
            .set_end_condition("step >= 5")
            .Context("X", default=0, validator="X < 3")
            .State("A", model=om.ConstModel("to_a"))
            .Action("to_a", next_state="A")
            .Build(n_sims=1)
        )
        with self.assertRaisesRegex(
            om.OMError,
            "Validators \[X < 3\] failed for variables {X: 4, X_delta: 0, step: 0}",
        ):
            graph.sim(context={"X": 4})

    def test_failed_to_evaluate_expression(self):
        with self.assertRaisesRegex(
            om.OMError,
            "Validators \[X > 3\] failed for variables {X: None, X_delta: 0, step: 0}",
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .Variable("X", initially=None, validator="X > 3")
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )
        pass

    def test_context_needs_default_value(self):
        with self.assertRaisesRegex(om.OMError, "Context X must have a default value"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .Context("X")
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_end_condition_failure(self):
        with self.assertRaisesRegex(
            om.OMError, "Expression 0 >= unknown_var fails unexpectedly on {'step': 0}."
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= unknown_var")
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_two_validators_happy_path(self):
        _ = (
            om.GraphBuilder("TEST")
            .set_starting_state("A")
            .set_end_condition("step >= 5")
            .global_validator("X <= Y")
            .Variable("X", initially=0, validator="X < 3")
            .Variable("Y", initially=0, validator=["Y < 3", "Y > -1"])
            .State("A", model=om.ConstModel("to_a"))
            .Action("to_a", next_state="A")
            .update(["X", "Y"], model=om.ConstModel([1, 2]))
            .Build(n_sims=1)
        )

    def test_invalid_validator(self):
        with self.assertRaisesRegex(om.OMError, "Invalid validator 0 < step < -3"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .global_validator("0 < step < -3")
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_context_validator_only_uses_context_variable(self):
        with self.assertRaisesRegex(
            om.OMError,
            "Validator X \+ Y < 100 uses variable Y which is not X.  Use global_validator instead.",
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .Context("X", default=0, validator="X + Y < 100")
                .Context("Y")
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_not_overwrite_builtin_variables(self):
        with self.assertRaisesRegex(
            om.OMError,
            "Variables {'step'} are built-in special variables and cannot be redefined",
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .Variable("step", initially=0)
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_not_define_variables_and_contexts(self):
        with self.assertRaisesRegex(om.OMError, "X is redefined"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .Variable("X", initially=0)
                .Context("X", default=0)
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_global_validators_use_known_variables(self):
        with self.assertRaisesRegex(
            om.OMError, "Validator X \+ Y < 100 uses variable Y which is not defined"
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .global_validator("X + Y < 100")
                .Variable("X", initially=0)
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_no_delta_variables(self):
        with self.assertRaisesRegex(
            om.OMError,
            "Variable X_delta cannot contain delta",
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .Variable("X", initially=0)
                .Variable("X_delta", initially=0)
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .Build(n_sims=1)
            )

    def test_delta_variable_validation_violation(self):
        with self.assertRaisesRegex(om.OMError, "Invalid validator X_delta = 2"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .set_end_condition("step >= 5")
                .Variable("X", initially=0, validator="X_delta = 2")
                .State("A", model=om.ConstModel("to_a"))
                .Action("to_a", next_state="A")
                .update("X", model=om.IncModel("X"))
                .Build(n_sims=1)
            )

    def test_passing_non_journal_to_train_fails(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output, num_steps=5
        )
        with self.assertRaisesRegex(om.OMError, "Debug mode must be a Journal"):
            generator = MockGenerator(
                [
                    "State,Action,step",
                    "A,to_b_from_a,0",
                    "B,to_c_from_b,1",
                    "C,to_a_from_c,2",
                    "A,to_b_from_a,3",
                    "B,to_c_from_b,4",
                ]
            )
            trainable_graph.train(generator, debug="bogus")

    def test_ignore_errors_will_pass_bad_values(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output, num_steps=10
        )
            
        generator = MockGenerator(
            [
                "State,Action,step",
                "A,to_b_from_a,0",
                "B,to_c_from_b,1",
                "C,to_d_from_c,2",
                "C,to_a_from_c,3",
                "D,to_c_from_a,4",
                "A,to_b_from_a,5",
                "B,to_a_from_b,6",
                "A,to_b_from_a,7",
                "C,to_c_from_b,8",
                "A,to_b_from_a,9",
            ]
        )

        trainable_graph.train(generator, on_error="ignore")

        self.assertListEqual(
            a_input,
            [{"step": 0}, {"step": 5}, {"step": 7}, {"step": 9}],
        )
        self.assertListEqual(
            a_output,
            ["to_b_from_a", "to_b_from_a", "to_b_from_a", "to_b_from_a"],
        )
        self.assertListEqual(
            b_input,
            [{"step": 1}, {"step": 6}],
        )
        self.assertListEqual(
            b_output,
            ["to_c_from_b", "to_a_from_b"],
        )
        self.assertListEqual(
            c_input,
            [{"step": 2}, {"step": 3}, {"step": 8}],
        )
        self.assertListEqual(
            c_output,
            ["to_d_from_c", "to_a_from_c", "to_c_from_b"],
        )

    def test_skip_errors_will_skip_rows(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output, num_steps=10
        )
            
        generator = MockGenerator(
            [
                "State,Action,step",
                "A,to_b_from_a,0",
                "B,to_c_from_b,1",
                "C,to_d_from_c,2",  # Bad action, skip this and next
                "C,to_a_from_c,3",  # Skip from above
                "D,to_c_from_a,4",  # Bad state, skip this
                "A,to_b_from_a,5",
                "B,to_a_from_b,6",
                "A,to_b_from_a,7",
                "C,to_c_from_b,8",  # Bad action, skip this and next
                "A,to_b_from_a,9",  # Skip from above
            ]
        )

        trainable_graph.train(generator, on_error="skip_row")

        self.assertListEqual(
            a_input,
            [{"step": 0}, {"step": 5}, {"step": 7}],
        )
        self.assertListEqual(
            a_output,
            ["to_b_from_a", "to_b_from_a", "to_b_from_a"],
        )
        self.assertListEqual(
            b_input,
            [{"step": 1}, {"step": 6}],
        )
        self.assertListEqual(
            b_output,
            ["to_c_from_b", "to_a_from_b"],
        )
        self.assertListEqual(
            c_input,
            [],
        )
        self.assertListEqual(
            c_output,
            [],
        )

    def test_errors_skip_games(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output, num_steps=10
        )
            
        generator = MockGenerator(
            [
                "State,Action,step",
                "A,to_b_from_a,0",
                "B,to_c_from_b,1",
                "C,to_d_from_c,2",  # Bad action, skip this and next
                "C,to_a_from_c,3",  # Skip from above
                "D,to_c_from_a,4",  # Bad state, skip this
                "A,to_b_from_a,5",
                "B,to_a_from_b,6",
                "A,to_b_from_a,7",
                "C,to_c_from_b,8",  # Bad action, skip this and next
                "A,to_b_from_a,9",  # Skip from above
            ]
        )

        trainable_graph.train(generator, on_error="skip_game")

        self.assertListEqual(a_input, [])
        self.assertListEqual(a_output, [])
        self.assertListEqual(b_input, [])
        self.assertListEqual(b_output, [])
        self.assertListEqual(c_input, [])
        self.assertListEqual(c_output, [])

    def test_errors_with_assert(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output, num_steps=10
        )
            
        generator = MockGenerator(
            [
                "State,Action,step",
                "A,to_b_from_a,0",
                "B,to_c_from_b,1",
                "C,to_d_from_c,2",  # Bad action, skip this and next
                "C,to_a_from_c,3",  # Skip from above
                "D,to_c_from_a,4",  # Bad state, skip this
                "A,to_b_from_a,5",
                "B,to_a_from_b,6",
                "A,to_b_from_a,7",
                "C,to_c_from_b,8",  # Bad action, skip this and next
                "A,to_b_from_a,9",  # Skip from above
            ]
        )

        with self.assertRaisesRegex(om.OMError, "Invalid training rows: {2: 'UNREACHABLE_ACTION', 3: 'UNREACHABLE_ACTION_PREV_ROW', 4: 'INVALID_STATE', 8: 'UNREACHABLE_ACTION', 9: 'UNREACHABLE_ACTION_PREV_ROW'}"):
            trainable_graph.train(generator, on_error="assert")

    def test_errors_record_to_journal(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output, num_steps=10
        )
            
        generator = MockGenerator(
            [
                "State,Action,step",
                "A,to_b_from_a,0",
                "B,to_c_from_b,1",
                "C,to_d_from_c,2",  # Bad action, skip this and next
                "C,to_a_from_c,3",  # Skip from above
                "D,to_c_from_a,4",  # Bad state, skip this
                "A,to_b_from_a,5",
                "B,to_a_from_b,6",
                "A,to_b_from_a,7",
                "C,to_c_from_b,8",  # Bad action, skip this and next
                "A,to_b_from_a,9",  # Skip from above
            ]
        )

        journal = om.Journal()
        trainable_graph.train(generator, debug=journal)
        self.assertListEqual([(e["error"], e["row_num"]) for e in journal.errors], [
            ('UNREACHABLE_ACTION', 2),
            ('UNREACHABLE_ACTION_PREV_ROW', 3),
            ('INVALID_STATE', 4),
            ('UNREACHABLE_ACTION', 8),
            ('UNREACHABLE_ACTION_PREV_ROW', 9),
        ])

    def test_training_data_starting_state(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output, num_steps=1
        )
            
        generator = MockGenerator(
            [
                "State,Action,step",
                "B,to_a_from_b,0",
            ]
        )

        journal = om.Journal()
        trainable_graph.train(generator, debug=journal)
        self.assertListEqual([(e["error"], e["row_num"]) for e in journal.errors], [
            ('INVALID_STARTING_STATE', -1),
        ])
    
    def test_end_condition_not_met(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output, num_steps=10
        )
            
        generator = MockGenerator(
            [
                "State,Action,step",
                "A,to_b_from_a,0",
            ]
        )

        journal = om.Journal()
        trainable_graph.train(generator, debug=journal)
        self.assertListEqual([(e["error"], e["row_num"]) for e in journal.errors], [
            ('INVALID_EXIT', 1),
        ])

    def test_end_condition_met_early(self):
        a_input, a_output = [], []
        b_input, b_output = [], []
        c_input, c_output = [], []
        trainable_graph = self._happy_trainable(
            a_input, a_output, b_input, b_output, c_input, c_output, num_steps=1
        )
            
        generator = MockGenerator(
            [
                "State,Action,step",
                "A,to_b_from_a,0",
                "B,to_a_from_b,1",
            ]
        )

        journal = om.Journal()
        trainable_graph.train(generator, debug=journal)
        self.assertListEqual([(e["error"], e["row_num"]) for e in journal.errors], [
            ('EARLY_EXIT', 1),
        ])

    def test_variable_only_updated_when_allowed(self):
        pass
        a_input, a_output = [], []
        b_input, b_output = [], []
        x_input, x_output = [], []
        trainable_graph = (
            om.GraphBuilder("Three Node")
            .set_starting_state("A")
            .set_end_condition(f"step >= 5")
            .Variable("x", initially=0)
            .RegisterModel("MockModel", MockTrainable)
            .State(
                "A",
                model=om.UDM(
                    "MockModel",
                    input=["step"],
                    model_args={"received_input": a_input, "received_output": a_output},
                ),
            )
            .Action("to_a_from_a", next_state="A")
            .update("x", model=om.UDM("MockModel", input=["x"], model_args={"received_input": x_input, "received_output": x_output}))
            .Action("to_b_from_a", next_state="B")
            .State(
                "B",
                model=om.UDM(
                    "MockModel",
                    input=["step"],
                    model_args={"received_input": b_input, "received_output": b_output},
                ),
            )
            .Action("to_a_from_b", next_state="A")
            .Action("to_b_from_b", next_state="B")
            .Build()
        )
        
        print("HELLLLLOOOOO")
        journal = om.Journal()
        trainable_graph.train(
            MockGenerator(
                [
                    "State,Action,step,x",
                    "A,to_a_from_a,0,0",
                    "A,to_b_from_a,1,1",
                    "B,to_a_from_b,2,2",
                    "A,to_a_from_a,3,2",
                    "A,to_b_from_a,4,3",
                ]
            ),
            debug=journal,
        )
        print("============")
        self.assertListEqual([(e["error"], e["row_num"]) for e in journal.errors], [
            ('EARLY_EXIT', 1),
        ])

if __name__ == "__main__":
    unittest.main()
