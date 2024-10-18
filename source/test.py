import random
from typing import List
import unittest

import graph as om


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

    def test_invalid_states_are_not_used(self):
        pass

    def test_invalid_updates_are_not_used(self):
        pass

    def test_passing_non_journal_to_train_fails(self):
        pass

    def test_fail_this_test(self):
        with self.assertRaisesRegex(om.OMError, "X"):
            _ = "HELLO"


if __name__ == "__main__":
    unittest.main()
