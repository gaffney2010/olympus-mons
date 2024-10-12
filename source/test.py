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

    def test_happy_path_deterministic_no_variables_with_udm(self):
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
            .Build(n_sims=1)
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

    def test_happy_path_deterministic_with_variable_udm(self):
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
            .Build(n_sims=1)
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
        with self.assertRaisesRegex(om.OMError, "Model for State A must be a Model"):
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

    def test_fail_this_test(self):
        with self.assertRaisesRegex(om.OMError, "X"):
            _ = "HELLO"


if __name__ == "__main__":
    unittest.main()
