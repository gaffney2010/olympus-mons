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
            .Build()
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
                .Build()
            )

    def test_starting_state_valid(self):
        with self.assertRaisesRegex(
            om.OMError, "Starting state non-state is not in states: \[\]"
        ):
            _ = om.GraphBuilder("TEST").set_starting_state("non-state").Build()

    def test_state_specifies_model(self):
        with self.assertRaisesRegex(om.OMError, "State A doesn't specify a model"):
            _ = (
                om.GraphBuilder("TEST")
                .State("A")  # .Action("to_a", next_state="A")
                .set_starting_state("A")
                .Build()
            )

    def test_state_model_is_string_or_model(self):
        with self.assertRaisesRegex(om.OMError, "Model for State A must be a Model"):
            _ = (
                om.GraphBuilder("TEST")
                .State("A", model=1)
                .set_starting_state("A")
                .Build()
            )

    def test_state_model_is_registered(self):
        with self.assertRaisesRegex(om.OMError, "UDM SomeUnknownUDM is not registered"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .State("A", model=om.UDM("SomeUnknownUDM", input=[]))
                .Build()
            )

    def test_action_specifies_next_state(self):
        with self.assertRaisesRegex(
            om.OMError, "Action A doesn't specify a next_state"
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .State("A", model=om.ConstModel("to_a"))
                .Action("A")
                .Build()
            )

    def test_action_specifies_valid_next_state(self):
        with self.assertRaisesRegex(
            om.OMError,
            "Next state non-state specified by Action bad-action is not in states: \['A'\]",
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .State("A", model=om.ConstModel("to_a"))
                .Action("bad-action", next_state="non-state")
                .Build()
            )

    def test_state_model_reaches_reachable_actions(self):
        with self.assertRaisesRegex(
            om.OMError, "Model for State A returns Action to_b that is not reachable"
        ):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .State("A", model=om.ConstModel("to_b"))
                .Action("to_a", next_state="A")
                .Build()
            )

    def test_udm_must_define_input(self):
        class Regression(om.Model):
            pass

        with self.assertRaisesRegex(om.OMError, "UDM Regression must specify an input"):
            _ = (
                om.GraphBuilder("TEST")
                .set_starting_state("A")
                .RegisterModel("Regression", Regression)
                .State("A", model=om.UDM("Regression"))
                .Action("to_a", next_state="A")
                .Build()
            )

    def test_fail_this_test(self):
        with self.assertRaisesRegex(om.OMError, "X"):
            _ = "HELLO"


if __name__ == "__main__":
    unittest.main()
