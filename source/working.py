import random

import graph as om


class SplitA(om.Model):
    def sim(self, input):
        if random.random() < 0.9 / (input["step"] + 1):
            return "to_b"
        return "to_c"


random.seed(4)

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
    .Build()
)

# print(graph.model_args_by_state)
# print(graph.reachable_actions_from_state)

print("HELLO WORLD")
print(graph.name)
journal = om.Journal()
print(graph.sim(debug=journal))
print(journal.df)
# graph.draw()
