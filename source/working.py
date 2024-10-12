import random

import graph as om


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

# print(graph.model_args_by_state)
# print(graph.reachable_actions_from_state)

random.seed(4)
print("HELLO WORLD")
print(graph.name)
journal = om.Journal()
print(graph.sim(debug=journal))
print(journal.df)
# graph.draw()
