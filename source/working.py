import random

import graph as om


class SplitA(om.Model):
    def sim(self, input):
        if random.random() < input["decay_factor"]**input["num_b_visits"]:
            return "to_b"
        return "to_c"


class IncNumBVisits(om.Model):
    def sim(self, input):
        return input["num_b_visits"] + 1


graph = (
    om.GraphBuilder("Out and back")
    .set_starting_state("A")
    .set_end_condition("step >= total_steps")
    .RegisterModel("SplitA", SplitA)
    .RegisterModel("IncNumBVisits", IncNumBVisits)
    .Context("total_steps", default=5)
    .Context("decay_factor", default=0.5, validator="0.0 < decay_factor < 1.0")
    .Variable("num_b_visits", initially=0)
    .State("A", model=om.UDM("SplitA", input=["num_b_visits", "decay_factor"]))
        .Action("to_b", next_state="B").Action("to_c", next_state="C")
    .State("B", model=om.ConstModel("to_a_from_b"))
        .Action("to_a_from_b", next_state="A").update("num_b_visits", om.UDM("IncNumBVisits", input=["num_b_visits"]))
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
print(graph.sim(debug=journal, context={"total_steps": 10}))
print(journal.df)
# graph.draw()
