from collections import defaultdict
import random

import graph as om


class OldSplitA(om.Model):
    def sim(self, input):
        if random.random() < input["decay_factor"] ** input["num_b_visits"]:
            return "to_b"
        return "to_c"


class IncNumBVisits(om.Model):
    def sim(self, input):
        return input["num_b_visits"] + 1


graph = (
    om.GraphBuilder("Out and back")
    .set_starting_state("A")
    .set_end_condition("step >= total_steps")
    .RegisterModel("SplitA", OldSplitA)
    .RegisterModel("IncNumBVisits", IncNumBVisits)
    .Context("total_steps", default=5)
    .Context(
        "decay_factor",
        default=0.5,
        validator="(0.0 < decay_factor)&(decay_factor < 1.0)",
    )
    .Variable("num_b_visits", initially=0)
    .State("A", model=om.UDM("SplitA", input=["num_b_visits", "decay_factor"]))
    .Action("to_b", next_state="B")
    .Action("to_c", next_state="C")
    .State("B", model=om.ConstModel("to_a_from_b"))
    .Action("to_a_from_b", next_state="A")
    .update("num_b_visits", om.UDM("IncNumBVisits", input=["num_b_visits"]))
    .State("C", model=om.ConstModel("to_a_from_c"))
    .Action("to_a_from_c", next_state="A")
    .Build()
)


class SplitA(om.Model):
    def __init__(self, **kwargs):
        self.is_fit = False
        self.decay_factor = None
        super().__init__(**kwargs)

    def sim(self, input):
        if not self.is_fit:
            raise om.UntrainedException()
        if random.random() < self.decay_factor ** input["num_b_visits"]:
            return "to_b"
        return "to_c"

    def train(self, input, output):
        count_by_num_b_visits = defaultdict(int)
        goto_b_by_num_b_visits = defaultdict(int)
        for i, o in zip(input, output):
            count_by_num_b_visits[i["num_b_visits"]] += 1
            if "to_b" == o:
                goto_b_by_num_b_visits[i["num_b_visits"]] += 1

        self.decay_factor = 0
        for k, v in goto_b_by_num_b_visits.items():
            if k == 0:
                continue
            tot = count_by_num_b_visits[k]
            weight = tot / len(output)
            self.decay_factor += weight * (v / tot) ** (1 / k)
        self.is_fit = True


trainable_graph = (
    om.GraphBuilder("Out and back")
    .set_starting_state("A")
    .set_end_condition("step >= total_steps")
    .RegisterModel("SplitA", SplitA)
    .RegisterModel("IncNumBVisits", IncNumBVisits)
    .Context("total_steps", default=5)
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


for decay_factor in (0.2, 0.4, 0.6, 0.8):

    class MyGraphGenerator(om.PandasBulkTrainer):
        def get_game(self):
            global graph
            for _ in range(100):
                journal = om.Journal()
                graph.sim(
                    debug=journal,
                    context={"total_steps": 100, "decay_factor": decay_factor},
                )
                yield om.PandasDatum(
                    journal.df,
                    context={"total_steps": 100, "decay_factor": decay_factor},
                    game_id=_,
                )

    trainable_graph.train(MyGraphGenerator())
    print("HELLO")
    print(decay_factor)
    print(trainable_graph.materialized_models["A"].decay_factor)

# print(graph.model_args_by_state)
# print(graph.reachable_actions_from_state)

# random.seed(4)
# print("HELLO WORLD")
# print(graph.name)
# journal = om.Journal()
# print(graph.sim(debug=journal, context={"total_steps": 10}))
# print(journal.df)
# graph.draw()
