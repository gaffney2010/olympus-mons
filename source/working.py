import graph as om

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

# print(graph.model_args_by_state)
# print(graph.reachable_actions_from_state)

print("HELLO WORLD")
print(graph.name)
journal = om.Journal()
print(graph.sim(debug=journal))
print(journal.df)
# graph.draw()
