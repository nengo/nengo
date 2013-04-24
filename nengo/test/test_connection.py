from nengo

m = nengo.Model("test_connection")
input = m.make_node("input", [0.5])
pop = m.make_ensemble("pop", 10, 1)
m.connect("input:output", pop)
m.build()
m.run(1)

print pop.vector_inputs
