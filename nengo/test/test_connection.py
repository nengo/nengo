import nengo

m = nengo.Model("test_connection")
input = m.make_node("input", [0.5])
pop = m.make_ensemble("pop", 10, 1)
m.make_ensemble('b', 10, 1)

filter = nengo.pstc(0.01)
print filter
filter.pop('type')
filter = nengo.pstc(0.01)
print filter

m.connect("input:output", pop)
m.connect('pop','b')
#m.probe('input:output')
m.build()
m.run(1)

print pop.vector_inputs
