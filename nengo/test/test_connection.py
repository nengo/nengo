from nengo.nef.model import Model

m = Model()
input = m.make_node("input", 0.5)
pop = m.make_ensemble(
m.connect(input, pop)

