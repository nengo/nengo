import nengo
import nengo.spa as spa

D = 16
N = 500

model = nengo.Network(label="circular convolution")
with model:
    
    vocab=spa.Vocabulary(D)
    vocab.parse('BLUE')
    
    a = nengo.Ensemble(N, D)
    b = nengo.Ensemble(N, D)
    c = nengo.Ensemble(N, D)
    d = nengo.Ensemble(N, D)
    e = nengo.Ensemble(N, D)
    
    a.vocab=vocab
    b.vocab=vocab

    circonv = nengo.networks.CircularConvolution(100, D)

    circonv2 = nengo.networks.CircularConvolution(100, D, invert_b=True)
    
    nengo.Connection(a, circonv.A)
    nengo.Connection(b, circonv.B)
    nengo.Connection(circonv.output, d)
    nengo.Connection(d, circonv2.A)
    nengo.Connection(c, circonv2.B)
    nengo.Connection(circonv2.output, e)
    
    nengo.Probe(a)
    nengo.Probe(b)
    nengo.Probe(c)
    nengo.Probe(d)
    nengo.Probe(e)
    


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.08724877554467153
gui[model].offset = 196.4453646185188,33.585931430501546
gui[a].pos = 50.000, 2937.500
gui[a].scale = 1.000
gui[b].pos = 50.000, 3012.500
gui[b].scale = 1.000
gui[c].pos = 50.000, 3087.500
gui[c].scale = 1.000
gui[d].pos = 1350.000, 2975.000
gui[d].scale = 1.000
gui[e].pos = 1350.000, 3050.000
gui[e].scale = 1.000
gui[circonv].pos = 700.000, 1512.500
gui[circonv].scale = 1.000
gui[circonv].size = 1030.000, 2897.000
gui[circonv.A].pos = 225.000, 1475.000
gui[circonv.A].scale = 1.000
gui[circonv.B].pos = 225.000, 1550.000
gui[circonv.B].scale = 1.000
gui[circonv.output].pos = 1175.000, 1512.500
gui[circonv.output].scale = 1.000
gui[circonv.product].pos = 700.000, 1512.500
gui[circonv.product].scale = 1.000
gui[circonv.product].size = 680.000, 2785.000
gui[circonv.product.A].pos = 400.000, 1475.000
gui[circonv.product.A].scale = 1.000
gui[circonv.product.B].pos = 400.000, 1550.000
gui[circonv.product.B].scale = 1.000
gui[circonv.product.output].pos = 1000.000, 1512.500
gui[circonv.product.output].scale = 1.000
gui[circonv.product.product].pos = 700.000, 1512.500
gui[circonv.product.product].scale = 1.000
gui[circonv.product.product].size = 330.000, 2705.000
gui[circonv.product.product.ea_ensembles[0]].pos = 700.000, 200.000
gui[circonv.product.product.ea_ensembles[0]].scale = 1.000
gui[circonv.product.product.ea_ensembles[1]].pos = 700.000, 275.000
gui[circonv.product.product.ea_ensembles[1]].scale = 1.000
gui[circonv.product.product.ea_ensembles[2]].pos = 700.000, 350.000
gui[circonv.product.product.ea_ensembles[2]].scale = 1.000
gui[circonv.product.product.ea_ensembles[3]].pos = 700.000, 425.000
gui[circonv.product.product.ea_ensembles[3]].scale = 1.000
gui[circonv.product.product.ea_ensembles[4]].pos = 700.000, 500.000
gui[circonv.product.product.ea_ensembles[4]].scale = 1.000
gui[circonv.product.product.ea_ensembles[5]].pos = 700.000, 575.000
gui[circonv.product.product.ea_ensembles[5]].scale = 1.000
gui[circonv.product.product.ea_ensembles[6]].pos = 700.000, 650.000
gui[circonv.product.product.ea_ensembles[6]].scale = 1.000
gui[circonv.product.product.ea_ensembles[7]].pos = 700.000, 725.000
gui[circonv.product.product.ea_ensembles[7]].scale = 1.000
gui[circonv.product.product.ea_ensembles[8]].pos = 700.000, 800.000
gui[circonv.product.product.ea_ensembles[8]].scale = 1.000
gui[circonv.product.product.ea_ensembles[9]].pos = 700.000, 875.000
gui[circonv.product.product.ea_ensembles[9]].scale = 1.000
gui[circonv.product.product.ea_ensembles[10]].pos = 700.000, 950.000
gui[circonv.product.product.ea_ensembles[10]].scale = 1.000
gui[circonv.product.product.ea_ensembles[11]].pos = 700.000, 1025.000
gui[circonv.product.product.ea_ensembles[11]].scale = 1.000
gui[circonv.product.product.ea_ensembles[12]].pos = 700.000, 1100.000
gui[circonv.product.product.ea_ensembles[12]].scale = 1.000
gui[circonv.product.product.ea_ensembles[13]].pos = 700.000, 1175.000
gui[circonv.product.product.ea_ensembles[13]].scale = 1.000
gui[circonv.product.product.ea_ensembles[14]].pos = 700.000, 1250.000
gui[circonv.product.product.ea_ensembles[14]].scale = 1.000
gui[circonv.product.product.ea_ensembles[15]].pos = 700.000, 1325.000
gui[circonv.product.product.ea_ensembles[15]].scale = 1.000
gui[circonv.product.product.ea_ensembles[16]].pos = 700.000, 1400.000
gui[circonv.product.product.ea_ensembles[16]].scale = 1.000
gui[circonv.product.product.ea_ensembles[17]].pos = 700.000, 1475.000
gui[circonv.product.product.ea_ensembles[17]].scale = 1.000
gui[circonv.product.product.ea_ensembles[18]].pos = 700.000, 1550.000
gui[circonv.product.product.ea_ensembles[18]].scale = 1.000
gui[circonv.product.product.ea_ensembles[19]].pos = 700.000, 1625.000
gui[circonv.product.product.ea_ensembles[19]].scale = 1.000
gui[circonv.product.product.ea_ensembles[20]].pos = 700.000, 1700.000
gui[circonv.product.product.ea_ensembles[20]].scale = 1.000
gui[circonv.product.product.ea_ensembles[21]].pos = 700.000, 1775.000
gui[circonv.product.product.ea_ensembles[21]].scale = 1.000
gui[circonv.product.product.ea_ensembles[22]].pos = 700.000, 1850.000
gui[circonv.product.product.ea_ensembles[22]].scale = 1.000
gui[circonv.product.product.ea_ensembles[23]].pos = 700.000, 1925.000
gui[circonv.product.product.ea_ensembles[23]].scale = 1.000
gui[circonv.product.product.ea_ensembles[24]].pos = 700.000, 2000.000
gui[circonv.product.product.ea_ensembles[24]].scale = 1.000
gui[circonv.product.product.ea_ensembles[25]].pos = 700.000, 2075.000
gui[circonv.product.product.ea_ensembles[25]].scale = 1.000
gui[circonv.product.product.ea_ensembles[26]].pos = 700.000, 2150.000
gui[circonv.product.product.ea_ensembles[26]].scale = 1.000
gui[circonv.product.product.ea_ensembles[27]].pos = 700.000, 2225.000
gui[circonv.product.product.ea_ensembles[27]].scale = 1.000
gui[circonv.product.product.ea_ensembles[28]].pos = 700.000, 2300.000
gui[circonv.product.product.ea_ensembles[28]].scale = 1.000
gui[circonv.product.product.ea_ensembles[29]].pos = 700.000, 2375.000
gui[circonv.product.product.ea_ensembles[29]].scale = 1.000
gui[circonv.product.product.ea_ensembles[30]].pos = 700.000, 2450.000
gui[circonv.product.product.ea_ensembles[30]].scale = 1.000
gui[circonv.product.product.ea_ensembles[31]].pos = 700.000, 2525.000
gui[circonv.product.product.ea_ensembles[31]].scale = 1.000
gui[circonv.product.product.ea_ensembles[32]].pos = 700.000, 2600.000
gui[circonv.product.product.ea_ensembles[32]].scale = 1.000
gui[circonv.product.product.ea_ensembles[33]].pos = 700.000, 2675.000
gui[circonv.product.product.ea_ensembles[33]].scale = 1.000
gui[circonv.product.product.ea_ensembles[34]].pos = 700.000, 2750.000
gui[circonv.product.product.ea_ensembles[34]].scale = 1.000
gui[circonv.product.product.ea_ensembles[35]].pos = 700.000, 2825.000
gui[circonv.product.product.ea_ensembles[35]].scale = 1.000
gui[circonv.product.product.input].pos = 575.000, 1512.500
gui[circonv.product.product.input].scale = 1.000
gui[circonv.product.product.output].pos = 825.000, 1475.000
gui[circonv.product.product.output].scale = 1.000
gui[circonv.product.product.product].pos = 825.000, 1550.000
gui[circonv.product.product.product].scale = 1.000
gui[circonv2].pos = 700.000, 4512.500
gui[circonv2].scale = 1.000
gui[circonv2].size = 1030.000, 2897.000
gui[circonv2.A].pos = 225.000, 4475.000
gui[circonv2.A].scale = 1.000
gui[circonv2.B].pos = 225.000, 4550.000
gui[circonv2.B].scale = 1.000
gui[circonv2.output].pos = 1175.000, 4512.500
gui[circonv2.output].scale = 1.000
gui[circonv2.product].pos = 700.000, 4512.500
gui[circonv2.product].scale = 1.000
gui[circonv2.product].size = 680.000, 2785.000
gui[circonv2.product.A].pos = 400.000, 4475.000
gui[circonv2.product.A].scale = 1.000
gui[circonv2.product.B].pos = 400.000, 4550.000
gui[circonv2.product.B].scale = 1.000
gui[circonv2.product.output].pos = 1000.000, 4512.500
gui[circonv2.product.output].scale = 1.000
gui[circonv2.product.product].pos = 700.000, 4512.500
gui[circonv2.product.product].scale = 1.000
gui[circonv2.product.product].size = 330.000, 2705.000
gui[circonv2.product.product.ea_ensembles[0]].pos = 700.000, 3200.000
gui[circonv2.product.product.ea_ensembles[0]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[1]].pos = 700.000, 3275.000
gui[circonv2.product.product.ea_ensembles[1]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[2]].pos = 700.000, 3350.000
gui[circonv2.product.product.ea_ensembles[2]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[3]].pos = 700.000, 3425.000
gui[circonv2.product.product.ea_ensembles[3]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[4]].pos = 700.000, 3500.000
gui[circonv2.product.product.ea_ensembles[4]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[5]].pos = 700.000, 3575.000
gui[circonv2.product.product.ea_ensembles[5]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[6]].pos = 700.000, 3650.000
gui[circonv2.product.product.ea_ensembles[6]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[7]].pos = 700.000, 3725.000
gui[circonv2.product.product.ea_ensembles[7]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[8]].pos = 700.000, 3800.000
gui[circonv2.product.product.ea_ensembles[8]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[9]].pos = 700.000, 3875.000
gui[circonv2.product.product.ea_ensembles[9]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[10]].pos = 700.000, 3950.000
gui[circonv2.product.product.ea_ensembles[10]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[11]].pos = 700.000, 4025.000
gui[circonv2.product.product.ea_ensembles[11]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[12]].pos = 700.000, 4100.000
gui[circonv2.product.product.ea_ensembles[12]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[13]].pos = 700.000, 4175.000
gui[circonv2.product.product.ea_ensembles[13]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[14]].pos = 700.000, 4250.000
gui[circonv2.product.product.ea_ensembles[14]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[15]].pos = 700.000, 4325.000
gui[circonv2.product.product.ea_ensembles[15]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[16]].pos = 700.000, 4400.000
gui[circonv2.product.product.ea_ensembles[16]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[17]].pos = 700.000, 4475.000
gui[circonv2.product.product.ea_ensembles[17]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[18]].pos = 700.000, 4550.000
gui[circonv2.product.product.ea_ensembles[18]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[19]].pos = 700.000, 4625.000
gui[circonv2.product.product.ea_ensembles[19]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[20]].pos = 700.000, 4700.000
gui[circonv2.product.product.ea_ensembles[20]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[21]].pos = 700.000, 4775.000
gui[circonv2.product.product.ea_ensembles[21]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[22]].pos = 700.000, 4850.000
gui[circonv2.product.product.ea_ensembles[22]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[23]].pos = 700.000, 4925.000
gui[circonv2.product.product.ea_ensembles[23]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[24]].pos = 700.000, 5000.000
gui[circonv2.product.product.ea_ensembles[24]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[25]].pos = 700.000, 5075.000
gui[circonv2.product.product.ea_ensembles[25]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[26]].pos = 700.000, 5150.000
gui[circonv2.product.product.ea_ensembles[26]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[27]].pos = 700.000, 5225.000
gui[circonv2.product.product.ea_ensembles[27]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[28]].pos = 700.000, 5300.000
gui[circonv2.product.product.ea_ensembles[28]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[29]].pos = 700.000, 5375.000
gui[circonv2.product.product.ea_ensembles[29]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[30]].pos = 700.000, 5450.000
gui[circonv2.product.product.ea_ensembles[30]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[31]].pos = 700.000, 5525.000
gui[circonv2.product.product.ea_ensembles[31]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[32]].pos = 700.000, 5600.000
gui[circonv2.product.product.ea_ensembles[32]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[33]].pos = 700.000, 5675.000
gui[circonv2.product.product.ea_ensembles[33]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[34]].pos = 700.000, 5750.000
gui[circonv2.product.product.ea_ensembles[34]].scale = 1.000
gui[circonv2.product.product.ea_ensembles[35]].pos = 700.000, 5825.000
gui[circonv2.product.product.ea_ensembles[35]].scale = 1.000
gui[circonv2.product.product.input].pos = 575.000, 4512.500
gui[circonv2.product.product.input].scale = 1.000
gui[circonv2.product.product.output].pos = 825.000, 4475.000
gui[circonv2.product.product.output].scale = 1.000
gui[circonv2.product.product.product].pos = 825.000, 4550.000
gui[circonv2.product.product.product].scale = 1.000
