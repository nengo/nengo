def make(net,node,index=0,dimensions=8,pattern='I',pstc=0.01,use_single_input=False):
    STN=node.getNode('STN')

    transform=numeric.zeros((STN.dimension,dimensions),'f')

    if dimensions in hrr.Vocabulary.defaults.keys():
        vocab=hrr.Vocabulary.defaults[dimensions]
    else:
        vocab=hrr.Vocabulary(dimensions)

    terms=[t.name for t in node.terminations]
    STNterms=[t.name for t in STN.terminations]

    count=0
    while 'rule_%02d'%count in terms or 'rule_%02d'%count in STNterms:
        count=count+1

    name='rule_%02d'%count

    transform[index,:]=vocab.parse(pattern).v


    if use_single_input:
        input=node.getNode('input')
        input.addDecodedTermination(name,transform,pstc,False)
        node.exposeTermination(input.getTermination(name),name)
    else:
        StrD1=node.getNode('StrD1')
        StrD2=node.getNode('StrD2')

        STN.addDecodedTermination(name,transform,pstc,False)
        node.exposeTermination(STN.getTermination(name),name+'_STN')
        StrD1.addDecodedTermination(name,transform*(0.8),pstc,False)
        node.exposeTermination(StrD1.getTermination(name),name+'_StrD1')
        StrD2.addDecodedTermination(name,transform*(1.2),pstc,False)
        node.exposeTermination(StrD2.getTermination(name),name+'_StrD2')

    if net.network.getMetaData("bgrule") == None:
        net.network.setMetaData("bgrule", HashMap())

    bgrules = net.network.getMetaData("bgrule")

    rule=HashMap(6)
    rule.put("name", node.getName())
    rule.put("index", index)
    rule.put("dimensions", dimensions)
    rule.put("pattern", pattern)
    rule.put("pstc", pstc)
    rule.put("use_single_input", use_single_input)

    bgrules.put(node.getName(), rule)

    if net.network.getMetaData("templates") == None:
        net.network.setMetaData("templates", ArrayList())
    templates = net.network.getMetaData("templates")
    templates.add(node.getName())
