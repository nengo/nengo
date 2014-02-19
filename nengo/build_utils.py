import numpy as np

from . import objects
from . import model


def generate_dot(objs, connections):
    text = []
    text.append('digraph G {')
    for obj in objs:
        text.append('  "%d" [label="%s"];' % (id(obj), obj.label))

    def label(transform):
        transform = np.asarray(transform)
        if len(transform.shape) == 0:
            return ''
        return '%dx%d' % transform.shape
    for c in connections:
        text.append('  "%d" -> "%d" [label="%s"];' % (
            id(c.pre), id(c.post), label(c.transform)))
    text.append('}')
    return '\n'.join(text)


def remove_passthrough_nodes(objs, connections):  # noqa

    with open('pre.dot', 'w') as f:
        f.write(generate_dot(objs, connections))

    inputs = {obj: [] for obj in objs}
    outputs = {obj: [] for obj in objs}

    c_removed = set()
    c_new = set()

    obj_removed = set()

    for c in connections:
        inputs[c.post].append(c)
        outputs[c.pre].append(c)

    for obj in objs:
        if isinstance(obj, objects.Node) and obj.output is None:
            print('found passthrough node', obj, id(obj))
            obj_removed.add(obj)

            for c in inputs[obj]:
                c_removed.add(c)
                outputs[c.pre].remove(c)
            for c in outputs[obj]:
                c_removed.add(c)
                inputs[c.post].remove(c)

            for c_in in inputs[obj]:
                for c_out in outputs[obj]:
                    filter = c_in.filter
                    if filter is None:
                        filter = c_out.filter
                    else:
                        assert c_out.filter is None

                    transform = np.dot(c_out.transform, c_in.transform)
                    function = c_in.function
                    assert c_out.function is None

                    assert c_in.pre != obj
                    assert c_out.post != obj

                    if not np.all(transform == 0):
                        dummy = model.Model()
                        with dummy:
                            args = {}
                            if function is not None:
                                args['function'] = function
                            c = objects.Connection(c_in.pre, c_out.post,
                                                   filter=filter,
                                                   transform=transform, **args)
                        c_new.add(c)
                        outputs[c.pre].append(c)
                        inputs[c.post].append(c)

    for c in c_new:
        if c not in c_removed:
            connections.append(c)
    for c in c_removed:
        if c not in c_new:
            connections.remove(c)
    for obj in obj_removed:
        objs.remove(obj)

    with open('post.dot', 'w') as f:
        f.write(generate_dot(objs, connections))
    import os
    os.system('dot.bat -Tpng -opre.png pre.dot')
    os.system('dot.bat -Tpng -opost.png post.dot')

    return objs, connections
