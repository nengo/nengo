from nengo.utils.simulator import validate_ops


def test_validate_ops():
    """tests validate_ops, including may_share_memory"""

    class mySet:
        is_view = False
        base = 2

        def __index__(self):
            return 0

        def __add__(self, o):
            return [1]

        def may_share_memory(self, o):
            return False

    sets = [mySet(), mySet()]
    ups = [mySet()]
    incs = 3
    validate_ops(sets, ups, incs)
