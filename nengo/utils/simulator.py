from collections import defaultdict
import itertools

from .compat import iteritems
from .graphs import add_edges
from .stdlib import groupby


def operator_depencency_graph(operators):  # noqa: C901
    # -- all views of a base object in a particular dictionary
    by_base_sets = defaultdict(list)
    by_base_writes = defaultdict(list)
    by_base_reads = defaultdict(list)
    reads = defaultdict(list)
    sets = defaultdict(list)
    incs = defaultdict(list)
    ups = defaultdict(list)

    for op in operators:
        for sig in op.sets:
            by_base_sets[sig.base].append(sig)

        for sig in op.sets + op.incs:
            by_base_writes[sig.base].append(sig)

        for sig in op.reads:
            by_base_reads[sig.base].append(sig)

        for sig in op.reads:
            reads[sig].append(op)

        for sig in op.sets:
            sets[sig].append(op)

        for sig in op.incs:
            incs[sig].append(op)

        for sig in op.updates:
            ups[sig].append(op)

    validate_ops(sets, ups, incs)

    # -- Scheduling algorithm for serial evaluation:
    #    1) All sets on a given memory block
    #    2) All incs on a given memory block
    #    3) All reads on a given memory block
    #    4) All updates on a given memory block

    dg = dict((op, set()) for op in operators)  # ops are nodes of the graph

    # -- incs depend on sets
    for sig, post_ops in iteritems(incs):
        pre_ops = list(sets[sig])
        for sig2 in by_base_sets[sig.base]:
            if sig.may_share_memory(sig2):
                pre_ops.extend(sets[sig2])
        add_edges(dg, itertools.product(set(pre_ops), post_ops))

    # -- reads depend on writes (sets and incs)
    for sig, post_ops in iteritems(reads):
        pre_ops = sets[sig] + incs[sig]
        for sig2 in by_base_writes[sig.base]:
            if sig.may_share_memory(sig2):
                pre_ops.extend(sets[sig2] + incs[sig2])
        add_edges(dg, itertools.product(set(pre_ops), post_ops))

    # -- updates depend on reads, sets, and incs.
    for sig, post_ops in iteritems(ups):
        pre_ops = sets[sig] + incs[sig] + reads[sig]
        for sig2 in by_base_writes[sig.base] + by_base_reads[sig.base]:
            if sig.may_share_memory(sig2):
                pre_ops.extend(sets[sig2] + incs[sig2] + reads[sig2])
        add_edges(dg, itertools.product(set(pre_ops), post_ops))

    return dg


def validate_ops(sets, ups, incs):
    # -- assert that only one op sets any particular view
    for sig in sets:
        assert len(sets[sig]) == 1, (sig, sets[sig])

    # -- assert that only one op updates any particular view
    for sig in ups:
        assert len(ups[sig]) == 1, (sig, ups[sig])

    # --- assert that any sig that is incremented is also set/updated
    for sig in incs:
        assert len(sets[sig] + ups[sig]) > 0, (sig)

    # -- assert that no two views are both set and aliased
    for _, base_group in groupby(sets, lambda x: x.base, hashable=True):
        for sig, sig2 in itertools.combinations(base_group, 2):
            assert not sig.may_share_memory(sig2), (
                "%s shares memory with %s" % (sig, sig2))

    # -- assert that no two views are both updated and aliased
    for _, base_group in groupby(ups, lambda x: x.base, hashable=True):
        for sig, sig2 in itertools.combinations(base_group, 2):
            assert not sig.may_share_memory(sig2), (
                "%s shares memory with %s" % (sig, sig2))
