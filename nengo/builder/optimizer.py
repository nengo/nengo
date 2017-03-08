"""Operator graph optimizers."""

from collections import defaultdict, Mapping, namedtuple
import logging
import warnings

import numpy as np

from nengo.builder.neurons import SimNeurons
from nengo.builder import operator
from nengo.builder.operator import DotInc, ElementwiseInc, SlicedCopy
from nengo.builder.signal import Signal
from nengo.utils.compat import iteritems, itervalues, zip_longest
from nengo.utils.graphs import reverse_edges, transitive_closure
from nengo.utils.stdlib import Timer, WeakKeyDefaultDict, WeakSet

logger = logging.getLogger(__name__)

__all__ = ["optimize"]


def optimize(model, dg, max_passes=None):
    """Optimizes the operator graph by merging operators.

    This reduces the number of iterators to iterate over in slow Python code
    (as opposed to fast C code). The resulting merged operators will also
    operate on larger chunks of sequential memory, making better use of CPU
    caching and prefetching.

    The optimization algorithm has worst case complexity :math:`O(n^2 + e)`,
    where :math:`n` is the number of operators and :math:`e` is the number
    of edges in the dependency graph. In practice the run time will be much
    better because not all :math:`n^2` pairwise combinations of operators
    will be evaluated. A grouping depending on the operator type and view
    bases is done with dictionaries. This grouping can be done in amortized
    linear time and reduces the actual worst-case runtime of the optimization
    algorithm to :math:`O(gm^2 + e)`, where :math:`g` is the number of groups
    and :math:`m` is the number of elements in a group. Moreover, information
    about memory alignment will be used to cut the inner loop short in
    many cases and gives a runtime much closer to linear in most cases.

    Note that this function modifies both ``model`` and ``dg``.

    Parameters
    ----------
    model : `nengo.builder.Model`
        Builder output to optimize.
    dg : dict
        Dict of the form ``{a: {b, c}}`` where ``b`` and ``c`` depend on ``a``,
        specifying the operator dependency graph of the model.
    """

    logger.info("Optimizing model...")

    # We try first to merge operators with views only as these have a fixed
    # order for the memory alignment whereas operators without views could
    # be merged in a random order. Merging the views of operators will
    # propagate requirements in the memory ordering via the other
    # associated signals of the operator to other operators.

    # Once no more operators with views can be merged, we try to merge
    # operators without views and then try again merging views (because
    # each operator merge might generate new views).

    sig_replacements = {}
    single_pass = OpMergePass(dg, sig_replacements)

    n_initial_ops = len(dg)
    cum_duration = 0.
    before, after = None, None
    i = 0
    only_merge_ops_with_view = True
    while only_merge_ops_with_view or after < before:
        if max_passes is not None and i >= max_passes:
            break

        i += 1

        if logger.isEnabledFor(logging.DEBUG):
            for tp, ops in iteritems(groupby(dg, type)):
                logger.debug("%s: %d", tp, len(ops))

        only_merge_ops_with_view = before is None or before != after
        before = len(dg)

        with Timer() as t:
            single_pass(only_merge_ops_with_view)

        after = len(dg)
        logger.info(
            "Pass %i [%s]: Reduced %i to %i operators in %fs.",
            i, "views" if only_merge_ops_with_view else "non-views",
            before, after, t.duration)

        # Prevent optimizer from running too long if we get up diminishing
        # returns.
        # Note that we don't break if there was no reduction at all because
        # in that case we want to toggle only_merge_ops_with_view which might
        # still yield some significant reduction.
        cum_duration += t.duration
        mean_reduction_rate = (n_initial_ops - after) / cum_duration
        last_reduction_rate = (before - after) / t.duration
        threshold = 0.01
        if .0 < last_reduction_rate < threshold * mean_reduction_rate:
            logger.info(
                "Operator reduction rate fell below {} mean reduction rate. "
                "Stopping optimizer.".format(threshold))
            break

    # Update model signals
    for key in model.sig:
        for name, val in iteritems(model.sig[key]):
            model.sig[key][name] = sig_replacements.get(val, val)

    # Reinitialize the model's operator list
    del model.operators[:]
    for op in dg:
        model.add_op(op)


class BidirectionalDAG(object):
    """Directed acyclic graph supporting bidirectional traversal.

    Parameters
    ----------
    forward : dict
        Forward edges for each vertex in the form
        {1: {2, 3}, 2: {3}, 3: set()}.

    Attributes
    ----------
    forward : dict
        Maps vertices to edges in forward direction.
    backward : dict
        Maps vertices to edges in backward direction.
    """

    def __init__(self, forward):
        self.forward = forward
        self.backward = reverse_edges(forward)

    def merge(self, vertices, merged_vertex):
        """Merges vertices in the graph.

        Parameters
        ----------
        vertices : set
            Vertices that are being merged.
        merged_vertex
            The vertex that replaces *vertices*.
        """

        forward_edges = set()
        for v in vertices:
            forward_edges.update(self.forward[v])
            del self.forward[v]
        self.forward[merged_vertex] = forward_edges

        backward_edges = set()
        for v in vertices:
            backward_edges.update(self.backward[v])
            del self.backward[v]
        self.backward[merged_vertex] = backward_edges

        for e in forward_edges:
            self.backward[e].difference_update(vertices)
            self.backward[e].add(merged_vertex)

        for e in backward_edges:
            self.forward[e].difference_update(vertices)
            self.forward[e].add(merged_vertex)


class OpMergePass(object):
    def __init__(self, dg, sig_replacements):
        self.dg = BidirectionalDAG(dg)
        self.sig_replacements = sig_replacements
        self.might_merge = set(dg)

        self.sig2ops = WeakKeyDefaultDict(WeakSet)
        self.base2views = WeakKeyDefaultDict(WeakSet)
        for op in self.dg.forward:
            for s in op.all_signals:
                self.sig2ops[s].add(op)
                self.base2views[s.base].add(s)

        # These variables will be initialized and used on each pass
        self.dependents = None
        self.only_merge_ops_with_view = None

        self.merged = set()
        self.merged_dependents = set()
        self.opinfo = OpInfo()

    def __call__(self, only_merge_ops_with_view):
        """Perform a single optimization pass.

        Parameters
        ----------
        only_merge_ops_with_view : bool
            Limits operator merges to operators with views.
        """

        # --- Initialize pass state
        self.dependents = transitive_closure(self.dg.forward)
        self.only_merge_ops_with_view = only_merge_ops_with_view
        self.merged.clear()
        self.merged_dependents.clear()
        self.opinfo.clear()

        # --- Do an optimization pass
        self.perform_merges()

    def perform_merges(self):
        """Go through all operators and merge them where possible.

        Parameters
        ----------
        only_merge_ops_with_view : bool
            Limit merges to operators with views.
        """

        # We go through the ops grouped by type as only ops with the same
        # type can be merged.
        by_type = groupby(self.might_merge, type)

        # Note that we will stop once we merge any operator, so merges are
        # performed on at most one type of operator per pass.
        # The depedency graph and other information will be updated
        # before merging other operator types.

        # We go through ops in a heuristic order to reduce runtime
        firstops = [ElementwiseInc, SlicedCopy, DotInc, SimNeurons]
        sortedops = firstops + [op for op in by_type if op not in firstops]
        for optype in sortedops:

            if OpMerger.is_type_mergeable(optype):
                self.perform_merges_for_subset(by_type[optype])

            # If we're not only merging views, the memory layout changes
            # and non-views are turned into views. In that case we need
            # to update the signals the operators are referring to before
            # trying to merge a different type of operators. Thus, we break
            # the loop here.
            if not self.only_merge_ops_with_view and len(self.merged) > 0:
                break

    def perform_merges_for_subset(self, subset):
        """Performs operator merges for a subset of operators.

        Parameters
        ----------
        subset : list
            Subset of operators.
        """
        by_view = groupby(subset, lambda op: self.opinfo[op].v_base)
        if self.only_merge_ops_with_view:
            if None in by_view:
                # If an op has no views, v_base will be None.
                # If we're only merging views, then we get rid of this subset.
                del by_view[None]

            for view_subset in itervalues(by_view):
                if len(view_subset) > 1:
                    self.perform_merges_for_view_subset(view_subset)
        elif None in by_view and len(by_view[None]) > 1:
            self.perform_merges_for_view_subset(by_view[None])

    def perform_merges_for_view_subset(self, subset):
        """Perform merges for a subset of operators with the same view base.

        Parameters
        ----------
        subset : list
            Subset of operators. These need to have the same view base (can be
            None if it is None for all) for their first signal in
            `all_signals`.
        """

        # Sort to have sequential memory.
        offsets = np.array([self.opinfo[op].v_offset for op in subset])
        sort_indices = np.argsort(offsets)
        offsets = offsets[sort_indices]
        sorted_subset = [subset[i] for i in sort_indices]

        for i, op1 in enumerate(sorted_subset):
            if op1 in self.merged:
                # Cannot merge merged operator again until dependency graph
                # has been updated
                continue

            if op1 in self.merged_dependents or any(
                    op in self.merged for op in self.dependents[op1]):
                continue

            tomerge = OpsToMerge(op1, self.merged, self.merged_dependents,
                                 self.dependents)

            # For a merge to be possible the view of the next operator has to
            # start where the view of op1 ends. Because we have sorted the
            # operators by the start of their views we can do a binary search
            # and potentially skip a number of operators at the beginning.
            start = np.searchsorted(
                offsets, offsets[i] + self.opinfo[op1].v_size, side='left')

            for op2 in sorted_subset[start:]:

                if tomerge.not_sequential(op2):
                    # If this check is true the view of op2 does not
                    # immediately follow the view of the operators being
                    # merged. Because we iterate over the operators sorted by
                    # view offset there will be a gap between the end of the
                    # operators being merged and all remaining operators to
                    # loop over. With such a gap a merge is never possible and
                    # we can cut the loop short.
                    break

                if op2 in self.merged:
                    continue

                if OpMerger.is_mergeable(op2, tomerge):
                    tomerge.add(op2)

            if len(tomerge.ops) > 1:
                self.merge(tomerge)
            elif self.only_merge_ops_with_view:
                self.might_merge.remove(op1)

    def merge(self, tomerge):
        """Merges the given operators.

        This method will also update ``op_replacements``, ``sig_replacements``,
        and the internal list of merged operators to prevent further merges
        on the same operators before all required operators and signals have
        been replaced.
        """
        merged_op, merged_sig = OpMerger.merge(tomerge.ops)
        self.dg.merge(tomerge.ops, merged_op)

        # Update tracking what has been merged and might be mergeable in the
        # future
        self.might_merge.difference_update(tomerge.ops)
        self.might_merge.add(merged_op)
        self.merged.update(tomerge.ops)
        self.merged_dependents.update(tomerge.all_dependents)

        for op in tomerge.ops:
            # Mark all operators referencing the same signals as merged
            # (even though they are not) to prevent them from getting
            # merged before their signals have been updated.
            for s in op.all_signals:
                self.merged.update(self.sig2ops[s])

        # Signal related updates
        self.resolve_views_on_replaced_signals(merged_sig)
        self.sig_replacements.update(merged_sig)
        self.replace_op_signals(merged_sig)
        self.update_signal_indexing(merged_op, merged_sig)

    def resolve_views_on_replaced_signals(self, replaced_signals):
        for sig in list(replaced_signals):
            for view in self.base2views[sig]:
                if view is sig:
                    continue
                assert view.base is sig
                base_replacement = replaced_signals[sig]
                offset = view.offset
                strides = tuple(
                    a // b * c for a, b, c in zip_longest(
                        view.strides,
                        view.base.strides,
                        base_replacement.strides,
                        fillvalue=1))
                if base_replacement.is_view:
                    offset += base_replacement.offset
                    base_replacement = base_replacement.base
                buf = base_replacement.initial_value
                initial_value = np.ndarray(buffer=buf,
                                           dtype=view.dtype,
                                           shape=view.shape,
                                           offset=offset,
                                           strides=strides)
                replaced_signals[view] = Signal(initial_value,
                                                name=view.name,
                                                base=base_replacement,
                                                readonly=view.readonly)

    def replace_op_signals(self, replaced_signals):
        ops = [op for s in replaced_signals for op in self.sig2ops[s]]
        for v in ops:
            # Update the op's signals
            for key in dir(v):
                sig = getattr(v, key)
                if isinstance(sig, Signal):
                    setattr(v, key, replaced_signals.get(sig, sig))

            v.sets = [replaced_signals.get(s, s) for s in v.sets]
            v.incs = [replaced_signals.get(s, s) for s in v.incs]
            v.reads = [replaced_signals.get(s, s) for s in v.reads]
            v.updates = [replaced_signals.get(s, s) for s in v.updates]

    def update_signal_indexing(self, merged_op, replaced_signals):
        for s in merged_op.all_signals:
            self.sig2ops[s].add(merged_op)
            if s.is_view:
                self.base2views[s.base].add(s)

        for from_sig, to_sig in iteritems(replaced_signals):
            self.sig2ops[to_sig] = self.sig2ops[from_sig]
            if to_sig.is_view:
                self.base2views[to_sig.base].add(to_sig)


class OpInfo(Mapping):
    """Analyze and store extra information about operators."""

    _OpDetails = namedtuple(
        "_OpDetails", ["first_view", "v_offset", "v_size", "v_base"])

    def __init__(self):
        self.info = {}

    def __getitem__(self, op):
        if op not in self.info:
            try:
                first_view = next(s for s in op.all_signals if s.is_view)
                self.info[op] = self._OpDetails(first_view=first_view,
                                                v_offset=first_view.offset,
                                                v_size=first_view.nbytes,
                                                v_base=first_view.base)
            except StopIteration:
                self.info[op] = self._OpDetails(
                    first_view=None, v_offset=0, v_size=0, v_base=None)
        return self.info[op]

    def __iter__(self):
        return iter(self.info)

    def __len__(self):
        return len(self.info)

    def __repr__(self):
        return repr(self.info)

    def __str__(self):
        return str(self.info)

    def clear(self):
        self.info.clear()


class OpsToMerge(object):
    """Analyze and store extra information about a list of ops to be merged."""

    def __init__(self, initial_op, merged, merged_dependents, dependents):
        self.merged = merged
        self.merged_dependents = merged_dependents
        self.dependents = dependents
        self.ops = [initial_op]
        self.optype = type(initial_op)
        self.opinfo = OpInfo()

        self.all_signals = set(initial_op.all_signals)
        self.all_dependents = set(self.dependents[initial_op])

    @property
    def last_op(self):
        return self.ops[-1]

    def add(self, op):
        self.ops.append(op)
        self.all_signals.update(op.all_signals)
        self.all_dependents.update(self.dependents[op])

    @staticmethod
    def check_signals(op1, op2):
        for s1, s2 in zip(op1.all_signals, op2.all_signals):
            # If one op's signal is a view, the other must be as well
            if s1.is_view is not s2.is_view:
                return False

            if s1.is_view:
                # Views must be on the same base
                if s1.base is not s2.base:
                    return False
                # Views must have the same dtype
                elif s1.dtype is not s2.dtype:
                    return False
                # Views must have the same strides
                elif s1.strides != s2.strides:
                    return False

        return True

    def not_sequential(self, op):
        lastop = self.opinfo[self.ops[-1]]
        return (lastop.v_offset + lastop.v_size < self.opinfo[op].v_offset)


class OpMerger(object):
    mergers = {}

    @classmethod
    def is_mergeable(cls, op, tomerge):
        merger = cls.mergers[tomerge.optype]

        independent_of_ops_tomerge = (
            op not in tomerge.all_dependents and
            len(tomerge.dependents[op].intersection(tomerge.ops)) == 0)
        independent_of_prior_merges = (
            op not in tomerge.merged and
            op not in tomerge.merged_dependents and
            all(o not in tomerge.merged for o in tomerge.dependents[op]))
        return (
            type(op) is tomerge.optype and
            independent_of_ops_tomerge and
            independent_of_prior_merges and
            cls.is_type_mergeable(tomerge.optype) and
            all(tomerge.check_signals(o, op) for o in tomerge.ops) and
            merger.check_signals(op, tomerge) and
            merger.is_mergeable(tomerge.last_op, op))

    @classmethod
    def is_type_mergeable(cls, optype):
        return optype in cls.mergers

    @classmethod
    def merge(cls, ops):
        return cls.mergers[type(ops[0])].merge(ops)

    @classmethod
    def register(cls, optype):
        def register(merger):
            if optype in cls.mergers:
                warnings.warn(
                    "Merger for operator type {} overwritten.".format(optype))
            cls.mergers[optype] = merger
            return merger
        return register


class Merger(object):
    @staticmethod
    def check_signals(op, tomerge):
        return len(tomerge.all_signals.intersection(op.all_signals)) == 0

    @staticmethod
    def is_mergeable(op1, op2):
        return False

    @staticmethod
    def merge(ops):
        raise NotImplementedError("Cannot merge arbitrary ops.")

    @staticmethod
    def merge_dicts(*dicts):
        """Merges the given dictionaries into a single dictionary.

        This function assumes and enforces that no keys overlap.
        """
        d = {}
        for other_d in dicts:
            assert all(k not in d for k in other_d)
            d.update(other_d)
        return d


@OpMerger.register(operator.TimeUpdate)
class TimeUpdateMerger(Merger):

    @staticmethod
    def is_mergeable(op1, op2):
        return True

    @staticmethod
    def merge(ops):
        step, step_sigr = SigMerger.merge([o.step for o in ops])
        time, time_sigr = SigMerger.merge([o.time for o in ops])
        return (operator.TimeUpdate(step, time),
                Merger.merge_dicts(step_sigr, time_sigr))


@OpMerger.register(operator.Reset)
class ResetMerger(Merger):

    @staticmethod
    def is_mergeable(op1, op2):
        return SigMerger.check([op1.dst, op2.dst]) and op1.value == op2.value

    @staticmethod
    def merge(ops):
        dst, replacements = SigMerger.merge([o.dst for o in ops])
        return operator.Reset(dst, ops[0].value), replacements


@OpMerger.register(operator.Copy)
class CopyMerger(Merger):

    @staticmethod
    def is_mergeable(op1, op2):
        return (SigMerger.check([op1.dst, op2.dst])
                and SigMerger.check([op1.src, op2.src]))

    @staticmethod
    def merge(ops):
        dst, dst_sigr = SigMerger.merge([o.dst for o in ops])
        src, src_sigr = SigMerger.merge([o.src for o in ops])
        return operator.Copy(src, dst), Merger.merge_dicts(dst_sigr, src_sigr)


@OpMerger.register(operator.SlicedCopy)
class SlicedCopyMerger(Merger):

    @staticmethod
    def is_mergeable(op1, op2):
        return (SigMerger.check([op1.src, op2.src]) and
                SigMerger.check([op1.dst, op2.dst]) and
                op1.src_slice is Ellipsis and op1.dst_slice is Ellipsis and
                op2.src_slice is Ellipsis and op2.dst_slice is Ellipsis and
                op1.inc == op2.inc)

    @staticmethod
    def merge_slice(signals, slices):
        if all(s is Ellipsis for s in slices):
            return Ellipsis
        elif any(s is Ellipsis for s in slices):
            raise ValueError("Mixed Ellipsis with list of indices.")

        offset = 0
        merged_slice = []
        for sig, sl in zip(signals, slices):
            merged_slice.extend([i + offset for i in sl])
            offset += sig.size
        return merged_slice

    @staticmethod
    def merge(ops):
        src_sigs = [o.src for o in ops]
        dst_sigs = [o.dst for o in ops]

        src, src_sigr = SigMerger.merge(src_sigs)
        dst, dst_sigr = SigMerger.merge(dst_sigs)
        src_slice = SlicedCopyMerger.merge_slice(
            src_sigs, [o.src_slice for o in ops])
        dst_slice = SlicedCopyMerger.merge_slice(
            dst_sigs, [o.dst_slice for o in ops])
        return operator.SlicedCopy(
            src, dst, src_slice=src_slice, dst_slice=dst_slice,
            inc=ops[0].inc), Merger.merge_dicts(src_sigr, dst_sigr)


@OpMerger.register(operator.ElementwiseInc)
class ElementwiseIncMerger(Merger):

    @staticmethod
    def is_mergeable(op1, op2):
        return (SigMerger.check([op1.A, op2.A], axis=op1.A.ndim - 1) and
                SigMerger.check([op1.X, op2.X], axis=op1.X.ndim - 1) and
                SigMerger.check([op1.Y, op2.Y], axis=op1.Y.ndim - 1))

    @staticmethod
    def merge(ops):
        A, A_sigr = SigMerger.merge([o.A for o in ops], axis=ops[0].A.ndim - 1)
        X, X_sigr = SigMerger.merge([o.X for o in ops], axis=ops[0].X.ndim - 1)
        Y, Y_sigr = SigMerger.merge([o.Y for o in ops], axis=ops[0].Y.ndim - 1)
        return (operator.ElementwiseInc(A, X, Y),
                Merger.merge_dicts(A_sigr, X_sigr, Y_sigr))


@OpMerger.register(operator.DotInc)
class DotIncMerger(Merger):

    @staticmethod
    def check_signals(op, tomerge):
        none_shared = (Merger.check_signals(op, tomerge) and
                       len(set(o.X for o in tomerge.ops)) == len(tomerge.ops))
        all_x_shared = (
            not any(set((op.X, op.A, op.Y)).intersection((o.A, o.Y))
                    for o in tomerge.ops) and
            all(op.X is o.X for o in tomerge.ops))
        return none_shared or all_x_shared

    @staticmethod
    def is_mergeable(op1, op2):
        if op1.X is op2.X:
            # simple merge might be possible
            return (SigMerger.check([op1.Y, op2.Y])
                    and SigMerger.check([op1.A, op2.A]))

        # check if BSR merge is possible
        try:
            # Not using check() for A, because A must not be a view.
            SigMerger.check_signals([op1.A, op2.A])
            from scipy.sparse import bsr_matrix
            assert bsr_matrix
        except ImportError:
            warnings.warn("Skipping some optimization steps because SciPy is "
                          "not installed. Installing SciPy may result in "
                          "faster simulations.")
            return False
        except ValueError:
            return False
        return (SigMerger.check([op1.X, op2.X]) and
                SigMerger.check([op1.Y, op2.Y]) and
                op1.A.shape == op2.A.shape)

    @staticmethod
    def merge(ops):
        # Simple merge if all X are the same.
        if all(o.X is ops[0].X for o in ops):
            A, A_sigr = SigMerger.merge([o.A for o in ops])
            Y, Y_sigr = SigMerger.merge([o.Y for o in ops])
            return (operator.DotInc(A, ops[0].X, Y),
                    Merger.merge_dicts(A_sigr, Y_sigr))

        assert all(o1.X is not o2.X
                   for i, o1 in enumerate(ops) for o2 in ops[i+1:])

        # BSR merge if X differ
        X, X_sigr = SigMerger.merge([o.X for o in ops])
        Y, Y_sigr = SigMerger.merge([o.Y for o in ops])

        # Construct sparse A representation
        data = np.array([o.A.initial_value for o in ops])
        if data.ndim == 1:
            data = data.reshape((data.size, 1, 1))
        elif data.ndim == 2:
            data = data.reshape(data.shape + (1,))
        indptr = np.arange(len(ops) + 1, dtype=int)
        indices = np.arange(len(ops), dtype=int)
        name = 'bsr_merged<{first}, ..., {last}>'.format(
            first=ops[0].A.name, last=ops[-1].A.name)
        readonly = all([o.A.readonly for o in ops])
        A = Signal(data, name=name, readonly=readonly)
        A_sigr = {}
        for i, s in enumerate([o.A for o in ops]):
            A_sigr[s] = Signal(data[i], name="%s[%i]" % (s.name, i), base=A)
            assert np.all(s.initial_value == A_sigr[s].initial_value)
            assert s.shape == A_sigr[s].shape or (
                s.shape == () and A_sigr[s].shape == (1, 1))

        reshape = operator.reshape_dot(
            ops[0].A.initial_value, ops[0].X.initial_value,
            ops[0].Y.initial_value, tag=ops[0].tag)
        return (
            operator.BsrDotInc(
                A, X, Y, indices=indices, indptr=indptr, reshape=reshape),
            Merger.merge_dicts(X_sigr, Y_sigr, A_sigr))


@OpMerger.register(SimNeurons)
class SimNeuronsMerger(Merger):

    @staticmethod
    def is_mergeable(op1, op2):
        return (op1.neurons == op2.neurons and
                all(SigMerger.check(s) for s in
                    zip(op1.all_signals, op2.all_signals)))

    @staticmethod
    def merge(ops):
        def gather(ops, key):
            return [getattr(o, key) for o in ops]

        J, J_sigr = SigMerger.merge(gather(ops, 'J'))
        output, out_sigr = SigMerger.merge(gather(ops, 'output'))
        states = []
        states_sigr = {}
        for signals in zip(*gather(ops, 'states')):
            st, st_sigr = SigMerger.merge(signals)
            states.append(st)
            states_sigr.update(st_sigr)
        return (SimNeurons(ops[0].neurons, J, output, states),
                Merger.merge_dicts(J_sigr, out_sigr, states_sigr))


class SigMerger(object):

    @staticmethod
    def check(signals, axis=0):
        """Checks that all signals can be concatenated along a given axis.

        For views, this includes also a check that the signals have a common
        base and agree on the strides.

        In comparison to the ``check_*`` functions, this function  does
        not throw exceptions and allows for either signals or signal views.
        """
        if len(set(signals)) != len(signals):
            # Signal appears twice in list.
            return False

        if all(s.is_view for s in signals):
            try:
                SigMerger.check_views(signals, axis=axis)
            except ValueError:
                return False
        elif all(not s.is_view for s in signals):
            try:
                SigMerger.check_signals(signals, axis=axis)
            except ValueError:
                return False
        else:
            # Mix of signals and not signals
            return False

        # If we haven't failed yet, then the signals are compatible
        return True

    @staticmethod
    def check_signals(signals, axis=0):
        """Checks that all signals can be merged along a given axis.

        If this is not possible, or any signals are views, a
        ``ValueError`` will be raised.
        """
        if any(s.is_view for s in signals):
            raise ValueError("Cannot merge views.")

        for s in signals:
            if s.ndim != signals[0].ndim:
                raise ValueError(
                    "Signals must have the same number of dimensions.")
            if s.ndim <= 0 and s.initial_value != signals[0].initial_value:
                raise ValueError(
                    "0-d signals must have the same initial value.")
            if (s.shape[:axis] != signals[0].shape[:axis] or
                    s.shape[axis+1:] != signals[0].shape[axis+1:]):
                raise ValueError(
                    "Signals must have same shape except on concatenation "
                    "axis.")
            if s.dtype is not signals[0].dtype:
                raise ValueError(
                    "Signals must have the same dtype.")

    @staticmethod
    def check_views(signals, axis=0):
        """Checks that all signal views can be merged along a given axis.

        If this is not possible, or any signals are not views,
        a ``ValueError`` will be raised.

        ``signals`` must be ordered by the offset into the base signal.
        """
        if any(not s.is_view for s in signals):
            raise ValueError("Cannot merge non-views.")

        start = signals[0].offset
        for s in signals:
            if s.base is not signals[0].base:
                raise ValueError("Signals must share the same base.")
            if s.dtype is not signals[0].dtype:
                raise ValueError("Signals must have same dtype.")
            if s.ndim != signals[0].ndim:
                raise ValueError(
                    "Signals must have the same number of dimensions.")
            if s.strides != signals[0].strides:
                raise ValueError("Signals must have equal strides.")
            if (s.shape[:axis] != signals[0].shape[:axis] or
                    s.shape[axis+1:] != signals[0].shape[axis+1:]):
                raise ValueError(
                    "Signals must have same shape except on concatenation "
                    "axis.")
            if s.offset != start:
                raise ValueError("Views are not sequential.")
            start = s.offset + s.nbytes

    @staticmethod
    def merge(signals, axis=0):
        """Merges multiple signals or signal views into one contiguous signal.

        Note that if any of the signals are linked to another signal (by being
        the base of a view), the merged signal will not reflect those links
        anymore.

        Parameters
        ----------
        signals : sequence
            Signals to merge. Must not contain views.
        axis : int, optional
            Axis along which to concatenate the signals.

        Returns
        -------
        merged_signal : Signal
            The merged signal.
        replacements : dict
            Dictionary mapping from the old signals to new  signals that are
            a view into the merged signal. Used to replace old signals.
        """
        are_views = [s.is_view for s in signals]
        if all(are_views):
            return SigMerger.merge_views(signals, axis=axis)
        elif not any(are_views):
            return SigMerger.merge_signals(signals, axis=axis)
        else:
            raise ValueError("Cannot merged mixed views and non-views.")

    @staticmethod
    def merge_signals(signals, axis=0):
        """Merges multiple signal into one contiguous signal.

        Note that if any of the signals are linked to another signal (by being
        the base of a view), the merged signal will not reflect
        those links anymore.

        Parameters
        ----------
        signals : sequence
            Signals to merge. Must not contain views.
        axis : int, optional
            Axis along which to concatenate the signals.

        Returns
        -------
        merged_signal : Signal
            The merged signal.
        replacements : dict
            Dictionary mapping from the old signals to new  signals that are
            a view into the merged signal. Used to replace old signals.
        """
        SigMerger.check_signals(signals, axis=axis)

        if signals[0].ndim > 0:
            initial_value = np.concatenate(
                [s.initial_value for s in signals], axis=axis)
        else:
            initial_value = signals[0].initial_value
        readonly = all(s.readonly for s in signals)
        name = 'merged<' + signals[0].name + ', ..., ' + signals[-1].name + '>'
        merged_signal = Signal(initial_value, name=name, readonly=readonly)

        if signals[0].ndim > 0:
            replacements = {}
            start = 0
            for s in signals:
                size = s.shape[axis]
                indexing = [slice(None)] * initial_value.ndim
                indexing[axis] = slice(start, start + size)
                replacements[s] = merged_signal[tuple(indexing)]
                start += size
        else:
            replacements = {s: merged_signal for s in signals}

        return merged_signal, replacements

    @staticmethod
    def merge_views(signals, axis=0):
        """Merges multiple signal views into one continguous signal view.

        Parameters
        ----------
        signals : sequence
            Signals to merge. Must only contain views.
        axis : int, optional
            Axis along which to concatenate the signals.

        Returns
        -------
        merged_signal : Signal
            The merged signal.
        replacements : dict
            Dictionary mapping from the old signals to new  signals that are
            a view into the merged signal. Used to replace old signals.
        """
        SigMerger.check_views(signals, axis=axis)

        shape = (
            signals[0].shape[:axis] + (sum(s.shape[axis] for s in signals),) +
            signals[0].shape[axis+1:])
        initial_value = np.ndarray(
            buffer=signals[0].base.initial_value, dtype=signals[0].dtype,
            shape=shape, offset=signals[0].offset, strides=signals[0].strides)
        merged_signal = Signal(
            initial_value, name=signals[0].base.name, base=signals[0].base,
            readonly=all(s.readonly for s in signals))

        return merged_signal, {}


def groupby(lst, keyfunc=lambda item: item):
    """Groups the given list by the value returned by ``keyfunc``.

    Similar to ``itertools.groupby``, but returns a dict, and does not depend
    on the order of the input list.
    """
    d = defaultdict(list)
    for item in lst:
        d[keyfunc(item)].append(item)
    return d
