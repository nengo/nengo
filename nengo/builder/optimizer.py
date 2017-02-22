"""Operator graph optimizers."""

from collections import defaultdict
import logging
import time
import warnings

import numpy as np

from nengo.builder.neurons import SimNeurons
from nengo.builder import operator
from nengo.builder.operator import DotInc, ElementwiseInc, SlicedCopy
from nengo.builder.signal import Signal
from nengo.config import SupportRcDefaultsMixin
from nengo.params import BoolParam, IntParam, RcDefault
from nengo.utils.compat import iteritems, zip_longest
from nengo.utils.graphs import toposort, transitive_closure

logger = logging.getLogger(__name__)


class OpMergeOptimizer(SupportRcDefaultsMixin):
    """Optimizes the operator graph by merging operators.

    This reduces the number of iterators to iterate over in slow Python code
    (as opposed to fast C code) and due to the merging operations will be
    performed on larger chunks of sequential memory making better use of CPU
    caching and prefetching.

    The optimization algorithm has a worst case complexity of O(n^2 + e) where
    n is the number of operators and e the number of edges in the dependency
    graph. In practice the run time will be much better because not all n^2
    pairwise combinations of operators will be evaluated. A grouping depending
    on the operator type and view bases is done with dictionaries. This
    grouping can be done in amortized linear time and reduces the actual worst
    case runtime of the optimization algorithm to O(g*m^2 + e) where g is the
    number of groups and m the number of elements in a group. Moreover,
    information about memory alignment will be used to cut the inner loop short
    in many cases and gives a runtime much closer to linear in most cases.

    Note that both `model` and `dg` will be changed by the `optimize` method.

    Parameters
    ----------
    model : :class:`nengo.builder.builder.Model`
        Builder output to optimize.
    dg : dict
        Dict of the form ``{a: {b, c}}`` where ``b`` and ``c`` depend on a
        specifying the operator dependency graph of the model.
    """

    op_mergers = {}

    enabled = BoolParam('enabled', default=True, optional=False)
    max_passes = IntParam('max_passes', default=None, optional=True)

    def __init__(self, model, dg):
        self.enabled = RcDefault
        self.max_passes = RcDefault

        self.model = model
        self.dg = dg

        # Dict mapping from a Signal to the Operators accessing those signals.
        self._sig2op = None
        # Set of Operators merged during the current optimization pass.
        self._merged = None

    @classmethod
    def supports_merge(cls, op_type):
        return op_type in cls.op_mergers

    @classmethod
    def can_merge_ops(cls, op1, op2):
        return (
            type(op1) == type(op2) and
            type(op1) in cls.op_mergers and
            cls.op_mergers[type(op1)].can_merge(op1, op2))

    @classmethod
    def merge_ops(cls, ops):
        return cls.op_mergers[type(ops[0])].merge(ops)

    @classmethod
    def register_merger(cls, op_type):
        def register(merger):
            if op_type in cls.op_mergers:
                warnings.warn(
                    "Merger for operator of type {} overwritten.".format(
                        op_type))
            cls.op_mergers[op_type] = merger()
            return merger
        return register

    def optimize(self):
        """Perform the optimization."""
        if not self.enabled:
            return

        logger.info("Running %s ...", self.__class__.__name__)

        # We try first to merge operators with views only as these have a fixed
        # order for the memory alignment whereas operators without views could
        # be merged in a random order. Merging the views of operators will
        # propagate requirements in the memory ordering via the other
        # associated signals of the operator to other operators.
        #
        # Once no more operators with views can be merged, we try to merge
        # operators without views and then try again merging views (because
        # each operator merge might generate new views).

        before, after = None, None
        i = 0
        only_merge_ops_with_view = True
        while only_merge_ops_with_view or after < before:
            if self.max_passes is not None and i >= self.max_passes:
                break

            i += 1
            self._log_counts()

            only_merge_ops_with_view = before is None or before != after
            before = len(self.dg)
            t_start = time.time()

            self._perform_single_pass(only_merge_ops_with_view)

            t_end = time.time()
            after = len(self.dg)
            logger.info(
                "Pass %i [%s]: Reduced %i to %i operators in %fs.",
                i, "only views" if only_merge_ops_with_view else "all",
                before, after, t_end - t_start)

        self._reinitialize_model_ops(self.dg)

    def _perform_single_pass(self, only_merge_ops_with_view):
        """Perform a single optimization pass.

        Parameters
        ----------
        only_merge_ops_with_view : bool
            Limits operator merges to operators with views.
        """
        self._init_pass_helper_vars()
        op_replacements, sig_replacements = self._perform_merges(
            only_merge_ops_with_view)
        ops = list(op_replacements.values())
        self._replace_op_signals(ops, sig_replacements)
        self._update_dg(op_replacements)
        self._update_model_sigs(sig_replacements)

    def _init_pass_helper_vars(self):
        """Initializes the internal helper variables for an optimization pass.
        """
        self._sig2op = defaultdict(list)
        for op in self.dg:
            for s in op.all_signals:
                self._sig2op[s].append(op)
        self._merged = set()

    def _ops_by_type(self, ops):
        """Returns `ops` as a dictionary sorted by type."""
        by_type = defaultdict(list)
        for op in ops:
            by_type[type(op)].append(op)
        return by_type

    def _log_counts(self):
        """Logs for each operator type the count in the dependency graph."""
        if logger.isEnabledFor(logging.DEBUG):
            for tp, ops in self._ops_by_type(self.dg).items():
                logger.debug("%s: %i", tp, len(ops))

    def _perform_merges(self, only_merge_ops_with_view):
        """Go through all operators and merge them where possible.

        Parameters
        ----------
        only_merge_ops_with_view : bool
            Limit merges to operators with views.

        Returns
        -------
        op_replacements : dict
            Dictionary that maps from original operators to the newly merged
            operators. To be used to update the dependency graph.
        sig_replacements : dict
            Dictionary that maps from original signals to new signals resulting
            from the merges. To be used to update the signals referenced by
            the operators.
        """
        step_order = toposort(self.dg)
        tc = transitive_closure(self.dg, step_order)
        by_type = self._ops_by_type(step_order)

        op_replacements = {}
        sig_replacements = {}

        # Note that the following loops will stop as soon as any operators
        # have been merged, so that merges are performed on at most one type
        # of operators. This function will be called again once the dependency
        # graph and signals have been updated with those merges to continue
        # merging other operator types.

        # First loop with a heuristic order to reduce runtime and number of
        # passes
        for tp in [ElementwiseInc, SlicedCopy, DotInc, SimNeurons]:
            opr, sigr = self._perform_merges_for_subset(
                by_type[tp], tc, only_merge_ops_with_view)
            op_replacements.update(opr)
            sig_replacements.update(sigr)
            del by_type[tp]  # no need to process tp again in the second loop
            n_changes = sum(new is not old for old, new in iteritems(opr))
            if not only_merge_ops_with_view and n_changes > 0:
                # If we're not only merging views, the memory layout changes
                # and non-views are turned into views. In that case we need
                # to update the signals the operators are referring to before
                # trying to merge a different type of operators. Thus, we break
                # the loop here (and have a similar condition in the second
                # loop). The updating of the signal references is done outside
                # of this function.
                break

        # Process remaining operations
        for tp, subset in by_type.items():
            if self.supports_merge(tp) and (
                    only_merge_ops_with_view or n_changes <= 0):
                opr, sigr = self._perform_merges_for_subset(
                    subset, tc, only_merge_ops_with_view)
                op_replacements.update(opr)
                sig_replacements.update(sigr)
            else:
                # If an operator type does not support merging we still need to
                # add the operators to op_replacements to not have them removed
                # from the dependency graph.
                op_replacements.update({op: op for op in subset})

        sig_replacements.update(self._get_sig_view_replacements(
            op_replacements.values(), sig_replacements))

        return op_replacements, sig_replacements

    def _get_view_indices(self, op):
        """Enumerates all signals of an operator and returns the indices that
        correspond to signal views."""
        view_indices = []
        for idx, s in enumerate(op.all_signals):
            if s.is_view:
                view_indices.append(idx)
        return view_indices

    def _mergeable(self, op1, op2, op1_view_indices, tc):
        """Determines if `op1` can be merged with `op2` given that `op1`'s
        signals corresponding to `op1_view_indices` are views. `tc` needs to be
        the transitive closure of the dependency graph.
        """
        # Ensure the operators don't form a dependency cycle and that op2
        # hasn't been merged with another operator yet. (Whether op1 has been
        # merged with another operator will already be checked in
        # _perform_merges_for_view_subset outside of this function
        # to reduce the number of checks inside the inner loop of the
        # optimizer.)
        independent = (
            (op1 not in tc[op2] and op2 not in tc[op1]) and
            op2 not in self._merged)

        op1_sigs = op1.all_signals
        op2_sigs = op2.all_signals

        # Ensure that the signal views are compatible, otherwise the operators
        # cannot be merged.
        views_match = op1_view_indices == self._get_view_indices(op2)
        for idx in op1_view_indices:
            s1 = op1_sigs[idx]
            s2 = op2_sigs[idx]
            views_match &= s1.dtype is s2.dtype
            views_match &= s1.base == s2.base
            views_match &= s1.strides == s2.strides

        return independent and views_match and self.can_merge_ops(op1, op2)

    def _is_memory_access_sequential(self, ops):
        """Checks that the corresponding signals of the operators `ops` are all
        views onto a sequential piece of memory. If this is the case the views
        can be merged into a single view onto the same memory."""
        for signals in zip(*[op.all_signals for op in ops]):
            if all(not s.is_view for s in signals):
                continue
            elif not all(s.is_view for s in signals):
                return False

            end = signals[0].offset + signals[0].size * signals[0].itemsize
            for s in signals[1:]:
                if end != s.offset:
                    return False
                end = s.offset + s.size * s.itemsize
        return True

    @staticmethod
    def _first_view_offset(op):
        """Returns the offset of the first signal view of operator `op`.

        Returns 0 if the operator has no signal views."""
        for s in op.all_signals:
            if s.is_view:
                return s.offset
        return 0

    @staticmethod
    def _first_view_size(op):
        """Returns the size of the first signal view of operator `op`.

        Returns 0 if the operator has no signal views."""
        for s in op.all_signals:
            if s.is_view:
                return s.size * s.itemsize
        return 0

    @staticmethod
    def _first_view_base(op):
        """Returns the base of the first signal view of operator `op`.

        Returns ``None`` if the operator has no signal views."""
        for s in op.all_signals:
            if s.is_view:
                return s.base
        return None

    def _check_sequential(self, op1, op2):
        """Checks that the first signal views of `op1` and `op2` are
        non-overlapping and the view of `op2` starts after the view of `op1`.

        This is similar to the `_is_memory_access_sequential`, but has some
        minute differences. In contrast to `_is_memory_access_sequential`, this
        function

        * only checks the first view instead of all views
        * does not check that all signals are views
        * only checks two operators at a time instead of a whole list
        * only returns ``True`` if there is a gap between the end of the view
          of `op1` and the start of the view of `op2`

        This function is only intended to give a quick indication whether the
        merge of `op1` and `op2` could potentially be possible to allow for an
        early break out of the inner loop given a list of sorted operators by
        the offset of their first signal view; whereas
        `_is_memory_access_sequential` is used for the final check that all
        the views match up in their memory alignment.
        """
        return (
            self._first_view_offset(op1) + self._first_view_size(op1) <
            self._first_view_offset(op2))

    def _perform_merges_for_subset(
            self, subset, tc, only_merge_ops_with_view=True):
        """Performs operator merges for a subset of operators.

        Parameters
        ----------
        subset : list
            Subset of operators.
        tc : dict
            Transitive closure of dependency graph.
        only_merge_ops_with_view : bool
            Limit merges to operators with views.

        Returns
        -------
        op_replacements : dict
            Dictionary that maps from original operators to the newly merged
            operators. To be used to update the dependency graph.
        sig_replacements : dict
            Dictionary that maps from original signals to new signals resulting
            from the merges. To be used to update the signals referenced by
            the operators.
        """
        op_replacements = {op: op for op in subset}
        sig_replacements = {}
        by_view = defaultdict(list)

        # Sort by base of views
        for op in subset:
            by_view[self._first_view_base(op)].append(op)

        if only_merge_ops_with_view and None in by_view:
            del by_view[None]

        for view_subset in by_view.values():
            if len(view_subset) > 1:
                self._perform_merges_for_view_subset(
                    view_subset, tc, op_replacements, sig_replacements)

        return op_replacements, sig_replacements

    def _perform_merges_for_view_subset(
            self, subset, tc, op_replacements, sig_replacements):
        """Perform merges for a subset of operators with the same view base.

        Parameters
        ----------
        subset : list
            Subset of operators. These need to have the same view base (can be
            None if it is None for all) for their first signal in
            `all_signals`.
        tc : dict
            Transitive closure of dependency graph.
        only_merge_ops_with_view : bool
            Limit merges to operators with views.
        op_replacements : dict
            Dictionary that maps from original operators to the newly merged
            operators. To be used to update the dependency graph. This dict
            will be updated by this function.
        sig_replacements : dict
            Dictionary that maps from original signals to new signals resulting
            from the merges. To be used to update the signals referenced by
            the operators. This dict will be updated by this function.
        """
        view_indices = []
        # Sort to have sequential memory.
        offsets = np.array([self._first_view_offset(op) for op in subset])
        sort_indices = np.argsort(offsets)
        offsets = offsets[sort_indices]
        sorted_subset = [subset[i] for i in sort_indices]

        for i, op1 in enumerate(sorted_subset):
            if op1 in self._merged:
                # Cannot merge merged operator again until dependency graph
                # has been updated (happens at the end of one optimization
                # pass by calling _update_dg).
                continue

            view_indices = self._get_view_indices(op1)

            # Find operators that can be merged.
            merge = [op1]
            # For a merge to be possible the view of the next operator has to
            # start where the view of op1 ends. Because we have sorted the
            # operators by the start of their views we can do a binary search
            # and potentially skip a number of operators at the beginning.
            start = np.searchsorted(
                offsets, offsets[i] + self._first_view_size(op1), side='left')
            for op2 in sorted_subset[start:]:
                can_merge = (
                    op2 not in self._merged and
                    self._mergeable(merge[-1], op2, view_indices, tc) and
                    self._is_memory_access_sequential([merge[-1], op2]))
                for op in merge:
                    if any(s in op.all_signals for s in op2.all_signals):
                        can_merge = False
                        break
                if can_merge:
                    merge.append(op2)
                elif self._check_sequential(merge[-1], op2):
                    # If this check is true the view of op2 does not
                    # immediately follow the view of the operators being
                    # merged. Because we iterate over the operators sorted by
                    # view offset there will be a gap between the end of the
                    # operators being merged and all remaining operators to
                    # loop over. With such a gap a merge is never possible and
                    # we can cut the loop short.
                    break

            if len(merge) > 1:
                self._merge(merge, op_replacements, sig_replacements)

        return op_replacements, sig_replacements

    def _merge(self, merge, op_replacements, sig_replacements):
        """Merges the operators in `merge` and updates `op_replacements` and
        `sig_replacements` with the required replacements of operators and
        signals.

        It will also update the internal list of merged operators to prevent
        further merges on the same operators before all required operators and
        signals have been replaced.
        """
        merged_op, merged_sig = self.merge_ops(merge)
        self._merged.update(merge)
        for op in merge:
            op_replacements[op] = merged_op
            # Mark all operators referencing the same signals as merged
            # (even though they are not) to prevent them from getting
            # merged before their signals have been updated.
            for s in op.all_signals:
                self._merged.update(self._sig2op[s])
        sig_replacements.update(merged_sig)

    def _update_dg(self, op_replacements):
        """Updates the dependency graph.

        This involves doing all operator replacements in `op_replacements` and
        ensuring that dependencies are preserved.
        """
        dg = {}
        for op in op_replacements:
            if op_replacements[op] not in dg:
                dg[op_replacements[op]] = set()
            for x in self.dg[op]:
                dg[op_replacements[op]].add(op_replacements[x])
        self.dg.clear()
        self.dg.update(dg)

    @staticmethod
    def _map_onto_op_signals(fn, ops):
        """Maps function `fn` on all signals of all operators in the sequence
        `ops`.
        """
        for op in ops:
            for key in dir(op):
                sig = getattr(op, key)
                if isinstance(sig, Signal):
                    setattr(op, key, fn(sig))

            op.sets = [fn(s) for s in op.sets]
            op.incs = [fn(s) for s in op.incs]
            op.reads = [fn(s) for s in op.reads]
            op.updates = [fn(s) for s in op.updates]

    @classmethod
    def _get_sig_view_replacements(cls, ops, sig_replacements):
        """Given signal replacements `sig_replacements`, returns additional
        signal replacements required to account for views into a replaced
        signal.

        Parameters
        ----------
        ops : sequence
            Operators to determine the additional signal replacements for.
        sig_replacements : dict
            Signal replacements.

        Returns
        -------
        dict
            Additional required replacements of views into replaced bases of
            signals.
        """
        view_replacements = {}

        def get_sig_view_replacement(s):
            if s.is_view and s.base in sig_replacements:
                base_replacement = sig_replacements[s.base]
                offset = s.offset
                strides = tuple(a // b * c for a, b, c in zip_longest(
                    s.strides, s.base.strides, base_replacement.strides,
                    fillvalue=1))
                if base_replacement.is_view:
                    offset += base_replacement.offset
                    base_replacement = base_replacement.base
                buf = base_replacement.initial_value
                initial_value = np.ndarray(
                    buffer=buf, dtype=s.dtype, shape=s.shape, offset=offset,
                    strides=strides)
                view_replacements[s] = Signal(
                    initial_value, name=s.name, base=base_replacement,
                    readonly=s.readonly)
            return s

        # This call is not actually changing the signals of the operators as
        # get_sig_view_replacement will return the unmodified signals and
        # store additional required replacements in view_replacements.
        cls._map_onto_op_signals(get_sig_view_replacement, ops)
        return view_replacements

    @classmethod
    def _replace_op_signals(cls, ops, sig_replacements):
        """Apply signal repacements `sig_replacements` to all opeators in
        `ops`.
        """
        cls._map_onto_op_signals(lambda s: sig_replacements.get(s, s), ops)

    def _reinitialize_model_ops(self, ops):
        """Reinitializes the operator list of the builder model from `ops`."""
        self.model.operators = []
        for op in ops:
            self.model.add_op(op)

    def _update_model_sigs(self, sig_replacements):
        """Updates the signal dictionary in the builder model with
        `sig_replacements`.
        """
        for k in self.model.sig:
            for name, v in self.model.sig[k].items():
                self.model.sig[k][name] = sig_replacements.get(v, v)


class AbstractMerger(object):
    def can_merge(self, op1, op2):
        """Checks if `op1` can be merged with `op2`.

        This function is expected to be transitive and symmetric.
        """
        raise NotImplementedError()

    def merge(self, ops):
        """Merges the operators `ops`.

        May lead to undefined behaviour if ``can_merge`` returns ``False`` for
        any of the elements in ``ops``.

        Returns
        -------
        Operator
            The merged operator.
        dict
            Dictionary mapping old signals to new signals to update the
            signals of other operators.
        """
        raise NotImplementedError()


@OpMergeOptimizer.register_merger(operator.TimeUpdate)
class TimeUpdateMerger(AbstractMerger):
    def can_merge(self, op1, op2):
        return True

    def merge(self, ops):
        replacements = {}
        step = merge_signals_or_views([o.step for o in ops], replacements)
        time = merge_signals_or_views([o.time for o in ops], replacements)
        return operator.TimeUpdate(step, time), replacements


@OpMergeOptimizer.register_merger(operator.Reset)
class ResetMerger(AbstractMerger):
    def can_merge(self, op1, op2):
        return compatible([op1.dst, op2.dst]) and op1.value == op2.value

    def merge(self, ops):
        replacements = {}
        dst = merge_signals_or_views([o.dst for o in ops], replacements)
        return operator.Reset(dst, ops[0].value), replacements


@OpMergeOptimizer.register_merger(operator.Copy)
class CopyMerger(AbstractMerger):
    def can_merge(self, op1, op2):
        return (
            compatible([op1.dst, op2.dst]) and compatible([op1.src, op2.src]))

    def merge(self, ops):
        replacements = {}
        dst = merge_signals_or_views([o.dst for o in ops], replacements)
        src = merge_signals_or_views([o.src for o in ops], replacements)
        return operator.Copy(src, dst), replacements


@OpMergeOptimizer.register_merger(operator.SlicedCopy)
class SlicedCopyMerger(AbstractMerger):
    def can_merge(self, op1, op2):
        return (
            compatible([op1.src, op2.src]) and
            compatible([op1.dst, op2.dst]) and
            op1.src_slice is Ellipsis and op1.dst_slice is Ellipsis and
            op2.src_slice is Ellipsis and op2.dst_slice is Ellipsis and
            op1.inc == op2.inc)

    def _merged_slice(self, signals, slices):
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

    def merge(self, ops):
        src_sigs = [o.src for o in ops]
        dst_sigs = [o.dst for o in ops]

        replacements = {}
        src = merge_signals_or_views(src_sigs, replacements)
        dst = merge_signals_or_views(dst_sigs, replacements)
        src_slice = self._merged_slice(src_sigs, [o.src_slice for o in ops])
        dst_slice = self._merged_slice(dst_sigs, [o.dst_slice for o in ops])
        return operator.SlicedCopy(
            src, dst, src_slice=src_slice, dst_slice=dst_slice,
            inc=ops[0].inc), replacements


@OpMergeOptimizer.register_merger(operator.ElementwiseInc)
class ElementwiseIncMerger(AbstractMerger):
    def can_merge(self, op1, op2):
        return (
            compatible([op1.A, op2.A], axis=op1.A.ndim - 1) and
            compatible([op1.X, op2.X], axis=op1.X.ndim - 1) and
            compatible([op1.Y, op2.Y], axis=op1.Y.ndim - 1))

    def merge(self, ops):
        replacements = {}
        A = merge_signals_or_views(
            [o.A for o in ops], replacements, axis=ops[0].A.ndim - 1)
        X = merge_signals_or_views(
            [o.X for o in ops], replacements, axis=ops[0].X.ndim - 1)
        Y = merge_signals_or_views(
            [o.Y for o in ops], replacements, axis=ops[0].Y.ndim - 1)
        return operator.ElementwiseInc(A, X, Y), replacements


@OpMergeOptimizer.register_merger(operator.DotInc)
class DotIncMerger(AbstractMerger):
    def can_merge(self, op1, op2):
        if op1.X is op2.X:
            # simple merge might be possible
            return compatible([op1.Y, op2.Y]) and compatible([op1.A, op2.A])

        # check if BSR merge is possible
        try:
            # Not using compatible() for A, because A must not be a view.
            check_signals_mergeable([op1.A, op2.A])
            from scipy.sparse import bsr_matrix
            assert bsr_matrix
        except (ValueError, ImportError):
            return False
        return (compatible([op1.X, op2.X]) and
                compatible([op1.Y, op2.Y]) and
                op1.A.shape == op2.A.shape)

    def merge(self, ops):
        replacements = {}

        # Simple merge if all X are the same.
        if all(o.X is ops[0].X for o in ops):
            A = merge_signals_or_views([o.A for o in ops], replacements)
            Y = merge_signals_or_views([o.Y for o in ops], replacements)
            return operator.DotInc(A, ops[0].X, Y), replacements

        # BSR merge if X differ
        X = merge_signals_or_views([o.X for o in ops], replacements)
        Y = merge_signals_or_views([o.Y for o in ops], replacements)

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
        for i, s in enumerate([o.A for o in ops]):
            replacements[s] = Signal(
                data[i], name="%s[%i]" % (s.name, i), base=A)
            assert np.all(s.initial_value == replacements[s].initial_value)
            assert s.shape == replacements[s].shape

        reshape = operator.reshape_dot(
            ops[0].A.initial_value, ops[0].X.initial_value,
            ops[0].Y.initial_value, tag=ops[0].tag)
        return (
            operator.BsrDotInc(
                A, X, Y, indices=indices, indptr=indptr, reshape=reshape),
            replacements)


@OpMergeOptimizer.register_merger(SimNeurons)
class SimNeuronsMerger(AbstractMerger):
    def can_merge(self, op1, op2):
        return (
            op1.neurons == op2.neurons and
            all(compatible(s) for s in zip(op1.all_signals, op2.all_signals)))

    def _gather(self, ops, key):
        return [getattr(o, key) for o in ops]

    def merge(self, ops):
        replacements = {}
        J = merge_signals_or_views(self._gather(ops, 'J'), replacements)
        output = merge_signals_or_views(
            self._gather(ops, 'output'), replacements)
        states = []
        for signals in zip(*self._gather(ops, 'states')):
            states.append(merge_signals_or_views(signals, replacements))
        return SimNeurons(ops[0].neurons, J, output, states), replacements


def compatible(signals, axis=0):
    """Checks that all signals have a compatible shape along a given axis
    to allow for concatenation.

    For views this includes also a check that the signals have a common
    base and agree on the strides.

    In comparison to the `check_*_mergeable` functions, this function does
    not throw exceptions (either behavior might be desired dependent on the
    context) and allows for either signals or signal views.
    """
    if len(set(signals)) != len(signals):
        # Signal appears twice in list.
        return False
    for s in signals:
        if s.ndim != signals[0].ndim:
            return False
        if s.ndim <= 0 and s.initial_value != signals[0].initial_value:
            return False
        if (s.shape[:axis] != signals[0].shape[:axis] or
                s.shape[axis+1:] != signals[0].shape[axis+1:]):
            return False
        if s.dtype is not signals[0].dtype:
            return False
        if s.is_view or signals[0].is_view:
            if s._base is not signals[0]._base:
                return False
            if s.strides != signals[0].strides:
                return False
    return True


def check_signals_mergeable(signals, axis=0):
    """Checks that all signals can be merged along a given axis and raises
    a `ValueError` if this is not possible.

    For views a ValueError is always raised.
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
                "Signal mus have the same dtype.")


def merge_signals(signals, replacements, axis=0):
    """Merges multiple signal into one signal with sequential memory
    allocation.

    Note that if any of the signals are linked to another signal (by being
    the base of a view), the merged signal will not reflect
    those links anymore.

    Parameters
    ----------
    signals : sequence
        Signals to merge. Must not contain views.
    axis : int, optional
        Axis along which to concatenate the signals.
    replacements : dict
        Dictionary to update with a mapping from the old signals to new
        signals that are a view into the merged signal and can be used to
        replace the old signals.

    Returns
    -------
    merged_signal : Signal
        The merged signal.
    """
    check_signals_mergeable(signals, axis=axis)

    if signals[0].ndim > 0:
        initial_value = np.concatenate(
            [s.initial_value for s in signals], axis=axis)
    else:
        initial_value = signals[0].initial_value
    readonly = all(s.readonly for s in signals)
    name = 'merged<' + signals[0].name + ', ..., ' + signals[-1].name + '>'
    merged_signal = Signal(initial_value, name=name, readonly=readonly)

    if signals[0].ndim > 0:
        start = 0
        for s in signals:
            size = s.shape[axis]
            indexing = [slice(None)] * initial_value.ndim
            indexing[axis] = slice(start, start + size)
            replacements[s] = merged_signal[tuple(indexing)]
            start += size
    else:
        replacements.update({s: merged_signal for s in signals})

    return merged_signal


def check_views_mergeable(signals, axis=0):
    """Checks that all signal views can be merged along a given axis and
    raises a `ValueError` if this is not possible.

    The signals need to be ordered by their offset into the base signal.

    For non-views a ValueError is always raised.
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
        start = s.offset + s.size * s.itemsize


def merge_views(signals, axis=0):
    """Merges multiple signal views into one signal view with sequential
    memory access.

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
    """
    check_views_mergeable(signals, axis=axis)

    shape = (
        signals[0].shape[:axis] + (sum(s.shape[axis] for s in signals),) +
        signals[0].shape[axis+1:])
    initial_value = np.ndarray(
        buffer=signals[0].base.initial_value, dtype=signals[0].dtype,
        shape=shape, offset=signals[0].offset, strides=signals[0].strides)
    return Signal(
        initial_value, name=signals[0].base.name, base=signals[0].base,
        readonly=all(s.readonly for s in signals))


def merge_signals_or_views(signals, replacements, axis=0):
    """Merges multiple signal (or signal views) into one signal with
    sequential memory allocation.

    Note that if any of the signals are linked to another signal (by being
    the base of a view), the merged signal will not reflect those links
    anymore.

    Parameters
    ----------
    signals : sequence
        Signals to merge. Must not contain views.
    axis : int, optional
        Axis along which to concatenate the signals.
    replacements : dict
        Dictionary to update with a mapping from the old signals to new
        signals that are a view into the merged signal and can be used to
        replace the old signals.

    Returns
    -------
    merged_signal : Signal
        The merged signal.
    """
    are_views = [s.is_view for s in signals]
    if all(are_views):
        return merge_views(signals, axis=axis)
    elif not any(are_views):
        return merge_signals(signals, replacements, axis=axis)
    else:
        raise ValueError("Cannot merged mixed views and non-views.")
