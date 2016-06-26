"""Operator graph optimizers."""

from collections import defaultdict
import logging
import time

import numpy as np

from nengo.builder.neurons import SimNeurons
from nengo.builder.operator import DotInc, ElementwiseInc, SlicedCopy
from nengo.builder.signal import Signal
from nengo.utils.compat import zip_longest
from nengo.utils.graphs import toposort, transitive_closure

logger = logging.getLogger(__name__)


class OpMergeOptimizer(object):
    """Optimizes the operator graph by merging operators.

    This reduces the number of iterators to iterate over in slow Python code
    (as opposed to fast C code) and due to the merging operations will be
    performed on larger chunks of sequential memory making better use of CPU
    caching and prefetching.

    The optimization algorithm has a worst case complexity of O(n^2 + e) where
    n is the number of operators and e the number of edges in the dependency
    graph. In practice the run time will be much better because not all n^2
    pairwise combinations of operators will be evaluated. A grouping depending
    on the operator type and view bases will be done with dictionaries. This
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
        Dict of the form {a: {b, c}} where b and c depend on a specifying the
        operator dependency graph of the model.
    """

    def __init__(self, model, dg):
        self.model = model
        self.dg = dg

        # Internal dict mapping Signals to Operators accessing those signals.
        self._sig2op = None
        # Internal set of Operators merged during the current optimization
        # pass.
        self._merged = None

    def optimize(self):
        """Perform the optimization."""
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

        # Heuristic order to reduce runtime and number of passes
        for tp in [ElementwiseInc, SlicedCopy, DotInc, SimNeurons]:
            opr, sigr = self._perform_merges_for_subset(
                by_type[tp], tc, only_merge_ops_with_view)
            op_replacements.update(opr)
            sig_replacements.update(sigr)
            del by_type[tp]  # no need to process tp again in the second loop
            if not only_merge_ops_with_view and len(opr) > 0:
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
            if tp.supports_merge() and (
                    only_merge_ops_with_view or len(opr) <= 0):
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
        # merged with another operator will be checked outside of this function
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

        return independent and views_match and op1.can_merge(op2)

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
    def _view_offset(op):
        """Returns the offset of the first signal view of operator `op`.

        Returns 0 if the operator has no signal views."""
        for s in op.all_signals:
            if s.is_view:
                return s.offset
        return 0

    @staticmethod
    def _view_size(op):
        """Returns the size of the first signal view of operator `op`.

        Returns 0 if the operator has no signal views."""
        for s in op.all_signals:
            if s.is_view:
                return s.size * s.itemsize
        return 0

    @staticmethod
    def _view_base(op):
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
            self._view_offset(op1) + self._view_size(op1) <
            self._view_offset(op2))

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
            by_view[self._view_base(op)].append(op)

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
        subset = sorted(subset, key=self._view_offset)

        for i, op1 in enumerate(subset):
            if op1 in self._merged:
                # Cannot merge merged operator again until dependency graph
                # has been updated (happens outside of this function).
                continue

            view_indices = self._get_view_indices(op1)

            # Find operators that can be merged.
            merge = [op1]
            for op2 in subset[i+1:]:
                can_merge = (
                    op2 not in self._merged and
                    self._mergeable(merge[-1], op2, view_indices, tc) and
                    self._is_memory_access_sequential([merge[-1], op2]))
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
        merged_op, merged_sig = merge[0].merge(merge[1:])
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
