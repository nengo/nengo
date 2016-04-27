"""Operator graph optimizers."""

from collections import defaultdict
import logging

import numpy as np

from nengo.builder.neurons import SimNeurons
from nengo.builder.operator import Copy, Reset
from nengo.builder.signal import Signal
from nengo.utils.compat import zip_longest
from nengo.utils.graphs import toposort, transitive_closure

logger = logging.getLogger(__name__)


class OpMergeOptimizer(object):
    """Optimizes the operator graph by merging operators.

    This reduces the number of iterators to iterate over in slow Python code
    and due to the merging operations will be performed on larger chunks of
    sequential memory making better use of CPU caching and prefetching.

    The optimization algorithm has a complexity of O(n^2 + e) where n is the
    number of operators and e the number of edges in the dependency graph.

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

        self._sig2op = None
        self._merged = None

    def optimize(self):
        """Perform the optimization."""
        logger.info("Running %s ...", self.__class__.__name__)
        before, after = None, None
        i = 0
        while i == 0 or after < before:
            i += 1
            self._log_counts()
            before = len(self.dg)
            self._perform_single_pass()
            after = len(self.dg)
            logger.info(
                "Pass %i: Reduced %i to %i operators.", i, before, after)

        self._reinitialize_model_ops(self.dg)

    def _perform_single_pass(self):
        self._init_pass_helper_vars()
        op_replacements, sig_replacements = self._perform_merges()
        ops = list(op_replacements.values())
        self._replace_op_signals(ops, sig_replacements)
        self._update_dg(op_replacements)
        self._update_model_sigs(sig_replacements)

    def _init_pass_helper_vars(self):
        self._sig2op = defaultdict(list)
        for op in self.dg:
            for s in op.all_signals:
                self._sig2op[s].append(op)
        self._merged = set()

    def _ops_by_type(self, ops):
        by_type = defaultdict(list)
        for op in ops:
            by_type[type(op)].append(op)
        return by_type

    def _log_counts(self):
        if logger.isEnabledFor(logging.DEBUG):
            for tp, ops in self._ops_by_type(self.dg).items():
                logger.debug("%s: %i", tp, len(ops))

    def _perform_merges(self):
        step_order = toposort(self.dg)
        tc = transitive_closure(self.dg, step_order)

        by_type = defaultdict(list)
        for op in step_order:
            by_type[type(op)].append(op)

        op_replacements = {}
        sig_replacements = {}

        # Do merges on most expensive operations first
        for tp in [SimNeurons, Reset, Copy]:
            opr, sigr = self._perform_merges_for_subset(by_type[tp], tc)
            op_replacements.update(opr)
            sig_replacements.update(sigr)
            del by_type[tp]

        # Process remaining operations
        for tp, subset in by_type.items():
            if tp.supports_merge():
                opr, sigr = self._perform_merges_for_subset(subset, tc)
                op_replacements.update(opr)
                sig_replacements.update(sigr)
            else:
                op_replacements.update({op: op for op in subset})

        sig_replacements.update(self._get_sig_view_replacements(
            op_replacements.values(), sig_replacements))

        return op_replacements, sig_replacements

    def _get_view_indices(self, op):
        view_indices = []
        for idx, s in enumerate(op.all_signals):
            if s.is_view:
                view_indices.append(idx)
        return view_indices

    def _mergeable(self, op1, op2, op1_view_indices, tc):
        independent = (
            (op1 not in tc[op2] and op2 not in tc[op1]) and
            op2 not in self._merged)

        op1_sigs = op1.all_signals
        op2_sigs = op2.all_signals

        views_match = op1_view_indices == self._get_view_indices(op2)
        for idx in op1_view_indices:
            s1 = op1_sigs[idx]
            s2 = op2_sigs[idx]
            views_match &= s1.dtype is s2.dtype
            views_match &= s1.base == s2.base
            views_match &= s1.strides == s2.strides

        return independent and views_match and op1.can_merge(op2)

    def _is_memory_access_sequential(self, ops):
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

    def _perform_merges_for_subset(self, subset, tc):
        op_replacements = {op: op for op in subset}
        sig_replacements = {}
        view_indices = []

        def view_offset(op):
            for s in op.all_signals:
                if s.is_view:
                    return s.offset
            return 0

        def view_size(op):
            for s in op.all_signals:
                if s.is_view:
                    return s.size * s.itemsize
            return 0

        subset = sorted(
            subset, lambda o1, o2: view_offset(o1) - view_offset(o2))

        for i, op1 in enumerate(subset):
            if op1 in self._merged:
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
                elif (view_offset(merge[-1]) + view_size(merge[-1]) <
                      view_offset(op2)):
                    break

            if len(merge) > 1:
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

        return op_replacements, sig_replacements

    def _update_dg(self, op_replacements):
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
        cls._map_onto_op_signals(lambda s: sig_replacements.get(s, s), ops)

    def _reinitialize_model_ops(self, ops):
        self.model.operators = []
        for op in ops:
            self.model.add_op(op)

    def _update_model_sigs(self, sig_replacements):
        for k in self.model.sig:
            for name, v in self.model.sig[k].items():
                self.model.sig[k][name] = sig_replacements.get(v, v)
