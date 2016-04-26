"""Operator graph optimizers."""

from collections import defaultdict
import logging

import numpy as np

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

    def optimize(self):
        """Perform the optimization."""
        logger.info("Running %s ...", self.__class__.__name__)
        self._log_counts()
        before = len(self.dg)
        op_replacements, sig_replacements = self._perform_merges()
        ops = list(op_replacements.values())
        self._replace_op_signals(ops, sig_replacements)
        self._update_dg(op_replacements)
        self._reinitialize_model_ops(ops)
        self._update_model_sigs(sig_replacements)
        after = len(self.dg)
        logger.info("%i operations reduced to %i operations.", before, after)
        self._log_counts()

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

        op2idx = {op: i for i, op in enumerate(step_order)}
        sig2op = {}
        for op in step_order:
            for s in op.all_signals:
                sig2op[s] = op

        op_replacements = {}
        sig_replacements = {}
        merged = np.zeros_like(step_order, dtype=bool)
        sig_replaced = np.zeros_like(step_order, dtype=bool)
        tc = transitive_closure(self.dg, step_order)

        for i, op1 in enumerate(step_order):
            if merged[i]:
                continue

            op1_has_view = any(s.is_view for s in op1.all_signals)

            merge = []
            if not sig_replaced[i] and not op1_has_view:
                for j, op2 in enumerate(step_order[i+1:], start=i+1):
                    op2_has_view = any(s.is_view for s in op2.all_signals)
                    independent = (
                        (op1 not in tc[op2] and op2 not in tc[op1]) and
                        not sig_replaced[j] and not op2_has_view)
                    if independent and not merged[j] and op1.can_merge(op2):
                        merge.append(op2)
                        merged[j] = True

            if len(merge) > 0:
                merged_op, merged_sig = op1.merge(merge)
                for op in [op1] + merge:
                    op_replacements[op] = merged_op
                    for s in op.all_signals:
                        sig_replaced[op2idx[sig2op[s]]] = True
                sig_replacements.update(merged_sig)
            else:
                op_replacements[op1] = op1

        sig_replacements.update(self._get_sig_view_replacements(
            op_replacements.values(), sig_replacements))

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
