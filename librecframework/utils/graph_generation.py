#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, cast
import numpy as np
import scipy.sparse as sp
import torch
from .convert import scisp_to_torch

__all__ = ['complete_graph_from_pq']


def complete_graph_from_pq(
        pq_graph,
        pp_graph,
        qq_graph,
        return_sparse: bool,
        normalize: str,
        dtype=None,
        return_scipy: bool = False,
        eps: float = 1e-10) -> Union[torch.Tensor, sp.coo_matrix, np.ndarray]:
    """
    Create adjacency matrix as follows:

    ┌──────────┬──────────┐

    │ pp_graph │ pq_graph │

    ├──────────┼──────────┤

    │pq_graph.T│ qq_graph │

    └──────────┴──────────┘

    Args:
    - return_sparse: return sparse matrix (True) or dense matrix (False)
    - normalize: the method to normalize graph (Options: `none`, `out`, `in`, `laplace`)
    - dtype: target dtype of return value
    - return_scipy: return numpy/scipy sparse matrix (True) or pytorch tensor (False)
    """
    normalize = normalize.lower()
    assert normalize in ['none', 'out', 'in', 'laplace']
    pq_graph = pq_graph.tocoo()
    pp_graph = pp_graph.tocoo()
    qq_graph = qq_graph.tocoo()
    p_pq_graph = sp.hstack([pp_graph, pq_graph], format='coo')
    q_pq_graph = sp.hstack([pq_graph.T, qq_graph], format='coo')
    graph = sp.vstack([p_pq_graph, q_pq_graph], format='coo')
    if normalize == 'none':
        pass
    elif normalize == 'in':
        graph = graph.multiply(1 / (graph.sum(1) + eps))
    elif normalize == 'out':
        graph = graph.multiply(1 / (graph.sum(0) + eps))
    elif normalize == 'laplace':
        graph = graph.multiply(
            1 / (np.sqrt(graph.sum(0)) + eps)).multiply(1 / (np.sqrt(graph.sum(1)) + eps))
    else:
        raise ValueError(f'Improper `normalize` {normalize}')
    if not dtype is None:
        graph = graph.astype(dtype, copy=False)
    graph = cast(sp.coo_matrix, graph)
    if return_scipy:
        if return_sparse:
            return graph
        else:
            return graph.toarray()
    else:
        if return_sparse:
            graph = scisp_to_torch(graph)
        else:
            graph = torch.from_numpy(graph.toarray())
        return graph
