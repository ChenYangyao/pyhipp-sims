from __future__ import annotations
import typing
from typing import Self
from pyhipp.core import abc
from .trees import TreeLoader
import numpy as np

class Branch(abc.HasDictRepr):

    repr_attr_keys = ('root_off', 'n_hs')

    def __init__(self, tr_ld: TreeLoader,
                 root_off: float,
                 n_hs: int = -1,
                 **kw) -> None:
        super().__init__(**kw)

        if n_hs == -1:
            hids, leaf_hids = tr_ld['subhalo_id', 'main_leaf']
            n_hs = leaf_hids[root_off] - hids[root_off] + 1

        self.root_off = root_off
        self.n_hs = n_hs
        self.tr_ld = tr_ld

    def val(self, *keys):
        assert len(keys) > 0
        b, n = self.root_off, self.n_hs
        e = b + n
        if len(keys) == 1:
            key = keys[0]
            out = self.tr_ld[key][b:e][::-1].copy()
        else:
            out = tuple(self.val(key) for key in keys)
        return out

    def root_val(self, *keys):
        assert len(keys) > 0
        b = self.root_off
        if len(keys) == 1:
            key = keys[0]
            out = self.tr_ld[key][b]
        else:
            out = tuple(self.root_val(key) for key in keys)
        return out


class BranchSet(abc.HasDictRepr):

    repr_attr_keys = ('size',)

    def __init__(self, tr_ld: TreeLoader,
                 branches: list[Branch],
                 **kw) -> None:
        super().__init__(**kw)

        self.branches = branches
        self.tr_ld = tr_ld

    @classmethod
    def from_roots(cls, tr_ld: TreeLoader, root_offs) -> BranchSet:
        branches = [Branch(tr_ld, root_off) for root_off in root_offs]
        return cls(tr_ld, branches)

    @classmethod
    def from_snap(cls, tr_ld: TreeLoader, root_snap: int,
                  is_c=False,
                  m_s_lb: float = -1.0,
                  m_h_lb: float = -1.0,
                  ) -> BranchSet:
        sel = tr_ld['snap'] == root_snap
        if m_s_lb > 0:
            sel &= tr_ld['m_star'] >= m_s_lb
        if m_h_lb > 0:
            sel &= tr_ld['m_crit200'] >= m_h_lb
        if is_c is not None:
            is_cs = tr_ld['f_in_grp'] == tr_ld['subhalo_id']
            sel &= is_cs == is_c
        root_offs = sel.nonzero()[0]
        return cls.from_roots(tr_ld, root_offs)

    @property
    def size(self):
        return len(self.branches)

    def __len__(self):
        return self.size

    def __getitem__(self, ind):
        return self.branches[ind]

    def root_vals(self, *keys):
        assert len(keys) > 0
        if len(keys) == 1:
            key = keys[0]
            out = np.array([branch.root_val(key) for branch in self.branches])
        else:
            out = tuple(self.root_vals(key) for key in keys)
        return out

    def subset(self, inds):
        return BranchSet(self.tr_ld, [self.branches[ind] for ind in inds])
