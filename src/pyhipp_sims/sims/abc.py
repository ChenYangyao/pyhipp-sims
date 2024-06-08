from __future__ import annotations
import typing
from typing import Self, Literal, TypedDict, Unpack, Mapping
from pyhipp.core import abc, DataDict, DataTable
from .sim_info import SimInfo, predefined as predefined_sim_info
from pyhipp.stats import Rng
import numpy as np
from numpy.typing import NDArray
from pyhipp.io import h5
from pathlib import Path

_Value = int | float | bool | np.ndarray | None
_Range = tuple[_Value, _Value]
_ValueOpt = TypedDict('_ValueOpt', {
    'by': Literal['value', 'v'],
    'lo': _Value,
    'hi': _Value,
    'eq': _Value}, total=False)
_PercentileOpt = TypedDict('_PercentileOpt', {
    'by': Literal['percentile', 'p'],
    'p_lo': _Value, 'p_hi': _Value}, total=False)
_SelectOpt = dict[str, _Value | _Range | _ValueOpt | _PercentileOpt]


class HomogeneousObjectCatalog(abc.HasDictRepr, h5.abc.GroupLike):

    repr_attr_keys = ('n_objs', 'keys', 'header')
    h5_data_typedict = {
        'objs': DataTable,
        'header': DataDict
    }
    shallow_copy_attrs = ('header', )

    def __init__(self, objs: DataTable, header: DataDict, **base_kw) -> None:
        super().__init__(**base_kw)

        self.objs = objs
        self.header = header

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.objs.keys())

    @property
    def n_objs(self) -> int:
        for v in self.objs.values():
            return len(v)

    def __getitem__(self,
                    keys: str | tuple[str]
                    ) -> np.ndarray | tuple[np.ndarray]:
        return self.objs[keys]

    def select_by(self, opt_dict: _SelectOpt = {}) -> NDArray[np.bool_]:
        sel = np.ones(self.n_objs, dtype=bool)
        for key, opt in opt_dict.items():
            if isinstance(opt, tuple):
                lo, hi = opt
                sel &= self.select_by_value(key, lo=lo, hi=hi)
            elif isinstance(opt, Mapping):
                opt = {**opt}
                by = opt.pop('by')
                if by in ('value', 'v'):
                    sel &= self.select_by_value(key, **opt)
                elif by in ('percentile', 'p'):
                    sel &= self.select_by_percentile(key, **opt)
                else:
                    raise ValueError(f'Invalid option "by": {by}')
            else:
                sel &= self.select_by_value(key, eq=opt)

        return sel

    def select_by_value(self, key: str, lo=None, hi=None, eq=None
                        ) -> NDArray[np.bool_]:
        val, n_objs = self.objs[key], self.n_objs
        sel = np.ones(n_objs, dtype=bool)
        if lo is not None:
            sel &= val >= lo
        if hi is not None:
            sel &= val < hi
        if eq is not None:
            sel &= val == eq
        return sel

    def select_by_percentile(
            self, key: str, p_lo=None, p_hi=None) -> NDArray[np.bool_]:
        val = self.objs[key]
        if p_lo is None or p_lo <= 0.:
            lo = None
        else:
            lo = np.quantile(val, p_lo)

        if p_hi is None or p_hi >= 1.:
            hi = None
        else:
            hi = np.quantile(val, p_hi)

        return self.select_by_value(key, lo=lo, hi=hi)

    def shallow_copy(self, **init_kw) -> Self:
        init_kw = {
            k: getattr(self, k) for k in self.shallow_copy_attrs
        } | init_kw
        return type(self)(**init_kw)

    def subset(self, args: np.ndarray, **init_kw) -> Self:
        objs = self.objs.subset(args)
        return self.shallow_copy(objs=objs, **init_kw)

    def subset_by(self, opt_dict: _SelectOpt = {}, **init_kw) -> Self:
        '''
        Select a subset of objects by the combination of multiple criteria, 
        each applied to a column and then AND-ed.
        
        The key of `opt_dict` is the column name, and the value is one of 
        the following:
        - a tuple of two values, `(lo, hi)`.
        - a single value, `eq`.
        - a dictionary, with `by` indicating the method and the rest passed 
          as keyword arguments to the method.
          
        Examples
        --------
        
        ```py
        cat = HomogeneousObjectCatalog(...)
        cat.subset_by({
            'is_central': True,
            'm_h': (1.0e12, 1.0e13),
            'm_bh': (1.0e6, None),
            'm_s': {'by': 'p', 'p_lo': 0.1, 'p_hi': 0.9},
            'spin': {'by': 'v', 'lo': 0.1, 'hi': 0.5},
        })
        ```
        '''
        sel = self.select_by(opt_dict)
        return self.subset(sel, **init_kw)

    def subset_by_value(
            self, key: str, lo=None, hi=None, eq=None, **init_kw) -> Self:
        sel = self.select_by_value(key, lo=lo, hi=hi, eq=eq)
        return self.subset(sel, **init_kw)

    def subset_by_percentile(self, key: str, p_lo=None, p_hi=None,
                             **init_kw) -> Self:
        sel = self.select_by_percentile(key, p_lo=p_lo, p_hi=p_hi)
        return self.subset(sel, **init_kw)


class SimContext(abc.HasDictRepr, h5.abc.GroupLike):

    repr_attr_keys = ('sim_name', )

    def __init__(self, sim_info: SimInfo, rng: Rng.Initializer = 0) -> None:

        self.rng = Rng(rng)
        self.sim_info = sim_info

    @property
    def sim_name(self) -> str:
        return self.sim_info.name

    def _to_h5_data(self) -> dict:
        return super()._to_h5_data() | {
            'sim_name': self.sim_name
        }

    @classmethod
    def _from_h5_data(cls, data: dict, **init_kw) -> Self:
        sim_info = predefined_sim_info[data['sim_name'].decode()]
        return cls(sim_info, **init_kw)


class SimObjectCatalog(HomogeneousObjectCatalog):

    repr_attr_keys = HomogeneousObjectCatalog.repr_attr_keys + (
        'ctx', )
    h5_data_typedict = HomogeneousObjectCatalog.h5_data_typedict | {
        'ctx': SimContext}
    shallow_copy_attrs = HomogeneousObjectCatalog.shallow_copy_attrs + (
        'ctx',)

    def __init__(self, objs: DataTable, header: DataDict, ctx: SimContext,
                 **base_kw) -> None:
        super().__init__(objs=objs, header=header, **base_kw)

        self.ctx = ctx

    @property
    def rng(self) -> Rng:
        return self.ctx.rng

    @property
    def sim_info(self) -> SimInfo:
        return self.ctx.sim_info
