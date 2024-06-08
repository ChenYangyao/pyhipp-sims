from __future__ import annotations
from pyhipp.core.abc import HasName, HasSimpleRepr, IsImmutable, HasCache
from pyhipp.astro.cosmology.model import predefined as predef_cosms, LambdaCDM
from typing import Dict, Any, Union, Literal
from collections.abc import Iterable
from pathlib import Path
from functools import cached_property
import numpy as np, pandas as pd, json, os
from scipy.interpolate import interp1d
from dataclasses import dataclass
import importlib_resources

_PredefinedNames = Literal[
    'tng', 'eagle', 'elucid',
    'elucid_w9l3ha', 'elucid_ext_v2', 'elucid_w9l3ha_ext_v2']

class _Predefined(HasSimpleRepr, HasCache):

    def __init__(self) -> None:
        super().__init__()

        key = 'PYHIPP_SIMS_DATA_DIR'
        if key in os.environ:
            data_dir = Path(os.environ[key]) / 'data' / 'simulations'
        else:
            _data_dir = importlib_resources.files(
                'pyhipp_sims').joinpath('data/simulations')
            with importlib_resources.as_file(_data_dir) as p:
                data_dir = p
        self.data_dir = data_dir

    def __getitem__(self, name: _PredefinedNames | str) -> SimInfo:
        return self.get_cache_or(name, lambda: self.load(name))

    def to_simple_repr(self):
        c: dict[str, SimInfo] = self.cache
        return {
            'data_dir': str(self.data_dir),
            'models': {
                n: info.to_simple_repr() for n, info in c.items()
            },
        }

    def load(self, name) -> SimInfo:
        return SimInfo.from_conf_file(self.data_dir / f'{name}.json')

    def set_data_dir(self, path: Path) -> None:
        self.data_dir = Path(path)

predefined = _Predefined()

class SimInfo(HasName, HasSimpleRepr, IsImmutable):
    
    predefined = predefined

    @dataclass
    class PartTypeIndex:
        gas:        int = 0
        dm:         int = 1
        tracers:    int = 3
        stars:      int = 4
        bh:         int = 5

    part_type_index = PartTypeIndex()

    def __init__(self,
                 sim_def: Dict[str, Any],
                 root_dir: Path = None,
                 root_file: Path = None,
                 formal_name: str = None,
                 suite: str = None,
                 mahgic_ext: Dict[str, Any] = None,
                 name: str = None) -> None:

        super().__init__(name=name)

        self.sim_def = sim_def
        self.root_dir = Path(root_dir)
        self.root_file = Path(root_file)
        self.formal_name = formal_name
        self.suite = suite
        self.mahgic_ext = mahgic_ext

        self.is_baryon: bool = sim_def['is_baryon']
        self.is_fullbox: bool = sim_def['is_fullbox']
        self.is_fullcover: bool = sim_def['is_fullcover']
        self.partition: Union[None, Dict] = sim_def['partition']
        self.n_snapshots: int = sim_def['n_snapshots']
        self.mass_table = np.array(sim_def['mass_table'])
        self.scale_factors = np.array(sim_def['scale_factors'])
        self.redshifts = np.array(sim_def['redshifts'])
        self.lookback_times = np.array(sim_def['lookback_times'])
        self.box_size: float = float(sim_def['box_size'])
        self.sim_def = sim_def.copy()

        for key in 'scale_factors', 'redshifts', 'lookback_times':
            assert len(getattr(self, key)) == self.n_snapshots

        self.__make_interp()

    @staticmethod
    def from_conf_file(path: Path) -> SimInfo:
        file_name = str(path)
        assert file_name[-5:] == '.json', f'{file_name} must be a JSON file'
        with open(file_name, 'rb') as f:
            conf: dict = json.load(f)

        name: str = conf.pop('name', None)
        formal_name = conf.pop('formal_name', None)
        suite = conf.pop('suite', None)
        root_dir = path.parent / conf.pop('root_dir', None)
        root_file = conf.pop('root_file', None)
        sim_def = conf.pop('simulation_definition')
        mahgic_ext = conf.pop('mahgic_extension', None)

        return SimInfo(sim_def, root_dir=root_dir, root_file=root_file,
                       formal_name=formal_name, suite=suite,
                       mahgic_ext=mahgic_ext, name=name)

    def clone(self) -> SimInfo:
        new_sim_def = self.sim_def.copy()
        p = self.partition
        if p is not None:
            new_sim_def['partition'] = p.copy()
        new_ext = self.mahgic_ext.copy()

        return SimInfo(new_sim_def, root_dir=self.root_dir,
                       root_file=self.root_file, formal_name=self.formal_name,
                       suite=self.suite, mahgic_ext=new_ext,
                       name=self.name)

    def get_sub(self, i_sub: int) -> SimInfo:
        sim_def, p = self.sim_def, self.partition
        assert p is not None
        assert p['subbox_division_policy'] == 'no_overlap'

        sub_sim_def = sim_def.copy()
        sub_sim_def['is_fullbox'] = False
        sub_sim_def['is_fullcover'] = False
        sub_sim_def['partition'] = None
        full_box_size = sim_def['box_size']
        sub_sim_def['box_size'] = full_box_size / p['n_subbox'] ** (1./3.)
        sub_sim_def['full_box_size'] = full_box_size

        name = self.name
        formal_name = self.formal_name
        suite = self.suite
        root_dir = (self.root_dir /
                    p['subboxes_dir'] /
                    (p['subbox_dir_prefix'] + str(i_sub)))
        root_file = p['subbox_root_file']
        mahgix_ext = self.mahgic_ext.copy()

        return SimInfo(
            sub_sim_def, root_dir=root_dir, root_file=root_file,
            formal_name=formal_name, suite=suite, mahgic_ext=mahgix_ext,
            name=name)

    def get_mahgic_extended(self, has_branch_ext=True,
                            has_position_ext=True) -> SimInfo:
        info = self.clone()

        if has_position_ext:
            has_branch_ext = True

        info.mahgic_ext |= {
            'has_branch_ext': has_branch_ext,
            'has_position_ext': has_position_ext,
        }

        return info

    def get_variant(self, sub_dir: str = None) -> SimInfo:
        info = self.clone()
        if sub_dir is not None:
            info.root_dir = self.root_dir / sub_dir

        return info

    @property
    def full_box_size(self):
        if self.is_fullbox:
            return self.box_size
        return self.sim_def['full_box_size']

    def to_simple_repr(self) -> dict:
        return {
            'name': self.name,
            'formal_name': self.formal_name,
            'root_dir': str(self.root_dir),
            'root_file': str(self.root_file),
            'sim_def': self.sim_def,
            'mahgic_ext': self.mahgic_ext,
        }

    @property
    def extended_root_file(self) -> Path:
        fname = str(self.root_file)

        ext = self.mahgic_ext
        if ext['has_branch_ext']:
            fname += '.bext'
            if ext['has_position_ext']:
                fname += '.pext'

        return self.root_dir / fname

    @cached_property
    def cosmology(self) -> LambdaCDM:
        return predef_cosms[self.sim_def['cosmology']]

    @cached_property
    def ages(self) -> np.ndarray:
        '''
        Cosmic ages [Gyr/h].
        '''
        return self.cosmology.age(self.redshifts)

    @cached_property
    def big_hubbles(self) -> np.ndarray:
        '''
        Hubble parameters H(z) [h/Gyr].
        '''
        return self.cosmology.big_hubble(self.redshifts)

    def z_to_snap(self, z: np.ndarray) -> np.ndarray:
        if isinstance(z, Iterable):
            return np.array([self.z_to_snap(_z) for _z in z], dtype=int)

        idx = np.argmin(np.abs(self.redshifts-z))
        return idx

    def lookback_time_to_z(self, lbt: np.ndarray) -> np.ndarray:
        return self._z_at_lbt(lbt)

    def z_to_lookback_time(self, z: np.ndarray) -> np.ndarray:
        return self._lbt_at_z(z)

    def __make_interp(self):
        zs, lbts = self.redshifts, self.lookback_times
        df = pd.DataFrame({'zs': zs, 'lbts': lbts}).drop_duplicates('zs')
        zs, lbts = df['zs'].to_numpy(), df['lbts'].to_numpy()
        assert (zs >= 0).all()
        assert (lbts >= 0).all()
        if zs[-1] > 0.:
            zs, lbts = np.concatenate((zs, (0.,))), \
                np.concatenate((lbts, (0.,)))

        kw = {'kind': 'slinear'}
        self._lbt_at_z = interp1d(zs, lbts, **kw)
        self._z_at_lbt = interp1d(lbts, zs, **kw)
