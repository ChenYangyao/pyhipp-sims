from __future__ import annotations
import numpy as np
from pyhipp.core import abc
from pyhipp.io import h5
from typing import Dict, Callable
from .sim_info import SimInfo
from pyhipp.core.dataproc import Array
from pyhipp.io.binary.fortran import BinaryFile
from functools import cached_property
from dataclasses import dataclass


class TreeLoader(abc.HasLog, abc.HasCache):
    '''
    Provide data fields for the whole tree or a contiguous slice.
    
    Derived class has to define 
    _preproc()
    dgrp_tree
    dataset_key_map
    or optionally, _load()
    
    Available keys
    --------------
    subhalo_id,   f_pro,  n_pro,
    des,main_leaf,
    last_pro, f_in_grp,
    n_in_grp, snap,
    x, v,                               -- cMpc/h, physical km/s, the same for other velocities
    m_star, m_gas, m_bh, m_dm,          -- 10^10 M_sun/h, the same for other masses
    sfr,                                -- M_sun/yr
    m_tophat, r_tophat, m_mean200, ...  -- cMpc/h
    m_v_max, r_vmax,
    m_sub, v_max, v_disp, 
    spin,                               -- (Mpc/h)(km/s)
    ...
                            
    Attrs
    -----
    sim_info, root_file
    '''

    def __init__(self, sim_info: SimInfo):

        super().__init__()

        self.sim_info = sim_info

        self.root_file: h5.File = None
        self.dgrp_tree: h5.Group = None
        self.dataset_key_map: Dict[str, (str | Callable, int, float)] = {}

        self._preproc()
        self._add_mixin()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.root_file.close()

    @staticmethod
    def create(sim_info: SimInfo, **kw) -> TreeLoader:

        cls = {
            'tng': TreeLoaderTng,
            'tng_dark': TreeLoaderTngDark,
            'eagle': TreeLoaderEagle,
            'elucid_ext': TreeLoaderElucidExt,
            'elucid_ext_v2': TreeLoaderElucidExtV2,
            'simba': TreeLoaderSimba,
        }[sim_info.name]

        return cls(sim_info, **kw)

    def __getitem__(self, key) -> np.ndarray | tuple[np.ndarray, ...]:
        if not isinstance(key, str):
            return tuple(self[k] for k in key)
        return self.get_cache_or(key, lambda: self._load(key))
    
    def load_row(self, row: slice|int, *key: str):
        assert len(key) > 0, 'At least one key is required to load a row.'
        if len(key) > 1:
            return tuple(self.load_row(row, _key) for _key in key)
        key, col_slice, scale =  self.dataset_key_map[key[0]]
        return self._load_row(self.dgrp_tree, row, key, col_slice, scale)

    def _preproc(self):
        raise NotImplementedError()

    def _add_mixin(self):
        map = self.dataset_key_map

        def get_is_c():
            return self['f_in_grp'] == self['subhalo_id']
        if 'is_c' not in map:
            map['is_c'] = (get_is_c, None, None)

    def _load(self, key) -> np.ndarray:
        key_used, col_slice, scale = self.dataset_key_map[key]
        return self._load_impl(self.dgrp_tree, key_used,
                               col_slice=col_slice, scale=scale)

    def _load_impl(self, g: h5.Group, key, col_slice=None, scale=None):
        '''
        @key: str | callable | tuple of keys. 
        For the last, col_slice and scale are applied to the value of each 
        key; returned values are stacked.
        '''
        if isinstance(key, str):
            d = g[key]
            val = d[()] if col_slice is None else d[:, col_slice]
            if scale is not None:
                val *= scale
        elif isinstance(key, Callable):
            val = key()
            if col_slice is not None:
                val = val[:, col_slice]
            if scale is not None:
                val *= scale
        else:
            val = np.stack(
                tuple(self._load_impl(g, k, col_slice, scale) for k in key),
                axis=-1)
        return val
    
    def _load_row(self, g: h5.Group, row: slice|int, 
                  key: str, col_slice=None, scale=None):
        if isinstance(key, str):
            dset = g[key]
            val = dset[row] if col_slice is None else dset[row, col_slice]    
            if scale is not None:
                val *= scale
            return val

class TreeLoaderTngDark(TreeLoader):

    def _preproc(self):
        info = self.sim_info
        self.root_file = h5.File(info.root_dir / info.root_file)
        self.dgrp_tree = self.root_file['Trees/SubLink']
        self.dataset_key_map = {
            'subhalo_id':           ('SubhaloID', None, None),
            'subfind_id':           ('SubfindID', None, None),

            'f_pro':                ('FirstProgenitorID', None, None),
            'n_pro':                ('NextProgenitorID', None, None),
            'des':                  ('DescendantID', None, None),
            'main_leaf':            ('MainLeafProgenitorID', None, None),
            'last_pro':             ('LastProgenitorID', None, None),
            'f_in_grp':             ('FirstSubhaloInFOFGroupID', None, None),
            'n_in_grp':             ('NextSubhaloInFOFGroupID', None, None),
            'snap':                 ('SnapNum', None, None),
            'x':                    ('SubhaloPos', None, 1.0e-3),
            'v':                    ('SubhaloVel', None, None),
            'm':                    ('SubhaloMassInRad', None, None),
            'm_sub':                ('SubhaloMass', None, None),
            'm_dm_sub':             ('SubhaloMassType', 1, None),
            'm_dm':                 ('SubhaloMassInRadType', 1, None),
            'm_dm_v_max':           ('SubhaloMassInMaxRadType', 1, None),

            'v_max':                ('SubhaloVmax', None, None),
            'r_v_max':              ('SubhaloVmaxRad', None, 1.0e-3),
            'm_v_max':              ('SubhaloMassInMaxRad', None, None),
            'v_disp':               ('SubhaloVelDisp', None, None),
            'spin':                 ('SubhaloSpin', None, 1.0e-3),

            'm_tophat':             ('Group_M_TopHat200', None, None),
            'm_crit200':            ('Group_M_Crit200', None, None),
            'm_mean200':            ('Group_M_Mean200', None, None),
            
            'r_tophat':             ('Group_R_TopHat200', None, 1.0e-3),
            'r_crit200':            ('Group_R_Crit200', None, 1.0e-3),
            'r_mean200':            ('Group_R_Mean200', None, 1.0e-3),
        }


class TreeLoaderTng(TreeLoaderTngDark):

    def _preproc(self):
        super()._preproc()

        self.dataset_key_map |= {
            'sfr':                  ('SubhaloSFRinRad', None, None),
            'sfr_sub':              ('SubhaloSFR', None, None),

            'r_half_mass_star':     ('SubhaloHalfmassRadType', 4, 1.0e-3),
            'm_star':               ('SubhaloMassInRadType', 4, None),
            'm_gas':                ('SubhaloMassInRadType', 0, None),
            'm_bh':                 ('SubhaloMassInRadType', 5, None),
            
            'n_ps_star':             ('SubhaloLenType', 4, None),
            'n_ps_gas':             ('SubhaloLenType', 0, None),
            'n_ps_bh':             ('SubhaloLenType', 5, None),

            'm_star_sub':           ('SubhaloMassType', 4, None),
            'm_gas_sub':            ('SubhaloMassType', 0, None),
            'm_bh_sub':             ('SubhaloMassType', 5, None),

            'm_star_v_max':         ('SubhaloMassInMaxRadType', 4, None),
            'm_gas_v_max':          ('SubhaloMassInMaxRadType', 0, None),
            'm_bh_v_max':           ('SubhaloMassInMaxRadType', 5, None),
        }


class TreeLoaderEagle(TreeLoader):
    def _preproc(self):
        info = self.sim_info
        self.root_file = h5.File(info.root_dir / info.root_file)
        self.dgrp_tree = self.root_file['Trees/DTrees']

        h = info.cosmology.hubble
        m_scale = 1.0e-10 * h
        x_scale = h

        self.dataset_key_map = {
            'subhalo_id': ('GalaxyID', None, None),
            'f_pro': ('FirstProgenitorID', None, None),
            'n_pro': ('NextProgenitorID', None, None),
            'des': ('DescendantID', None, None),
            'main_leaf': ('TopLeafID', None, None),
            'last_pro': ('LastProgID', None, None),
            'f_in_grp': ('FirstSubhaloInFOFGroupID', None, None),
            'snap': ('SnapNum', None, None),
            'x': (('CentreOfPotential_x', 'CentreOfPotential_y', 'CentreOfPotential_z'), None, x_scale),
            'v': (('Velocity_x', 'Velocity_y', 'Velocity_z')),
            'm_star': ('Mass_Star_Aperture30', None, m_scale),
            'm_gas': ('Mass_Gas_Aperture30', None, m_scale),
            'm_bh': ('Mass_BH_Aperture30', None, m_scale),
            'm_dm': ('Mass_DM_Aperture30', None, m_scale),
            'sfr': ('SFR_Aperture30', None, None),
            
            'm_tophat': ('Group_M_TopHat200', None, m_scale),
            'm_crit200': ('Group_M_Crit200', None, m_scale),
            'm_mean200': ('Group_M_Mean200', None, m_scale),
            
            'm_sub': ('Mass', None, m_scale),
            'v_max': ('Vmax', None, None),
            'v_disp': ('VelDisp_Aperture30', None, None),
        }


class TreeLoaderSimba(TreeLoader):
    def _preproc(self):
        info = self.sim_info

        self.root_file = h5.File(info.root_dir / info.root_file)
        self.dgrp_tree = self.root_file['Trees/descend_galaxy_star/Galaxies']

        h = info.cosmology.hubble
        m_scale = 1.0e-10 * h
        x_scale = 1.0e-3 * h

        self.dataset_key_map = {
            'subhalo_id': ('id', None, None),
            'f_pro': ('f_pro', None, None),
            'n_pro': ('n_pro', None, None),
            'des': ('des', None, None),
            'main_leaf': ('main_leaf', None, None),
            'last_pro': ('last_pro', None, None),
            'f_in_grp': ('f_in_grp', None, None),
            'n_in_grp': ('n_in_grp', None, None),
            'snap': ('snap', None, None),
            'x': ('minpotpos', None, x_scale),
            'v': ('minpotvel', None, None),
            'm_star': ('mass_stellar', None, m_scale),
            'm_gas': ('mass_gas', None, m_scale),
            'm_bh': ('mass_bh', None, m_scale),
            'sfr': ('sfr', None, None),
            'm_tophat': ('halo_m200c', None, m_scale),
        }


class TreeLoaderElucidRaw:

    subhalo_dtype = np.dtype([
        ('des', np.int32),
        ('f_pro', np.int32),
        ('n_pro', np.int32),
        ('f_in_grp', np.int32),
        ('n_in_grp', np.int32),
        ('len', np.int32),
        ('m_mean200', np.float32),
        ('m_crit200', np.float32),
        ('m_tophat', np.float32),
        ('x', np.float32, 3),
        ('v', np.float32, 3),
        ('v_disp', np.float32),
        ('v_max', np.float32),
        ('spin', np.float32, 3),
        ('most_bound_pid', np.int64),
        ('snap', np.int32),
        ('file_id', np.int32),
        ('subfind_id', np.int32),
        ('r_sub_half_mass', np.float32),
    ], align=True)

    @dataclass
    class TreeData:
        n_trees: int
        n_subhs: int
        n_subhs_of_trees: np.ndarray
        subhs: np.ndarray
        meta: dict

    def __init__(self, chunk_id=0, sim_info=SimInfo.predefined['elucid'],
                 data_dir='raw/postproc/treedata'):
        self.chunk_id = chunk_id
        self.sim_info = sim_info
        self.data_dir = data_dir

    def __getitem__(self, key):
        if not isinstance(key, str):
            return tuple(self[k] for k in key)
        td = self._tree_data
        if key in td.meta:
            val = td.meta[key]
        else:
            val = td.subhs[key]
        return val

    @cached_property
    def _tree_data(self):
        info = self.sim_info
        snap = info.n_snapshots - 1
        c = self.chunk_id
        file_path = info.root_dir / self.data_dir / f'trees_{snap}.{c}'
        with BinaryFile(file_path) as f:
            n_trees = int(f.load_rec(np.int32))
            n_subhs = int(f.load_rec(np.int32))
            n_subhs_of_trees = f.load_rec(np.int32, n_trees)
            subhs = f.load_rec(self.subhalo_dtype, n_subhs)
            assert len(f.load_bytes(1)) == 0
        
        offset_of_trees = Array.count_to_offset(n_subhs_of_trees)
        assert n_subhs == offset_of_trees[-1]
        subhalo_id = np.zeros(n_subhs, dtype=np.int32)
        tree_id = np.zeros(n_subhs, dtype=np.int32)
        for i, (b, e) in enumerate(
            zip(offset_of_trees[: -1],
                offset_of_trees[1:])):
            subhalo_id[b:e] = np.arange(e-b)
            tree_id[b:e] = i
        is_c = subhalo_id == subhs['f_in_grp']
        meta = {
            'offset_of_trees': offset_of_trees,
            'subhalo_id': subhalo_id,
            'tree_id': tree_id,
            'is_c': is_c,
        }
            
        return self.TreeData(n_trees, n_subhs, n_subhs_of_trees, subhs, meta)


class TreeLoaderElucid(TreeLoader):
    def __init__(self, file_id: int, *, sim_info=SimInfo.predefined['elucid']):

        self.file_id = file_id

        super().__init__(sim_info)

    def _preproc(self):
        info, file_id = self.sim_info, self.file_id

        root_file = h5.File(info.root_dir / info.root_file)
        dgrp = root_file[f'Trees/SubLink/{file_id}']
        dgrp_tree = dgrp[f'Subhalos']
        n_halos_in_trees = dgrp['Header'].datasets['NumHalosInTree']
        offset_of_trees = Array.count_to_offset(n_halos_in_trees)

        n_h_tot = n_halos_in_trees.sum()
        assert n_h_tot == offset_of_trees[-1]
        subhalo_id = np.zeros(n_h_tot, dtype=np.int32)
        tree_id = np.zeros(n_h_tot, dtype=np.int32)
        for i, (b, e) in enumerate(
            zip(offset_of_trees[: -1],
                offset_of_trees[1:])):
            subhalo_id[b:e] = np.arange(e-b)
            tree_id[b:e] = i

        def get_subhalo_id():
            return subhalo_id

        def get_tree_id():
            return tree_id

        dataset_key_map = {
            'tree_id': (get_tree_id, None, None),
            'subhalo_id': (get_subhalo_id, None, None),         # = id_in_tree
            'id_in_tree': (get_subhalo_id, None, None),

            'des': ('Descendant', None, None),
            'f_pro': ('FirstProgenitor', None, None),
            'n_pro': ('NextProgenitor', None, None),
            'f_in_grp': ('FirstHaloInFOFgroup', None, None),
            'n_in_grp': ('NextHaloInFOFgroup', None, None),
            'len': ('Len', None, None),
            'm_tophat': ('M_TopHat', None, None),
            'm_crit200': ('M_Crit200', None, None),
            'm_mean200': ('M_Mean200', None, None),
            'x': ('Pos', None, None),
            'v': ('Vel', None, None),
            'v_disp': ('VelDisp', None, None),
            'v_max': ('Vmax', None, None),
            'spin': ('Spin', None, None),
            'most_bound_pid': ('MostBoundID', None, None),
            'snap': ('SnapNum', None, None),
            'file_id': ('FileNr', None, None),
            'subfind_id': ('SubhaloIndex', None, None),
            'r_sub_half_mass': ('SubhalfMass', None, None),     # pMpc/h
        }

        self.root_file = root_file
        self.dgrp_tree = dgrp_tree
        self.n_halos_in_trees = n_halos_in_trees
        self.offset_of_trees = offset_of_trees
        self.dataset_key_map = dataset_key_map


class TreeLoaderElucidExt(TreeLoader):
    def _preproc(self):

        info = self.sim_info
        self.root_file = h5.File(info.extended_root_file)
        self.dgrp_tree = self.root_file['Trees/SubLink/Subhalos']

        self.dataset_key_map = {
            'subhalo_id': ('subhalo_id', None, None),
            'id_in_tree': ('id_in_tree', None, None),
            'tree_id': ('tree_id', None, None),

            'chunk_id': ('chunk_id', None, None),
            'src_chunk_id': ('src_chunk_id', None, None),
            'src_id_in_tree': ('src_id_in_tree', None, None),
            'src_tree_id': ('src_tree_id', None, None),

            'snap': ('snap', None, None),
            'f_pro': ('f_pro', None, None),
            'n_pro': ('n_pro', None, None),
            'des': ('des', None, None),
            'main_leaf': ('main_leaf', None, None),
            'f_in_grp': ('f_in_grp', None, None),
            'last_pro': ('last_pro', None, None),
            'n_in_grp': ('n_in_grp', None, None),

            'x': ('x', None, None),
            'v': ('v', None, None),

            'm_tophat': ('m_tophat', None, None),
            'm_crit200': ('m_crit_200', None, None),
            'm_mean200': ('m_mean_200', None, None),
            'sub_half_mass': ('sub_half_mass', None, None),
            'r_tophat': ('r_tophat', None, None),
            'v_max': ('v_max', None, None),

            'most_bound_pid': ('most_bound_pid', None, None),
            'spin': ('spin', None, None),
            'vel_disp': ('vel_disp', None, None),
        }


class TreeLoaderElucidExtV2(TreeLoader):
    def _preproc(self):

        info = self.sim_info
        has_ext = info.mahgic_ext['has_position_ext']

        file_name = 'sublink.hdf5'
        if has_ext:
            file_name += '.ext'
        root_file = h5.File(info.root_dir/file_name)
        dgrp_tree = root_file['Subhalos']

        def get_m_sub():
            ind_dm = info.part_type_index.dm
            m_sub = self['len'] * info.mass_table[ind_dm]
            return m_sub

        dataset_key_map = {
            'des': ('des', None, None),
            'f_in_grp': ('f_in_grp', None, None),
            'f_pro': ('f_pro', None, None),
            'last_pro': ('last_pro', None, None),
            'len': ('len', None, None),
            'm_crit_200': ('m_crit_200', None, None),
            'm_mean_200': ('m_mean_200', None, None),

            'm_crit200': ('m_crit_200', None, None),    # alias
            'm_mean200': ('m_mean_200', None, None),    # alias

            'm_tophat': ('m_tophat', None, None),
            'main_leaf': ('main_leaf', None, None),
            'most_bound_pid': ('most_bound_pid', None, None),
            'n_in_grp': ('n_in_grp', None, None),
            'n_pro': ('n_pro', None, None),
            'r_tophat': ('r_tophat', None, None),
            'snap': ('snap', None, None),
            'spin': ('spin', None, None),
            'sub_half_mass': ('sub_half_mass', None, None),
            'subhalo_id': ('subhalo_id', None, None),
            'v': ('v', None, None),
            'v_max': ('v_max', None, None),
            'vel_disp': ('vel_disp', None, None),
            'x': ('x', None, None),
            'm_sub': (get_m_sub, None, None),
        }

        if has_ext:
            dataset_key_map |= {
                'm_sub': ('m_sub', None, None),
                'x_ext': ('x_ext', None, None),
                'v_ext': ('v_ext', None, None),
                'is_kept': ('is_kept', None, None),
                'src_flag': ('src_flag', None, None),
                'src_subhalo_id': ('src_subhalo_id', None, None),
            }
        else:
            dataset_key_map |= {
                'chunk_id': ('chunk_id', None, None),
                'tree_id': ('tree_id', None, None),
                'id_in_tree': ('id_in_tree', None, None),
                'src_chunk_id': ('src_chunk_id', None, None),
                'src_tree_id': ('src_tree_id', None, None),
                'src_id_in_tree': ('src_id_in_tree', None, None),
            }

        self.root_file = root_file
        self.dgrp_tree = dgrp_tree
        self.dataset_key_map = dataset_key_map
