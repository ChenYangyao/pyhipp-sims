from __future__ import annotations
import numpy as np
from pyhipp.core import abc
from pyhipp.io import h5
from typing import Dict
from .sim_info import SimInfo

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
        self.dataset_key_map: Dict[str, (str, int, float)] = {}
        
        self._preproc()
        
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
    
    def __getitem__(self, key) -> np.ndarray | tuple[np.ndarray,...]:
        if not isinstance(key, str):
            return tuple(self[k] for k in key)
        return self.get_cache_or(key, lambda : self._load(key) )
    
    def _preproc(self):
        raise NotImplementedError()
    
    def _load(self, key) -> np.ndarray:
        key_used, col_slice, scale = self.dataset_key_map[key]
        return self._load_impl(self.dgrp_tree, key_used, 
                               col_slice=col_slice, scale=scale)

    def _load_impl(self, g: h5.Group, key, col_slice = None, scale = None):
        '''
        @key: str | tuple of keys. 
        For the later, col_slice and scale are applied to the value of each 
        key; returned values are stacked.
        '''
        if isinstance(key, str):
            d = g[key]
            val = d[()] if col_slice is None else d[:, col_slice]
            if scale is not None:
                val *= scale
            return val

        val = np.stack(
            tuple( self._load_impl(g, k, col_slice, scale) for k in key ), 
            axis=-1)
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
            'x': (('CentreOfPotential_x', 'CentreOfPotential_y','CentreOfPotential_z'), None, x_scale),
            'v': (('Velocity_x', 'Velocity_y', 'Velocity_z')),
            'm_star': ('Mass_Star_Aperture30', None, m_scale),
            'm_gas': ('Mass_Gas_Aperture30', None, m_scale),
            'm_bh': ('Mass_BH_Aperture30', None, m_scale),
            'm_dm': ('Mass_DM_Aperture30', None, m_scale),
            'sfr': ('SFR_Aperture30', None, None),
            'm_tophat': ('Group_M_TopHat200', None, m_scale),
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