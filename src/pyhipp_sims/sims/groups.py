from __future__ import annotations
from pyhipp.io import h5
from pyhipp.core import abc
from .sim_info import SimInfo
from typing import Dict
import numpy as np

class SubhaloLoader(abc.HasLog, abc.HasCache):
    '''
    grp_id              -- offset into the group catalog at the same snapshot.
    '''
    def __init__(self, sim_info: SimInfo, snap: int, **base_kw) -> None:
        super().__init__(**base_kw)
        
        self.sim_info = sim_info
        self.snap = snap
    
        self.n_subhs: int = None
        self.n_grps: int = None
        self.root_file: h5.File = None
        self.dgrp_subh: h5.Group = None
        self.dataset_key_map: Dict[str, (str, int, float)] = {}
        
        self._preproc()
    
    def __getitem__(self, key):
        if not isinstance(key, str):
            return tuple(self[k] for k in key)
        return self.get_cache_or(key, lambda : self._load(key) )
    
    @staticmethod
    def create(sim_info: SimInfo, **kw):
        cls = {
            'tng': SubhaloLoaderTng,
            'tng_dark': SubhaloLoaderTng,
            'eagle': SubhaloLoaderEagle,
        }[sim_info.name]
        return cls(sim_info, **kw)
    
    def _preproc(self):
        raise NotImplementedError()
    
    def _load(self, key) -> np.ndarray:
        val = self.dataset_key_map[key]
        if isinstance(val, tuple):
            key_used, col_slice, scale = val
            out = self._load_impl(self.dgrp_subh, key_used, 
                               col_slice=col_slice, scale=scale)
        else:
            out = val()
        return out

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
    
class SubhaloLoaderTng(SubhaloLoader):
    '''
    For TNGDark, no m_star and sfr.
    '''
    def _preproc(self):
        info = self.sim_info
        
        file = h5.File(info.root_dir / info.root_file)
        g_base = file[f'Groups/{self.snap}']
        head = g_base['Header'].attrs
        g_subh = g_base['Subhalo']
        g_grp = g_base['Group']
        
        n_subhs, n_grps = head['Nsubgroups_Total', 'Ngroups_Total']
        key_map = {
            'grp_id': ('SubhaloGrNr', None, None),
            'm_gas': ('SubhaloMassInRadType', 0, None),
            'm_star': ('SubhaloMassInRadType', 4, None),
            'm_bh_': ('SubhaloMassInRadType', 5, None),   # including attached gas
            'm_bh': ('SubhaloBHMass', None, None),        # real BH mass
            'sfr': ('SubhaloSFRinRad', None, None),
            'x': ('SubhaloPos', None, 1.0e-3),
            'v': ('SubhaloVel', None, None),
            
            'flag': self._load_flag,  
            'is_c': self._load_is_cent,
            
            'snap_off_dm': lambda : self._load_offset('snap_off_dm'),
            'id_in_tree': lambda : self._load_offset('id_in_tree'),
            'subhalo_id': lambda : self._load_offset('subhalo_id'),
            'last_pro': lambda : self._load_offset('last_pro'),
            
            'grp_m_tophat': lambda : self._load_grp('m_tophat'),
            'grp_m_mean200': lambda : self._load_grp('m_mean200'),
            'grp_m_crit200': lambda : self._load_grp('m_crit200'),
            'grp_x': lambda : self._load_grp('x'),
        }
        
        self.n_subhs = n_subhs
        self.n_grps = n_grps
        self.root_file = file
        self.dgrp_subh = g_subh
        self.dgrp_grp = g_grp
        self.dataset_key_map = key_map
        
    def _load_flag(self):
        if self.sim_info.is_baryon:
            out = self.dgrp_subh.datasets['SubhaloFlag']
            out = np.asarray(out, dtype=bool)
        else:
            out = np.ones(self.n_subhs, dtype=bool)
        return out
    
    def _load_is_cent(self):
        cid = self.root_file[f'Groups/{self.snap}/Group'].datasets['GroupFirstSub']
        is_c = np.zeros(self.n_subhs, dtype=bool)
        cid = cid[cid >= 0]
        is_c[cid] = True
        return is_c
    
    def _load_offset(self, key):
        offcat = self.root_file[f'Offsets/{self.snap}/Subhalo']
        subl_dsets = offcat['SubLink'].datasets
        if key == 'snap_off_dm':
            out = offcat['SnapByType'][:, 1]
        elif key == 'id_in_tree':
            out = subl_dsets['RowNum']
        elif key == 'subhalo_id':
            out = subl_dsets['SubhaloID']
        elif key == 'last_pro':
            out = subl_dsets['LastProgenitorID']
        else:
            raise KeyError(key)
        return out
    
    def _load_grp(self, key):
        grp_id = self['grp_id']
        
        g = self.dgrp_grp
        if key == 'x':
            out = g['GroupPos'][()] * 1.0e-3
        elif key == 'm_tophat':
            out = g['Group_M_TopHat200'][()]
        elif key == 'm_mean200':
            out = g['Group_M_Mean200'][()]
        elif key == 'm_crit200':
            out = g['Group_M_Crit200'][()]
        else:
            raise KeyError(key)
        
        return out[grp_id]
    
class SubhaloLoaderEagleRaw(abc.HasDictRepr):
    
    def __init__(self, sim_info: SimInfo, snap: int, **base_kw) -> None:
        super().__init__(**base_kw)
        
        path = sim_info.root_dir / sim_info.root_file
        file = h5.File(path)
        
        h = sim_info.cosmology.hubble
        m_scale = 1.0e-10 * h
        x_scale_mpc = h
        x_scale_kpc = 1.0e-3 * h
        apert_list = [1,3,5,10,20,30,40,50,70,100]      # pkpc
        prop_key_map = {
            'z': ('sub', 'Redshift', (), None),
            'snap': ('sub', 'SnapNum', (), None),
            'spurious': ('sub', 'Spurious', (), None),
            
            'des': ('sub', 'DescendantID', (), None),
            'last_pro': ('sub', 'LastProgID', (), None),
            'main_leaf': ('sub', 'TopLeafID', (), None),
            'subhalo_id': ('sub', 'GalaxyID', (), None),
            'group_id': ('sub', 'GroupID', (), None),
            'group_no': ('sub', 'GroupNumber', (), None),
            'subgroup_no': ('sub', 'SubGroupNumber', (), None),
            
            'x_com': ('sub', 'CentreOfMass', ('_x', '_y', '_z'), x_scale_mpc),  # cMpc/h
            'x': ('sub', 'CentreOfPotential', ('_x', '_y', '_z'), x_scale_mpc), # cMpc//h
            'v': ('sub', 'Velocity', ('_x', '_y', '_z'), None),                 # peculiar velocity, km/s
            
            
            'mass': ('sub', 'Mass', (), m_scale),                               # 1.0e10 Msun/h
            'mass_type_bh': ('sub', 'MassType_BH', (), m_scale),
            'mass_type_dm': ('sub', 'MassType_DM', (), m_scale),
            'mass_type_gas': ('sub', 'MassType_Gas', (), m_scale),
            'mass_type_star': ('sub', 'MassType_Star', (), m_scale),
            'mass_nsf': ('sub', 'NSF_Mass', (), m_scale),
            'mass_sf': ('sub', 'SF_Mass', (), m_scale),
            
            'sfr': ('sub', 'StarFormationRate', (), None),                      # Msun/yr
            'vel_disp_star': ('sub', 'StellarVelDisp', (), None),               # km/s
            'v_max': ('sub', 'Vmax', (), None),                                 # km/s
            'r_v_max': ('sub', 'VmaxRadius', (), x_scale_kpc),                  # pMpc/h
            
            'm_bh': ('sub', 'BlackHoleMass', (), m_scale),
            'm_bh_dot': ('sub', 'BlackHoleMassAccretionRate', (), None),        # Msun/yr
            
            'spin_gas': ('sub', 'GasSpin', ('_x', '_y', '_z'), x_scale_kpc),    # pMpc/h km/s
            'spin_nsf': ('sub', 'NSF_Spin', ('_x', '_y', '_z'), x_scale_kpc),
            'spin_sf': ('sub', 'SF_Spin', ('_x', '_y', '_z'), x_scale_kpc),
            'spin_star': ('sub', 'Stars_Spin', ('_x', '_y', '_z'), x_scale_kpc),
            
            'r_half_bh': ('sub', 'HalfMassRad_BH', (), x_scale_kpc),            # pMpc/h
            'r_half_dm': ('sub', 'HalfMassRad_DM', (), x_scale_kpc),
            'r_half_gas': ('sub', 'HalfMassRad_Gas', (), x_scale_kpc),
            'r_half_star': ('sub', 'HalfMassRad_Star', (), x_scale_kpc),
            'r_half_proj_bh': ('sub', 'HalfMassProjRad_BH', (), x_scale_kpc),
            'r_half_proj_dm': ('sub', 'HalfMassProjRad_DM', (), x_scale_kpc),
            'r_half_proj_gas': ('sub', 'HalfMassProjRad_Gas', (), x_scale_kpc),
            'r_half_proj_star': ('sub', 'HalfMassProjRad_Star', (), x_scale_kpc),
            
            'z_fof': ('grp', 'Redshift', (), None),
            'snap_fof': ('grp', 'SnapNum', (), None),
            
            'id_fof': ('grp', 'GroupID', (), None),
            'first_sub_fof': ('grp', 'GroupFirstSub', (), None),                # 0-indexed, offset
            'n_sub_fof': ('grp', 'NumOfSubhalos', (), None),
            
            'x_fof': ('grp', 'GroupCentreOfPotential', ('_x', '_y', '_z'), x_scale_mpc),
            'm_fof': ('grp', 'GroupMass', (), m_scale),
            
            'm_crit_200': ('grp', 'Group_M_Crit200', (), m_scale),
            'm_crit_500': ('grp', 'Group_M_Crit500', (), m_scale),
            'm_crit_2500': ('grp', 'Group_M_Crit2500', (), m_scale),
            'm_mean_200': ('grp', 'Group_M_Mean200', (), m_scale),
            'm_mean_500': ('grp', 'Group_M_Mean500', (), m_scale),
            'm_mean_2500': ('grp', 'Group_M_Mean2500', (), m_scale),
            'm_tophat': ('grp', 'Group_M_TopHat200', (), m_scale),
            'r_crit_200': ('grp', 'Group_R_Crit200', (), x_scale_kpc),          # pMpc/h
            'r_crit_500': ('grp', 'Group_R_Crit500', (), x_scale_kpc),
            'r_crit_2500': ('grp', 'Group_R_Crit2500', (), x_scale_kpc),
            'r_mean_200': ('grp', 'Group_R_Mean200', (), x_scale_kpc),
            'r_mean_500': ('grp', 'Group_R_Mean500', (), x_scale_kpc),
            'r_mean_2500': ('grp', 'Group_R_Mean2500', (), x_scale_kpc),
            'r_tophat': ('grp', 'Group_R_TopHat200', (), x_scale_kpc),
        }
        for apert in apert_list:
            ap_key = f'ap{apert}'
            prop_key_map |= {
                f'sfr_{ap_key}': (ap_key, 'SFR', (), None),
                f'mass_{ap_key}_star': (ap_key, 'Mass_Star', (), m_scale),
                f'mass_{ap_key}_gas': (ap_key, 'Mass_Gas', (), m_scale),
                f'mass_{ap_key}_dm': (ap_key, 'Mass_DM', (), m_scale),
                f'mass_{ap_key}_bh': (ap_key, 'Mass_BH', (), m_scale),
                f'vel_disp_star_{ap_key}': (ap_key, 'VelDisp', (), None),
            }
        
        self.sim_info = sim_info
        self.snap = snap
        self.file = file
        self.apert_list = apert_list
        self.prop_key_map = prop_key_map
        
    def __getitem__(self, *keys):
        if len(keys) == 0:
            return None
        if len(keys) == 1:
            return self.__load_one(keys[0])
        return tuple(self.__load_one(k) for k in keys)
        
    def __load_one(self, key: str):
        cat, dset, comps, scale = self.prop_key_map[key]
        
        if cat == 'sub':
            g = self.subhalos
        elif cat == 'grp':
            g = self.groups
        elif cat[:2] == 'ap':
            g = self.subhalos_apertured(cat[2:])
        else:
            raise KeyError(cat)
        
        n_comps = len(comps)
        if n_comps == 0:
            val = g.datasets[dset]
        else:
            val = np.column_stack([g.datasets[dset + comp] for comp in comps])
        
        if scale is not None:
            val = val * scale
            
        return val
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kw_args):
        self.file.close()
        
    @property
    def snap_data_group(self):
        return self.file[f'Groups/{self.snap}']
        
    @property
    def subhalos(self):
        return self.snap_data_group['SubHalo']
        
    @property
    def groups(self):
        return self.snap_data_group['FOF']
    
    def subhalos_apertured(self, size = '30'):
        return self.snap_data_group[f'Aperture/{size}']
    
class SubhaloLoaderEagle(SubhaloLoader):
    def __init__(self, sim_info: SimInfo, snap: int, aperture='30', **base_kw) -> None:
        self.aperture = aperture
        super().__init__(sim_info, snap, **base_kw)
    
    def _preproc(self):
        info, snap, aperture = self.sim_info, self.snap, self.aperture
        
        file = h5.File(info.root_dir / info.root_file)
        g_snap = file[f'Groups/{snap}']
        g_subh, g_aperture, g_grp = g_snap['SubHalo', f'Aperture/{aperture}', 'FOF']
        
        n_grps, n_subhs = g_snap['Header'].attrs['NumFOFGroups', 'NumSubhalos']
        m_cvt_scale = 1.0e-10 * info.cosmology.hubble
        
        key_map = {
            'grp_id': lambda: g_subh['GroupNumber'][()] - 1,
            'm_star': lambda: g_aperture['Mass_Star'][()] * m_cvt_scale,
            'm_bh_': lambda: g_aperture['Mass_BH'][()] * m_cvt_scale,
            'm_bh': lambda: g_subh['BlackHoleMass'][()] * m_cvt_scale,
            'sfr': lambda: g_aperture['SFR'][()],
            
            'flag': lambda: (g_subh['Spurious'][()] == 0),
            'is_c': self._load_is_cent,
            
            'grp_cid': lambda: g_grp['GroupFirstSub'][()],
            'grp_n_subs': lambda: g_grp['NumOfSubhalos'][()][self['grp_id']],
            'grp_m_tophat': lambda: g_grp['Group_M_TopHat200'][()][self['grp_id']] * m_cvt_scale,
            'grp_m_mean200': lambda: g_grp['Group_M_Mean200'][()][self['grp_id']] * m_cvt_scale,
            'grp_m_crit200': lambda: g_grp['Group_M_Crit200'][()][self['grp_id']] * m_cvt_scale,
        }
        
        self.n_subhs = n_subhs
        self.n_grps = n_grps
        self.root_file = file
        self.dgrp_subh = g_subh
        self.dgrp_aperture = g_aperture
        self.dgrp_grp = g_grp
        self.dataset_key_map = key_map
        
    def _load_is_cent(self):
        grp_cid = self['grp_cid']
        grp_cid = grp_cid[grp_cid >= 0]
        
        is_c = np.zeros(self.n_subhs, dtype=bool)
        is_c[grp_cid] = True
        
        return is_c