from __future__ import annotations
from typing import Any
from .sim_info import SimInfo
from pyhipp.io import h5
from pyhipp.io.binary.fortran import BinaryFile
from pyhipp.core import abc
from typing import Tuple
import numpy as np
from functools import cached_property


class SnapshotLoader(abc.HasLog):

    part_types: Tuple[str, ...] = ()

    def __init__(self, sim_info: SimInfo, z: float = None,
                 snap: int = None, **kw) -> None:
        super().__init__(**kw)

        if z is None:
            assert snap is not None
        else:
            assert snap is None
            snap = sim_info.z_to_snap(z)

        self.sim_info = sim_info
        self.snap = snap
        self._root_file: h5.File = None

    @property
    def root_file(self) -> h5.File:
        if self._root_file is None:
            self._root_file = self._open_root_file()
        return self._root_file
    
    def _open_root_file(self):
        info = self.sim_info
        path = info.root_dir / info.root_file
        return h5.File(path)

    def iter_particles(self, part_type='dm', keys=('x', )):
        raise NotImplementedError()
    
    def load_row(self, part_type: str, row: slice|int, *key: str):
        assert len(key) > 0
        if len(key) > 1:
            return tuple(self.load_row(part_type, row, _key) for _key in key)
        return self._load_row(part_type, row, key[0])
    
    def _load_row(self, part_type: str, row: slice|int, key: str) -> np.ndarray | Any:
        raise NotImplementedError()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._root_file is not None:
            self._root_file.close()
            self._root_file = None
        

class SnapshotLoaderDmo(SnapshotLoader):

    part_types = ('dm', )

    @property
    def part_mass_dm(self):
        raise NotImplementedError()

    @property
    def n_part_dm(self):
        raise NotImplementedError()

    @staticmethod
    def create(sim_type, **kw):
        return {
            'tng_dark': SnapshotLoaderTngDark,
        }[sim_type](**kw)


class SnapshotLoaderTngDark(SnapshotLoaderDmo):

    part_type_to_id = {
        'dm': 1,
    }

    # dataset name and scale to standard units
    prop_key_to_meta = {
        'x': ('Coordinates', 1.0e-3),
        'v': ('Velocities', None),
        'potential': ('Potential', None),
        'id': ('ParticleIDs', None),
    }

    def __init__(self, sim_info: SimInfo, z: float = None,
                 snap: int = None, **kw) -> None:
        super().__init__(sim_info, z, snap, **kw)

    @property
    def part_mass_dm(self):
        part_type = self.part_type_to_id['dm']
        return self.sim_info.mass_table[part_type]

    @property
    def n_part_dm(self):
        part_type = self.part_type_to_id['dm']
        f = self.root_file
        dgrp_snap = f[f'Snapshots/{self.snap}']
        n_ps = dgrp_snap['Header'].attrs['NumPart_Total'][part_type]
        return n_ps

    def iter_particles(self, part_type='dm', keys=('x', ),
                       *, n_chunk=1024*1024*4, n_min=0, n_max=None):
        '''
        @chunk_size: None for load all at once, int for the number of particles
        loaded at every iteration.
        '''
        part_type = self.part_type_to_id[part_type]
        metas = tuple(self.prop_key_to_meta[k] for k in keys)

        f = self.root_file
        dgrp_snap = f[f'Snapshots/{self.snap}']
        dgrp_ps = dgrp_snap[f'PartType{part_type}']
        n_ps = dgrp_snap['Header'].attrs['NumPart_Total'][part_type]
        dsets = tuple(dgrp_ps[k] for k, _ in metas)

        if n_max is None:
            n_max = n_ps
        if n_chunk is None:
            n_chunk = n_max - n_min

        for b in range(n_min, n_max, n_chunk):
            e = min(b + n_chunk, n_max)
            yield tuple(dset[b:e] if scale is None else dset[b:e]*scale
                        for dset, (_, scale) in zip(dsets, metas))


class SnapshotLoaderElucidRaw(SnapshotLoaderDmo):
    '''
    Load data in raw binary format.
    Instance is immutable.
    '''

    header_dtype = np.dtype([
        ('NumPart_ThisFile', np.uint32, 6),
        ('MassTable', np.float64, 6),
        ('Time', np.float64),
        ('Redshift', np.float64),
        ('FlagSFR', np.int32),
        ('FlagFeedback', np.int32),
        ('NumPart_Total', np.uint32, 6),
        ('FlagCooling', np.int32),
        ('NumFilesPerSnapshot', np.int32),
        ('BoxSize', np.float64),
        ('Omega0', np.float64),
        ('OmegaLambda', np.float64),
        ('HubbleParam', np.float64),
        ('FlagStellarAge', np.int32),
        ('FlagMetals', np.int32),
        ('HashTableSize', np.int32),
        ('fill', 'S84'),
    ], align=True)
    fields = [
        ('x', np.float32, (3,)),
        ('v', np.float32, (3,)),
        ('id', np.int64, ()),
    ]
    n_files = 2048
    p_type_dm = 1

    def __init__(self, sim_info: SimInfo, z: float = None, snap: int = None,
                 chunk_id=0, **kw) -> None:
        super().__init__(sim_info, z, snap, **kw)

        assert 0 <= chunk_id < self.n_files
        assert self.header_dtype.itemsize == 256
        self.chunk_id = chunk_id

    @property
    def root_file_name(self):
        snap, c = self.snap, self.chunk_id
        return self.sim_info.root_dir / f'raw/snapdir_{snap:03d}/snapshot_{snap:03d}.{c}'

    def _open_root_file(self):
        return BinaryFile(self.root_file_name, 'rb')

    @cached_property
    def header(self):
        f = self.root_file
        header = f.load_sect(self.header_dtype)
        return header

    @property
    def part_mass_dm(self):
        return float(self.header['MassTable'][self.p_type_dm])

    @property
    def n_part_dm(self):
        return int(self.header['NumPart_ThisFile'][self.p_type_dm])

    def load_particles(self, field='x'):
        n_p = self.n_part_dm
        f = self.root_file
        f.load_sect(self.header_dtype)
        val = None
        for _field, _dtype, _shape in self.fields:
            _dt = np.dtype(_dtype)
            full_shape = (n_p,) + _shape
            n = np.prod(full_shape)
            if _field != field:
                f.skip_sect(_dt, n=n)
            else:
                val = f.load_sect(_dt, n=n).reshape(full_shape)
        assert val is not None, f'Field {field} not found.'
        return val
    
    @cached_property
    def hash_table(self):
        n_p = self.n_part_dm
        f = self.root_file
        f.skip_sect(self.header_dtype)
        for field, dtype, shape in self.fields:
            _dtype = np.dtype(dtype)
            full_shape = (n_p,) + shape
            n = np.prod(full_shape)
            f.skip_sect(_dtype, n=n)
        bcell, ecell = f.load_sect(np.int32, 2)
        n_cells = ecell - bcell + 1
        p_offs = f.load_sect(np.int32, n_cells)
        return {
            'FirstCellID': bcell,
            'LastCellID': ecell,
            'ParticleOffsetsInCell': p_offs,
        }

class _SnapshotLoaderElucidChunk(SnapshotLoaderDmo):
    part_type_to_id = {
        'dm': 1,
    }
    prop_key_to_meta = {
        'x': ('Coordinates', None),
        'v': ('Velocities', None),
        'id': ('ParticleIDs', None),
    }

    def __init__(self, sim_info: SimInfo, z: float = None, snap: int = None,
                 chunk_id=0, **kw) -> None:
        super().__init__(sim_info, z, snap, **kw)

        self.chunk_id = chunk_id

    @property
    def part_mass_dm(self):
        part_type = self.part_type_to_id['dm']
        return self.sim_info.mass_table[part_type]

    @property
    def n_part_dm(self):
        part_type = self.part_type_to_id['dm']
        f = self.root_file
        dgrp_ps = f[f'Snapshots/{self.snap}/{self.chunk_id}/PartType{part_type}']
        n_ps = dgrp_ps.attrs['NumPart_ThisFile']
        return n_ps

    def iter_particles(self, part_type='dm', keys=('x', ), *,
                       n_chunk=1024 * 1024 * 4, n_min=0, n_max=None):
        part_type = self.part_type_to_id['dm']
        metas = tuple(self.prop_key_to_meta[k] for k in keys)

        f = self.root_file
        dgrp_ps = f[f'Snapshots/{self.snap}/{self.chunk_id}/PartType{part_type}']
        n_ps = dgrp_ps.attrs['NumPart_ThisFile']
        dsets = tuple(dgrp_ps[k] for k, _ in metas)

        if n_max is None:
            n_max = n_ps
        if n_chunk is None:
            n_chunk = n_max - n_min

        for b in range(n_min, n_max, n_chunk):
            e = min(b + n_chunk, n_max)
            yield tuple(dset[b:e] if scale is None else dset[b:e]*scale
                        for dset, (_, scale) in zip(dsets, metas))


class SnapshotLoaderElucid(SnapshotLoaderDmo):
    def __init__(self, sim_info: SimInfo, z: float = None,
                 snap: int = None, **kw) -> None:
        super().__init__(sim_info, z, snap, **kw)

    @property
    def part_mass_dm(self):
        part_type = _SnapshotLoaderElucidChunk.part_type_to_id['dm']
        return self.sim_info.mass_table[part_type]

    @property
    def n_part_dm(self):
        part_type = _SnapshotLoaderElucidChunk.part_type_to_id['dm']
        f = self.root_file
        n_ps = f[f'Header'].attrs['NumPart_Total'][part_type]
        return n_ps

    @property
    def n_chunks(self):
        f = self.root_file
        n_c = f[f'Header'].attrs['NumFilesPerSnapshot']
        return n_c

    def chunk_loader(self, chunk_id: int):
        return _SnapshotLoaderElucidChunk(
            sim_info=self.sim_info, snap=self.snap, chunk_id=chunk_id)

    def iter_particles(self, part_type='dm', keys=('x', ), *,
                       chunks: list[int] = None, chunk_loader_kw={}):
        if chunks is None:
            chunks = np.arange(self.n_chunks)
        for c in chunks:
            yield from self.chunk_loader(c)\
                .iter_particles(part_type, keys, **chunk_loader_kw)


class SnapshotLoaderBaryonic(SnapshotLoader):

    part_types = ('gas', 'dm', 'star', 'bh')

    @property
    def part_mass_gas(self):
        raise NotImplementedError()

    @property
    def part_mass_dm(self):
        raise NotImplementedError()

    @property
    def part_mass_star(self):
        raise NotImplementedError()

    @property
    def part_mass_bh(self):
        raise NotImplementedError()

    @staticmethod
    def create(sim_type, **kw):
        return {
            'tng': SnapshotLoaderTng,
        }[sim_type](**kw)


class SnapshotLoaderTng(SnapshotLoaderBaryonic):

    part_type_to_id = {
        'gas': 0,
        'dm': 1,
        'star': 4,
        'bh': 5,
    }

    prop_key_to_meta = {
        'x': ('Coordinates', 1.0e-3),
        'v': ('Velocities', None),
        'potential': ('Potential', None),
        'id': ('ParticleIDs', None),
        'mass': ('Masses', None),
        'birth_x': ('BirthPos', 1.0e-3),
        'birth_v': ('BirthVel', None),
        'birth_a': ('GFM_StellarFormationTime', None),
        'birth_mass': ('GFM_InitialMass', None)
    }

    @property
    def part_mass_dm(self):
        part_type = self.part_type_to_id['dm']
        return self.sim_info.mass_table[part_type]

    def iter_particles(self, part_type: str, keys=('x', ),
                       n_chunks=1024*1024*1024, n_max=None):
        part_type = self.part_type_to_id[part_type]
        metas = tuple(self.prop_key_to_meta[k] for k in keys)

        f = self.root_file
        dgrp_snap = f[f'Snapshots/{self.snap}']
        dgrp_ps = dgrp_snap[f'PartType{part_type}']
        n_ps = dgrp_snap['Header'].attrs['NumPart_Total'][part_type]
        dsets = tuple(dgrp_ps[k] for k, _ in metas)

        if n_max is None:
            n_max = n_ps
        if n_chunks is None:
            n_chunks = n_max

        for b in range(0, n_max, n_chunks):
            e = min(b + n_chunks, n_max)
            yield tuple(dset[b:e] if scale is None else dset[b:e]*scale
                        for dset, (_, scale) in zip(dsets, metas))
            
    @cached_property
    def dgrp_snap(self):
        return self.root_file[f'Snapshots/{self.snap}']

    @cached_property
    def dgrp_part_types(self):
        return {k: self.dgrp_snap[f'PartType{id}'] 
                for k, id in self.part_type_to_id.items()}
    
    def _load_row(self, part_type: str, row: slice|int, key: str) -> np.ndarray | Any:
        dgrp_ps = self.dgrp_part_types[part_type]
        key, scale = self.prop_key_to_meta[key]
        dset = dgrp_ps[key]
        val = dset[row]
        if scale is not None:
            val *= scale
        return val