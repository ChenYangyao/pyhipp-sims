from __future__ import annotations
from .sim_info import SimInfo
from pyhipp.io import h5
from pyhipp.core import abc
from typing import Tuple
import numpy as np


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

    @property
    def root_file(self) -> h5.File:
        info = self.sim_info
        path = info.root_dir / info.root_file
        return h5.File(path)

    def iter_particles(self, part_type='dm', keys=('x', )):
        raise NotImplementedError()


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
        with self.root_file as f:
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

        with self.root_file as f:
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
        with self.root_file as f:
            dgrp_ps = f[f'Snapshots/{self.snap}/{self.chunk_id}/PartType{part_type}']
            n_ps = dgrp_ps.attrs['NumPart_ThisFile']
            return n_ps

    def iter_particles(self, part_type='dm', keys=('x', ), *,
                       n_chunk=1024 * 1024 * 4, n_min=0, n_max=None):
        part_type = self.part_type_to_id['dm']
        metas = tuple(self.prop_key_to_meta[k] for k in keys)

        with self.root_file as f:
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
        with self.root_file as f:
            n_ps = f[f'Header'].attrs['NumPart_Total'][part_type]
            return n_ps

    @property
    def n_chunks(self):
        with self.root_file as f:
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
    }

    @property
    def part_mass_dm(self):
        part_type = self.part_type_to_id['dm']
        return self.sim_info.mass_table[part_type]

    def iter_particles(self, part_type: str, keys=('x', ),
                       n_chunks=1024*1024*1024, n_max=None):
        part_type = self.part_type_to_id[part_type]
        metas = tuple(self.prop_key_to_meta[k] for k in keys)

        with self.root_file as f:
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
