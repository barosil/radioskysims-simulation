from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sqlmodel import SQLModel

from radioskysims_simulation.application.ports.mapmaking_repository import (
    MapMakingRepositoryPort,
)
from radioskysims_simulation.domain.mapmaking import MapMakingResult
from radioskysims_skymap.domain.skymap import SkyMap
from radioskysims_simulation.infrastructure.db.mapmaking_db_model import (
    MapMakingRecord,
)


class MapMakingRepository(MapMakingRepositoryPort):
    """
    Repository híbrido para resultados de mapmaking.

    PostgreSQL:
        metadados consultáveis.

    HDF5:
        pixels HEALPix e hit map.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        engine: Any = None,
        session_factory: Any = None,
    ) -> None:
        self._root = Path(root)

        self._root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._engine = engine
        self._session_factory = session_factory

        if self._engine is not None:
            SQLModel.metadata.create_all(
                self._engine.engine,
                tables=[
                    MapMakingRecord.__table__,
                ],
            )

    @property
    def root(self) -> Path:
        return self._root

    def _session(self):
        if self._session_factory is not None:
            return self._session_factory()

        if self._engine is not None:
            return self._engine.session()

        raise RuntimeError("MapMakingRepository requires engine or session_factory.")

    @staticmethod
    def _validate_key(
        key: str,
    ) -> str:
        if not isinstance(key, str):
            raise TypeError("MapMaking repository key must be a string.")

        normalized = key.strip()

        if not normalized:
            raise ValueError("MapMaking repository key cannot be empty.")

        if "/" in normalized or "\\" in normalized or ".." in normalized:
            raise ValueError("MapMaking repository key must not contain path separators or '..'.")

        return normalized

    def _path_for_key(
        self,
        key: str,
    ) -> Path:
        normalized = self._validate_key(key)

        return self._root / f"{normalized}.h5"

    def save(
        self,
        *,
        result: MapMakingResult,
        key: str,
        simulation_id: int | None = None,
        overwrite: bool = False,
    ) -> int:
        if not isinstance(
            result,
            MapMakingResult,
        ):
            raise TypeError("result must be a MapMakingResult.")

        path = self._path_for_key(key)

        if path.exists() and not overwrite:
            raise FileExistsError(f"MapMaking result already exists: {path}")

        temporary_path = path.with_suffix(".h5.tmp")

        try:
            with h5py.File(
                temporary_path,
                "w",
            ) as handle:
                handle.create_dataset(
                    "pixels",
                    data=result.skymap.data,
                    compression="gzip",
                    shuffle=True,
                )

                handle.create_dataset(
                    "hit_map",
                    data=result.hit_map,
                    compression="gzip",
                    shuffle=True,
                )

                handle.attrs["nside"] = result.skymap.nside_value

                handle.attrs["unit"] = result.skymap.unit

                handle.attrs["frame"] = result.skymap.frame

                handle.attrs["field"] = result.skymap.field

                if result.skymap.freq_ghz is None:
                    handle.attrs["has_freq_ghz"] = False
                else:
                    handle.attrs["has_freq_ghz"] = True
                    handle.attrs["freq_ghz"] = result.skymap.freq_ghz

                handle.attrs["n_samples"] = result.n_samples

            temporary_path.replace(path)

        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()

            raise

        frequency_hz = (
            float(result.skymap.freq_ghz) * 1.0e9 if result.skymap.freq_ghz is not None else 0.0
        )

        row = MapMakingRecord(
            simulation_id=simulation_id,
            frequency_hz=frequency_hz,
            nside=result.skymap.nside_value,
            npix=result.skymap.npix,
            unit=result.skymap.unit,
            frame=result.skymap.frame,
            field=result.skymap.field,
            n_samples=result.n_samples,
            n_observed_pixels=(result.n_observed_pixels),
            coverage_fraction=(result.coverage_fraction),
            filepath=str(path),
        )

        with self._session() as session:
            session.add(row)
            session.commit()
            session.refresh(row)

        if row.id is None:
            raise RuntimeError("MapMakingRecord was not assigned an id.")

        return int(row.id)

    def load(
        self,
        record_id: int,
    ) -> MapMakingResult | None:
        with self._session() as session:
            row = session.get(
                MapMakingRecord,
                record_id,
            )

            if row is None:
                return None

        path = Path(row.filepath)

        if not path.exists():
            raise FileNotFoundError(f"MapMaking HDF5 file not found: {path}")

        with h5py.File(
            path,
            "r",
        ) as handle:
            pixels = np.asarray(
                handle["pixels"][...],
                dtype=np.float64,
            )

            hit_map = np.asarray(
                handle["hit_map"][...],
                dtype=np.int64,
            )

        skymap = SkyMap(
            dataset=pixels,
            nside=row.nside,
            unit=row.unit,
            frame=row.frame,
            freq_ghz=(row.frequency_hz / 1.0e9),
            field=row.field,
        )

        return MapMakingResult(
            skymap=skymap,
            hit_map=hit_map,
            observed_mask=(hit_map > 0),
            n_samples=row.n_samples,
        )

    def get_metadata(
        self,
        record_id: int,
    ) -> dict[str, object] | None:
        with self._session() as session:
            row = session.get(
                MapMakingRecord,
                record_id,
            )

            if row is None:
                return None

            return {
                "id": row.id,
                "simulation_id": row.simulation_id,
                "frequency_hz": row.frequency_hz,
                "nside": row.nside,
                "npix": row.npix,
                "unit": row.unit,
                "frame": row.frame,
                "field": row.field,
                "n_samples": row.n_samples,
                "n_observed_pixels": (row.n_observed_pixels),
                "coverage_fraction": (row.coverage_fraction),
                "filepath": row.filepath,
                "mapmaker": row.mapmaker,
            }

    def delete(
        self,
        record_id: int,
    ) -> bool:
        with self._session() as session:
            row = session.get(
                MapMakingRecord,
                record_id,
            )

            if row is None:
                return False

            path = Path(row.filepath)

            session.delete(row)
            session.commit()

        if path.exists():
            path.unlink()

        return True
