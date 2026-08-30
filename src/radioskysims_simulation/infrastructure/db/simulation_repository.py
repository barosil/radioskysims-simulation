"""\
Módulo: src/radioskysims/infrastructure/db/simulation_repository.py

Repositório (adapter) de simulações: persiste o resumo estruturado de
uma simulação no Postgres e permite recuperar/listar/atualizar registros.

Copyright BINGO Collaboration
Last modified: 2026-08-03
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlmodel import SQLModel, desc, select

from radioskysims_simulation.application.ports.simulations import SimulationRepositoryPort
from radioskysims_simulation.infrastructure.db.simulation_db_model import SimulationRecord

logger = logging.getLogger(__name__)


def _parse_iso(value: Any) -> datetime | None:
    """Converte ISO (astropy .isot, com espaço) em datetime UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace(" ", "T"))
    except (ValueError, TypeError):
        return None


def _remove_path(path: str) -> None:
    """Remove um arquivo ou diretório persistido (Zarr é um diretório)."""
    p = Path(path)
    try:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    except OSError as exc:
        logger.warning("simulation_file_remove_failed: %s (%s)", path, exc)


def _row_from_summary(
    summary: dict[str, Any],
    *,
    filepath: str | None = None,
    saved: bool = False,
    registered: bool = True,
) -> SimulationRecord:
    """Constrói um SimulationRecord a partir do resumo estruturado."""
    obs = summary.get("observer", {})
    loc = obs.get("location", {})
    fr = obs.get("frequency_range", {})
    sim = summary.get("simulation", {})
    tr = sim.get("time_range", {})
    sky = sim.get("sky", {})
    comp = summary.get("computation", {})

    antennas = obs.get("antennas", [])
    ant_names = [a.get("name") for a in antennas if a.get("name")]
    pols: list[str] = []
    for a in antennas:
        pols.extend(a.get("polarizations") or [])

    # Agrega beam (min/max) sobre todas as antenas
    fwhm_min = fwhm_max = dbi_min = dbi_max = None

    def _agg_min(current: float | None, value: float | None) -> float | None:
        if value is None:
            return current
        return value if current is None else min(current, value)

    def _agg_max(current: float | None, value: float | None) -> float | None:
        if value is None:
            return current
        return value if current is None else max(current, value)

    for a in antennas:
        beam = a.get("beam") or {}
        fwhm = beam.get("fwhm_deg") or {}
        dbi = beam.get("directivity_dbi") or {}
        fwhm_min = _agg_min(fwhm_min, fwhm.get("min"))
        fwhm_max = _agg_max(fwhm_max, fwhm.get("max"))
        dbi_min = _agg_min(dbi_min, dbi.get("min"))
        dbi_max = _agg_max(dbi_max, dbi.get("max"))

    # Contagens de fontes
    n_sources = comp.get("dimensions", {}).get("source")
    sat_block = sky.get("satellites") or {}
    if "ids" in sat_block:
        # Formato plano: contagem direta de ids
        n_satellites = len(sat_block.get("ids", []))
    else:
        # Formato agrupado por constelação (padrão de assemble_config)
        n_satellites = sum(
            (v.get("n_satellites") or 0) for v in sat_block.values() if isinstance(v, dict)
        )
    n_local = len(sky.get("localsources", {}) or {})
    radio = sky.get("radiosources", {}) or {}
    n_radio = None
    if "n_sources" in radio:
        n_radio = radio.get("n_sources")
    elif radio:
        n_radio = None

    dims = comp.get("dimensions", {}) or {}
    if n_sources is None:
        n_sources = dims.get("source")
    if n_satellites == 0 and "satellites" in sky:
        n_satellites = None

    constellations = list((sky.get("satellites") or {}).keys()) or None

    return SimulationRecord(
        telescope_name=obs.get("name") or "unknown",
        observer_lat_deg=loc.get("lat_deg"),
        observer_lon_deg=loc.get("lon_deg"),
        observer_height_m=loc.get("height_m"),
        freq_min_mhz=fr.get("min_mhz"),
        freq_max_mhz=fr.get("max_mhz"),
        n_channels=fr.get("n_channels"),
        fc_mhz=fr.get("fc_mhz"),
        antenna_names=json.dumps(ant_names) if ant_names else None,
        polarizations=json.dumps(pols) if pols else None,
        beam_fwhm_min_deg=fwhm_min,
        beam_fwhm_max_deg=fwhm_max,
        beam_directivity_min_dbi=dbi_min,
        beam_directivity_max_dbi=dbi_max,
        time_start_utc=_parse_iso(tr.get("start")),
        time_end_utc=_parse_iso(tr.get("end")),
        time_step_s=tr.get("step_sec"),
        n_sources=n_sources,
        n_satellites=n_satellites,
        n_local_sources=n_local,
        n_radio_sources=n_radio,
        constellations=json.dumps(constellations) if constellations else None,
        sky_summary=json.dumps(sky, default=str) if sky else None,
        computation_start_utc=comp.get("start"),
        computing_time_s=comp.get("computing_time_s"),
        dataset_dimensions=json.dumps(dims) if dims else None,
        dataset_chunks=json.dumps(comp.get("chunks"), default=str) if comp.get("chunks") else None,
        dataset_size_mb=comp.get("size_mb"),
        delayed=comp.get("delayed"),
        saved=saved or bool(comp.get("saved")),
        registered=registered or bool(comp.get("registered")),
        filepath=filepath or comp.get("filepath"),
        config_json=json.dumps(summary, default=str),
    )


class SimulationRepository(SimulationRepositoryPort):
    """Adapter Postgres do repositório de simulações.

    Além de persistir/recuperar resumos no banco, mantém um *item corrente*
    (resultado carregado) em memória: ``load`` define o item corrente,
    ``flush`` o descarta (libera RAM) e ``delete(by_id)`` remove o registro.
    """

    def __init__(self, engine: Any = None, session_factory: Any = None) -> None:
        self.engine = engine
        self._session_factory = session_factory
        # Item corrente em memória (resultado carregado via ``load``)
        self._current: Any = None
        self._current_id: int | None = None

        if self.engine is not None:
            try:
                SQLModel.metadata.create_all(
                    self.engine.engine,
                    tables=[SimulationRecord.__table__],
                )
            except Exception as exc:
                logger.warning("simulation_table_create_failed: %s", exc)

    @property
    def current(self) -> Any:
        """Item corrente carregado em memória (via ``load``/``save``)."""
        return self._current

    @property
    def current_id(self) -> int | None:
        """Id do item corrente carregado em memória."""
        return self._current_id

    def _session(self):
        if self._session_factory is not None:
            return self._session_factory()
        if self.engine is not None:
            return self.engine.session()
        msg = "SimulationRepository sem engine/session_factory configurados."
        raise RuntimeError(msg)

    def save(
        self,
        result: Any,
        *,
        filepath: str | None = None,
        saved: bool = False,
        registered: bool = True,
    ) -> int:
        """Persiste o resumo da simulação e retorna o id do registro."""
        if hasattr(result, "summary"):
            summary = result.summary()
        else:
            summary = result

        row = _row_from_summary(
            summary,
            filepath=filepath,
            saved=saved,
            registered=registered,
        )
        with self._session() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
        logger.info(
            "simulation_saved",
            extra={"sim_id": row.id, "telescope": row.telescope_name},
        )
        # Atualiza o config do resultado com o que foi de fato persistido
        # (flags saved/registered e filepath refletem a lógica usada).
        if hasattr(result, "config"):
            comp = result.config.setdefault("computation", {})
            if row.filepath:
                comp["filepath"] = str(row.filepath)
            comp["saved"] = bool(row.saved)
            comp["registered"] = bool(row.registered)
        # Mantém o item corrente apontando para o registro salvo
        self._current_id = int(row.id)
        self._current = result
        return int(row.id)

    def get(self, sim_id: int) -> dict[str, Any] | None:
        """Retorna o resumo completo (dict) de uma simulação por id."""
        with self._session() as session:
            row = session.get(SimulationRecord, sim_id)
            if row is None:
                return None
            return self._row_to_dict(row)

    def list(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """Lista resumos de simulações (mais recentes primeiro)."""
        with self._session() as session:
            stmt = (
                select(SimulationRecord)
                .order_by(desc(SimulationRecord.created_at), desc(SimulationRecord.id))
                .offset(offset)
                .limit(limit)
            )
            rows = session.exec(stmt).all()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _coerce_datetime_filter(value: datetime | str | None) -> datetime | None:
        """Normaliza filtros datetime aceitando string ISO/datetime."""
        parsed = _parse_iso(value)
        if parsed is None:
            return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(UTC).replace(tzinfo=None)
        return parsed

    def search(
        self,
        *,
        created_from: datetime | str | None = None,
        created_to: datetime | str | None = None,
        time_start: datetime | str | None = None,
        time_end: datetime | str | None = None,
        telescope_name: str | None = None,
        frequency_min_mhz: float | None = None,
        frequency_max_mhz: float | None = None,
        saved: bool | None = None,
        registered: bool | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Busca registros por metadados com filtros opcionais combináveis.

        A busca opera apenas sobre colunas escalares do registro no banco e
        não carrega datasets persistidos (HDF5/Zarr).
        """
        if limit <= 0:
            msg = "limit deve ser > 0"
            raise ValueError(msg)
        if offset < 0:
            msg = "offset deve ser >= 0"
            raise ValueError(msg)

        created_from_dt = self._coerce_datetime_filter(created_from)
        created_to_dt = self._coerce_datetime_filter(created_to)
        time_start_dt = self._coerce_datetime_filter(time_start)
        time_end_dt = self._coerce_datetime_filter(time_end)

        stmt = select(SimulationRecord)

        if created_from_dt is not None:
            stmt = stmt.where(SimulationRecord.created_at >= created_from_dt)
        if created_to_dt is not None:
            stmt = stmt.where(SimulationRecord.created_at <= created_to_dt)

        # Intervalo temporal da observação com semântica de sobreposição.
        if time_start_dt is not None:
            stmt = stmt.where(SimulationRecord.time_end_utc >= time_start_dt)
        if time_end_dt is not None:
            stmt = stmt.where(SimulationRecord.time_start_utc <= time_end_dt)

        if telescope_name:
            stmt = stmt.where(SimulationRecord.telescope_name == telescope_name)

        # Faixa de frequência com semântica de sobreposição.
        if frequency_min_mhz is not None:
            stmt = stmt.where(SimulationRecord.freq_max_mhz >= float(frequency_min_mhz))
        if frequency_max_mhz is not None:
            stmt = stmt.where(SimulationRecord.freq_min_mhz <= float(frequency_max_mhz))

        if saved is not None:
            stmt = stmt.where(SimulationRecord.saved == bool(saved))
        if registered is not None:
            stmt = stmt.where(SimulationRecord.registered == bool(registered))

        stmt = (
            stmt.order_by(desc(SimulationRecord.created_at), desc(SimulationRecord.id))
            .offset(offset)
            .limit(limit)
        )

        with self._session() as session:
            rows = session.exec(stmt).all()
        return [self._row_to_dict(r) for r in rows]

    def latest(self) -> dict[str, Any] | None:
        """Retorna o resumo da simulação mais recente."""
        with self._session() as session:
            row = session.exec(
                select(SimulationRecord).order_by(desc(SimulationRecord.created_at))
            ).first()
            if row is None:
                return None
            return self._row_to_dict(row)

    def load(self, sim_id: int, *, load_dataset: bool = True) -> Any | None:
        """Carrega o resultado de uma simulação e o define como item corrente.

        Recompoe um ``SimulationOrchestratorResult`` a partir do resumo persistido
        no banco (``config_json``) e do dataset salvo em arquivo (``filepath``).
        O objeto fica disponível em ``repo.current`` até ``flush()``.

        Parameters
        ----------
        sim_id:
            Id do registro no banco.
        load_dataset:
            Se ``True`` (padrão), carrega o dataset do arquivo persistido
            (HDF5/Zarr). Se ``False``, apenas recompõe o resumo.

        Returns
        -------
        SimulationOrchestratorResult | None
            Resultado carregado, ou ``None`` se o registro não existir. Se o
            arquivo de dados estiver ausente, o ``simulation_result`` fica
            ``None`` (o resumo/``print`` continua funcionando via ``config``).
        """
        from radioskysims_simulation.domain.simulation import (
            SimulationOrchestratorResult,
            SimulationResult,
        )

        summary = self.get(sim_id)
        if summary is None:
            return None

        sim_result = None
        filepath = (summary.get("computation") or {}).get("filepath")
        if load_dataset and filepath:
            if Path(filepath).exists():
                try:
                    sim_result = SimulationResult.load(filepath)
                except Exception as exc:
                    logger.warning(
                        "simulation_load_failed",
                        extra={"sim_id": sim_id, "filepath": filepath, "error": str(exc)},
                    )
            else:
                logger.warning(
                    "simulation_file_missing",
                    extra={"sim_id": sim_id, "filepath": filepath},
                )

        config = {k: v for k, v in summary.items() if not k.startswith("_")}
        result = SimulationOrchestratorResult(
            simulation_result=sim_result,
            config=config,
            events=None,
        )
        self._current_id = int(sim_id)
        self._current = result
        return result

    def flush(self) -> None:
        """Descarta o item corrente da memória (libera RAM).

        Não remove o registro do banco nem o arquivo persistido — apenas solta
        a referência ao resultado carregado.
        """
        self._current = None
        self._current_id = None
        logger.info("simulation_flushed")

    def update_status(
        self,
        sim_id: int,
        *,
        saved: bool | None = None,
        registered: bool | None = None,
        filepath: str | None = None,
    ) -> bool:
        """Atualiza o estado de persistência de um registro."""
        with self._session() as session:
            row = session.get(SimulationRecord, sim_id)
            if row is None:
                return False
            if saved is not None:
                row.saved = saved
            if registered is not None:
                row.registered = registered
            if filepath is not None:
                row.filepath = filepath
            session.add(row)
            session.commit()
        return True

    def delete(
        self,
        sim_id: int | None = None,
        *,
        by_id: int | None = None,
        delete_file: bool = False,
    ) -> bool:
        """Remove um registro de simulação (por ``sim_id`` ou ``by_id``).

        Parameters
        ----------
        sim_id:
            Id do registro (aceito posicionalmente ou como keyword).
        by_id:
            Alternativa de nome ao parâmetro ``sim_id`` (ex.: ``delete(by_id=3)``).
        delete_file:
            Se ``True``, também remove o arquivo/diretório persistido
            (``filepath``) associado ao registro.

        Returns
        -------
        bool
            ``True`` se o registro foi removido; ``False`` se não existia.
        """
        target = sim_id if sim_id is not None else by_id
        if target is None:
            msg = "delete requer sim_id ou by_id"
            raise ValueError(msg)

        if delete_file:
            summary = self.get(target)
            if summary:
                filepath = (summary.get("computation") or {}).get("filepath")
                if filepath:
                    _remove_path(filepath)

        with self._session() as session:
            row = session.get(SimulationRecord, target)
            if row is None:
                return False
            session.delete(row)
            session.commit()

        if self._current_id == target:
            self.flush()
        logger.info("simulation_deleted", extra={"sim_id": target})
        return True

    @staticmethod
    def _row_to_dict(row: SimulationRecord) -> dict[str, Any]:
        """Converte um registro em dict (resumo completo + metadados).

        O bloco ``computation`` é enriquecido com as colunas escalares de
        persistência (``filepath``/``saved``/``registered``), que ficam apenas
        no banco — permitindo ``load``/``delete(delete_file=True)`` localizar o
        arquivo persistido.
        """
        result: dict[str, Any] = {}
        if row.config_json:
            try:
                result = json.loads(row.config_json)
            except json.JSONDecodeError:
                result = {}
        comp = result.setdefault("computation", {})
        if row.filepath:
            comp["filepath"] = row.filepath
        if row.saved is not None:
            comp["saved"] = bool(row.saved)
        if row.registered is not None:
            comp["registered"] = bool(row.registered)
        result["_id"] = row.id
        result["_created_at"] = row.created_at.isoformat() if row.created_at else None
        return result
