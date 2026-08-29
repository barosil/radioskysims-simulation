"""\
Módulo: src/radioskysims/infrastructure/db/simulation_db_model.py

Modelo SQLModel para registro de simulações executadas pelo
``SimOrchestrator``. Armazena o resumo completo (``config_json``) e
colunas escalares consultáveis (observer, faixa de frequência, céu e
estado da computação).

Copyright BINGO Collaboration
Last modified: 2026-08-03
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class SimulationRecord(SQLModel, table=True):
    """Registro persistente de uma simulação de TOD."""

    __tablename__ = "simulations"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Momento do registro (UTC).",
    )

    # ── Observer ─────────────────────────────────────────────
    telescope_name: str = Field(default="", index=True)
    observer_lat_deg: float | None = None
    observer_lon_deg: float | None = None
    observer_height_m: float | None = None
    freq_min_mhz: float | None = None
    freq_max_mhz: float | None = None
    n_channels: int | None = None
    fc_mhz: float | None = None
    antenna_names: str | None = Field(default=None, description="Nomes das antenas (JSON list).")
    polarizations: str | None = Field(default=None, description="Polarizações (JSON list).")
    beam_fwhm_min_deg: float | None = None
    beam_fwhm_max_deg: float | None = None
    beam_directivity_min_dbi: float | None = None
    beam_directivity_max_dbi: float | None = None

    # ── Simulation ───────────────────────────────────────────
    time_start_utc: datetime | None = None
    time_end_utc: datetime | None = None
    time_step_s: float | None = None
    n_sources: int | None = None
    n_satellites: int | None = None
    n_local_sources: int | None = None
    n_radio_sources: int | None = None
    constellations: str | None = Field(
        default=None, description="Constelações simuladas (JSON list)."
    )
    sky_summary: str | None = Field(
        default=None, description="Bloco sky (JSON): sol, rádio, satélites."
    )

    # ── Computation ──────────────────────────────────────────
    computation_start_utc: str | None = None
    computing_time_s: float | None = None
    dataset_dimensions: str | None = Field(
        default=None, description="Dimensões do dataset (JSON dict)."
    )
    dataset_chunks: str | None = Field(default=None, description="Chunks do dataset (JSON).")
    dataset_size_mb: float | None = None
    delayed: bool | None = Field(default=True, description="Dataset lazy (dask)?")
    saved: bool = Field(default=False, description="Dataset persistido em arquivo?")
    registered: bool = Field(default=False, description="Registro completo no repositório?")
    filepath: str | None = Field(default=None, description="Caminho do arquivo salvo.")

    # ── Resumo completo (JSON) ───────────────────────────────
    config_json: str = Field(default="{}", description="Resumo completo (JSON).")
