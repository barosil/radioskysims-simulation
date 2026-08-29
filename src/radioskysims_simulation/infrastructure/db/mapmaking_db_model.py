from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class MapMakingRecord(
    SQLModel,
    table=True,
):
    """
    Registro persistente de um mapa reconstruído a partir de TOD.

    Os arrays HEALPix não são armazenados diretamente no PostgreSQL.
    O banco mantém metadados consultáveis e a referência ao arquivo HDF5.
    """

    __tablename__ = "mapmaking_results"

    id: int | None = Field(
        default=None,
        primary_key=True,
    )

    simulation_id: int | None = Field(
        default=None,
        index=True,
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    frequency_hz: float = Field(
        index=True,
    )

    nside: int = Field(
        index=True,
    )

    npix: int

    unit: str
    frame: str
    field: str

    n_samples: int
    n_observed_pixels: int

    coverage_fraction: float

    filepath: str

    mapmaker: str = Field(
        default="healpy_binned",
        index=True,
    )
