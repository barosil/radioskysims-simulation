"""\
Módulo: tests/application/orchestrators/test_mapmaking_orchestrator.py

Descrição:
    Testes da fachada ``MapMakingOrchestrator`` (UX Onda 3/R6) com um
    serviço fake — sem mapmaker real.

Copyright BINGO Collaboration
"""

from __future__ import annotations

from radioskysims_simulation import MapMakingOrchestrator
from radioskysims_simulation.application.orchestrators.mapmaking_orchestrator import (
    MapMakingOrchestrator as _Orch,
)


class FakeService:
    """Fake de MapMakingService: registra chamadas e devolve marcador."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def run(self, **kwargs: object) -> str:
        self.kwargs = dict(kwargs)
        return "mapmaking-ok"


def test_run_delega_para_service():
    fake = FakeService()
    orch = MapMakingOrchestrator(service=fake)  # type: ignore[arg-type]
    result = orch.run(
        tod="tod",  # type: ignore[arg-type]
        pointing_track="track",  # type: ignore[arg-type]
        nside=256,
    )
    assert result == "mapmaking-ok"
    assert fake.kwargs is not None
    assert fake.kwargs["tod"] == "tod"
    assert fake.kwargs["pointing_track"] == "track"
    assert fake.kwargs["nside"] == 256
    assert fake.kwargs["frame"] == "equatorial"


def test_run_aceita_frame_e_persist():
    fake = FakeService()
    _Orch.from_service(fake).run(  # type: ignore[arg-type]
        tod="t",
        pointing_track="p",
        nside=64,
        frame="galactic",
        persist=True,
        simulation_id=7,
    )
    assert fake.kwargs is not None
    assert fake.kwargs["frame"] == "galactic"
    assert fake.kwargs["persist"] is True
    assert fake.kwargs["simulation_id"] == 7


def test_repr():
    fake = FakeService()
    orch = _Orch(fake)  # type: ignore[arg-type]
    assert "MapMakingOrchestrator" in repr(orch)


def test_default_monta_com_mapmaker_binned():
    orch = MapMakingOrchestrator.default()
    assert isinstance(orch, MapMakingOrchestrator)
    assert "MapMakingOrchestrator" in repr(orch)
