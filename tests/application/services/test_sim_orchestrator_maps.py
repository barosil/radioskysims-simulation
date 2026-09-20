from __future__ import annotations

from types import SimpleNamespace

import pytest
import xarray as xr

from radioskysims_simulation.application.services.sim_orchestrator import (
    SimOrchestrator,
)


class DummyTODOrchestrator:
    def __init__(self) -> None:
        self.called_with = None

    def run_maps(
        self,
        *,
        maps,
        pointing_track,
        projector,
        use_beam=True,
    ):
        self.called_with = {
            "maps": maps,
            "pointing_track": pointing_track,
            "projector": projector,
            "use_beam": use_beam,
        }

        return xr.Dataset(
            {
                "T_antenna": (
                    ("time", "frequency"),
                    [[1.0]],
                )
            },
            coords={
                "time": [0.0],
                "frequency": [1.0e9],
            },
        )


def _make_orchestrator(
    *,
    tod_orchestrator=None,
    projector=None,
) -> SimOrchestrator:
    context = SimpleNamespace(
        local_sources_service=None,
        radio_source_catalog_service=None,
        tod_orchestrator=tod_orchestrator,
        galaxy_map_generator=None,
        skymap_repository=None,
        skymap_tod_projector=projector,
        config_loader=None,
        logger=None,
        neutral_hydrogen_map_generator=None,
        cosmology_parameters=None,
    )

    return SimOrchestrator(context)  # type: ignore[arg-type]


def _pointing() -> xr.Dataset:
    return xr.Dataset(
        {
            "ra": ("time", [0.0]),
            "dec": ("time", [0.0]),
        },
        coords={"time": [0.0]},
    )


def test_run_maps_delegates_to_tod_orchestrator():
    tod_orchestrator = DummyTODOrchestrator()
    projector = object()

    orchestrator = _make_orchestrator(
        tod_orchestrator=tod_orchestrator,
        projector=projector,
    )

    maps = [object()]
    pointing = _pointing()

    result = orchestrator.run_maps(
        maps=maps,
        pointing_track=pointing,
        use_beam=False,
    )

    assert isinstance(result, xr.Dataset)
    assert tod_orchestrator.called_with is not None
    assert tod_orchestrator.called_with["maps"] == maps
    assert tod_orchestrator.called_with["pointing_track"] is pointing
    assert tod_orchestrator.called_with["projector"] is projector
    assert tod_orchestrator.called_with["use_beam"] is False


def test_run_maps_requires_tod_orchestrator():
    orchestrator = _make_orchestrator(
        tod_orchestrator=None,
        projector=object(),
    )

    with pytest.raises(
        RuntimeError,
        match="TODOrchestrator is not configured",
    ):
        orchestrator.run_maps(
            maps=[object()],
            pointing_track=_pointing(),
        )


def test_run_maps_requires_projector():
    orchestrator = _make_orchestrator(
        tod_orchestrator=DummyTODOrchestrator(),
        projector=None,
    )

    with pytest.raises(
        RuntimeError,
        match="SkyMap TOD projector is not configured",
    ):
        orchestrator.run_maps(
            maps=[object()],
            pointing_track=_pointing(),
        )
