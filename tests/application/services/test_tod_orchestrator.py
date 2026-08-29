from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_simulation.application.services.tod_orchestrator import (
    TODOrchestrator,
)


class DummySimulator:
    def simulate_from_source(
        self,
        *,
        sat,
        mueller=None,
        combine_polarization="sum",
    ):
        _ = (
            mueller,
            combine_polarization,
        )

        return xr.Dataset(
            {
                "T_antenna": (
                    (
                        "time",
                        "frequency",
                    ),
                    np.full(
                        (
                            2,
                            2,
                        ),
                        sat.temperature,
                        dtype=float,
                    ),
                ),
                "power": (
                    (
                        "time",
                        "frequency",
                    ),
                    np.full(
                        (
                            2,
                            2,
                        ),
                        sat.power,
                        dtype=float,
                    ),
                ),
            },
            coords={
                "time": [
                    0.0,
                    1.0,
                ],
                "frequency": [
                    0.9e9,
                    1.0e9,
                ],
            },
        )

    def simulate_from_maps(
        self,
        *,
        maps,
        pointing_track,
        projector,
        use_beam=True,
    ):
        _ = (
            maps,
            pointing_track,
            projector,
            use_beam,
        )

        return xr.Dataset(
            {
                "T_antenna": (
                    (
                        "time",
                        "frequency",
                    ),
                    np.full(
                        (
                            2,
                            2,
                        ),
                        3.0,
                        dtype=float,
                    ),
                ),
                "power": (
                    (
                        "time",
                        "frequency",
                    ),
                    np.full(
                        (
                            2,
                            2,
                        ),
                        30.0,
                        dtype=float,
                    ),
                ),
            },
            coords={
                "time": [
                    0.0,
                    1.0,
                ],
                "frequency": [
                    0.9e9,
                    1.0e9,
                ],
            },
        )


class DummySource:
    def __init__(
        self,
        *,
        name,
        temperature,
        power,
    ):
        self.name = name
        self.temperature = temperature
        self.power = power


def _pointing():
    return xr.Dataset(
        {
            "ra": (
                "time",
                [
                    0.0,
                    0.1,
                ],
            ),
            "dec": (
                "time",
                [
                    0.0,
                    0.1,
                ],
            ),
        },
        coords={
            "time": [
                0.0,
                1.0,
            ],
        },
    )


def test_run_sources():
    orchestrator = TODOrchestrator(
        simulator=DummySimulator(),
    )

    result = orchestrator.run_sources(
        sources=[
            DummySource(
                name="A",
                temperature=1.0,
                power=10.0,
            ),
            DummySource(
                name="B",
                temperature=2.0,
                power=20.0,
            ),
        ]
    )

    assert "source" in result.dims

    assert list(result.coords["source"].values) == [
        "A",
        "B",
    ]


def test_run_maps():
    orchestrator = TODOrchestrator(
        simulator=DummySimulator(),
    )

    result = orchestrator.run_maps(
        maps=[
            object(),
        ],
        pointing_track=_pointing(),
        projector=object(),
    )

    np.testing.assert_allclose(
        result["T_antenna"].values,
        3.0,
    )


def test_run_combines_sources_and_maps():
    orchestrator = TODOrchestrator(
        simulator=DummySimulator(),
    )

    result = orchestrator.run(
        sources=[
            DummySource(
                name="A",
                temperature=1.0,
                power=10.0,
            ),
            DummySource(
                name="B",
                temperature=2.0,
                power=20.0,
            ),
        ],
        maps=[
            object(),
        ],
        pointing_track=_pointing(),
        projector=object(),
    )

    # Sources:
    #
    # 1 K + 2 K = 3 K
    #
    # Map:
    #
    # 3 K
    #
    # Total:
    #
    # 6 K
    np.testing.assert_allclose(
        result["T_antenna"].values,
        6.0,
    )

    np.testing.assert_allclose(
        result["power"].values,
        60.0,
    )

    assert result.attrs["contributions"] == [
        "source",
        "skymap",
    ]


def test_run_sources_only():
    orchestrator = TODOrchestrator(
        simulator=DummySimulator(),
    )

    result = orchestrator.run(
        sources=[
            DummySource(
                name="A",
                temperature=2.0,
                power=20.0,
            ),
        ]
    )

    assert result.attrs["contributions"] == [
        "source",
    ]


def test_run_maps_only():
    orchestrator = TODOrchestrator(
        simulator=DummySimulator(),
    )

    result = orchestrator.run(
        maps=[
            object(),
        ],
        pointing_track=_pointing(),
        projector=object(),
    )

    assert result.attrs["contributions"] == [
        "skymap",
    ]


def test_run_requires_contribution():
    orchestrator = TODOrchestrator(
        simulator=DummySimulator(),
    )

    with pytest.raises(ValueError):
        orchestrator.run()


def test_maps_require_pointing():
    orchestrator = TODOrchestrator(
        simulator=DummySimulator(),
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            maps=[
                object(),
            ],
            projector=object(),
        )


def test_maps_require_projector():
    orchestrator = TODOrchestrator(
        simulator=DummySimulator(),
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            maps=[
                object(),
            ],
            pointing_track=_pointing(),
        )


def test_combine_maps_adds_diffuse_components() -> None:
    time = np.array(
        [
            "2026-07-28T00:00:00",
            "2026-07-28T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    frequency = np.array(
        [
            1.0e9,
            1.1e9,
        ],
        dtype=float,
    )

    def make_dataset(
        temperature_k: float,
    ) -> xr.Dataset:
        temperature = xr.DataArray(
            np.full(
                (
                    len(time),
                    len(frequency),
                ),
                temperature_k,
            ),
            dims=(
                "time",
                "frequency",
            ),
            coords={
                "time": time,
                "frequency": frequency,
            },
        )

        return xr.Dataset(
            {
                "T_antenna": temperature,
                "power": (temperature * 1.380649e-23),
            }
        )

    galaxy = make_dataset(2.0)
    hi = make_dataset(0.5)

    result = TODOrchestrator.combine_maps(
        map_tods=[
            galaxy,
            hi,
        ],
    )

    np.testing.assert_allclose(
        result["T_antenna"].values,
        2.5,
    )

    np.testing.assert_allclose(
        result["power"].values,
        2.5 * 1.380649e-23,
    )

    assert result.attrs["n_map_components"] == 2


def test_combine_maps_requires_at_least_one_dataset() -> None:
    with pytest.raises(
        ValueError,
        match="at least one map TOD",
    ):
        TODOrchestrator.combine_maps(
            map_tods=[],
        )
