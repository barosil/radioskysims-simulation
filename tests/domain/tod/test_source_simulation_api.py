from __future__ import annotations

import numpy as np
import xarray as xr

from radioskysims_simulation.domain.tod.simulator import TODSimulator


class DummySource:
    def __init__(self):
        self._frequency = xr.DataArray(
            np.array(
                [
                    1.0e9,
                ]
            ),
            dims="frequency",
            coords={
                "frequency": np.array(
                    [
                        1.0e9,
                    ]
                )
            },
        )

        self._angle = xr.DataArray(
            np.array(
                [
                    0.0,
                    0.1,
                ]
            ),
            dims="time",
            coords={
                "time": np.array(
                    [
                        0.0,
                        1.0,
                    ]
                )
            },
        )

        self._range = xr.DataArray(
            np.full(
                2,
                1.0e6,
            ),
            dims="time",
            coords={
                "time": self._angle.coords["time"],
            },
        )

    def psd(self):
        result = xr.DataArray(
            np.ones(
                1,
                dtype=float,
            ),
            dims="frequency",
            coords={
                "frequency": (self._frequency.values),
            },
        )

        result.attrs["units"] = "W/Hz"

        return result

    def angle(self):
        return self._angle

    def range(self):
        return self._range


class DummyBeam:
    def normalized(
        self,
        kind="integral",
    ):
        return self

    def evaluate(
        self,
        *,
        frequency,
        angle,
        polarization=None,
    ):
        values = np.ones(
            (
                angle.size,
                frequency.size,
            ),
            dtype=float,
        )

        return xr.DataArray(
            values,
            dims=(
                "time",
                "frequency",
            ),
            coords={
                "time": angle.coords["time"],
                "frequency": (frequency.values),
            },
        )


def test_simulate_from_source_returns_tod():
    simulator = TODSimulator(
        beam=DummyBeam(),
        gain=1.0,
        effective_area_m2=1.0,
    )

    result = simulator.simulate_from_source(
        sat=DummySource(),
    )

    assert "T_antenna" in result
    assert "power" in result

    assert result["T_antenna"].dims == (
        "time",
        "frequency",
    )


def test_legacy_simulate_matches_simulate_from_source():
    simulator = TODSimulator(
        beam=DummyBeam(),
        gain=1.0,
        effective_area_m2=1.0,
    )

    source = DummySource()

    expected = simulator.simulate_from_source(
        sat=source,
    )

    legacy = simulator.simulate(
        sat=source,
    )

    xr.testing.assert_allclose(
        legacy,
        expected,
    )


def test_discrete_source_radiometry_uses_effective_area_and_single_pol_factor():
    from scipy.constants import k as k_B

    class FluxDensitySource:
        def __init__(self):
            self._frequency = xr.DataArray(
                np.array([1.0e9]),
                dims="frequency",
                coords={"frequency": np.array([1.0e9])},
            )

            self._angle = xr.DataArray(
                np.array([0.0]),
                dims="time",
                coords={"time": np.array([0.0])},
            )

            self._range = xr.DataArray(
                np.array([1.0]),
                dims="time",
                coords={"time": self._angle.coords["time"]},
            )

        def psd(self):
            result = xr.DataArray(
                np.array([1.0e-26]),
                dims="frequency",
                coords={"frequency": self._frequency.values},
            )
            result.attrs["units"] = "W/m²/Hz"
            return result

        def angle(self):
            return self._angle

        def range(self):
            return self._range

    simulator = TODSimulator(
        beam=DummyBeam(),
        gain=1.0,
        effective_area_m2=1.0,
    )

    result = simulator.simulate_from_source(
        sat=FluxDensitySource(),
    )

    expected_temperature = 1.0e-26 / (2.0 * k_B)

    np.testing.assert_allclose(
        result["T_antenna"].values,
        expected_temperature,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_discrete_source_requires_effective_area_for_absolute_radiometry():
    simulator = TODSimulator(
        beam=DummyBeam(),
        gain=1.0,
    )

    source = DummySource()

    try:
        simulator.simulate_from_source(
            sat=source,
        )
    except ValueError as exc:
        assert "effective_area_m2" in str(exc)
    else:
        raise AssertionError("Expected discrete-source simulation to require effective_area_m2.")
