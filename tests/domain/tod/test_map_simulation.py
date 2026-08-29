from __future__ import annotations

import healpy as hp
import numpy as np
import pytest
import xarray as xr

from radioskysims_beam.domain.beam.models import Beam
from radioskysims_skymap.domain.skymap import (
    NSide,
    SkyMap,
)
from radioskysims_simulation.domain.tod.simulator import (
    TODSimulator,
)


class DummyMapProjector:
    def project(
        self,
        *,
        skymap,
        pointing_track,
        fwhm_rad=None,
    ):
        return xr.DataArray(
            np.full(
                pointing_track.sizes["time"],
                skymap.data[0],
                dtype=float,
            ),
            dims="time",
            coords={
                "time": pointing_track.coords["time"],
            },
        )


class RecordingMapProjector:
    """
    Projector de teste que registra os valores de FWHM recebidos.

    Serve para verificar se o TODSimulator está calculando
    automaticamente o FWHM a partir do Beam e repassando o valor
    correto ao projector.
    """

    def __init__(self):
        self.fwhm_values = []

    def project(
        self,
        *,
        skymap,
        pointing_track,
        fwhm_rad=None,
    ):
        self.fwhm_values.append(fwhm_rad)

        return xr.DataArray(
            np.full(
                pointing_track.sizes["time"],
                skymap.data[0],
                dtype=float,
            ),
            dims="time",
            coords={
                "time": pointing_track.coords["time"],
            },
        )


def _map(
    value_uk: float,
    freq_ghz: float,
) -> SkyMap:
    nside = 8

    return SkyMap(
        dataset=np.full(
            hp.nside2npix(nside),
            value_uk,
            dtype=float,
        ),
        nside=NSide(nside),
        unit="uK_CMB",
        frame="galactic",
        freq_ghz=freq_ghz,
    )


def _pointing() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "ra": (
                "time",
                np.array(
                    [
                        0.0,
                        0.1,
                        0.2,
                    ]
                ),
            ),
            "dec": (
                "time",
                np.array(
                    [
                        0.0,
                        0.05,
                        0.1,
                    ]
                ),
            ),
        },
        coords={
            "time": np.array(
                [
                    0.0,
                    1.0,
                    2.0,
                ]
            ),
        },
    )


def _beam_with_frequency_dependent_fwhm():
    """
    Beam gaussiano simplificado em duas frequências.

    O objetivo deste helper não é reproduzir um instrumento real,
    mas fornecer um Beam válido para testar a extração automática
    de FWHM.
    """

    angles = np.linspace(
        -5.0,
        5.0,
        401,
    )

    frequencies = np.array(
        [
            0.9e9,
            1.0e9,
        ]
    )

    polarizations = np.array(
        [
            "H",
            "V",
        ]
    )

    amplitude = np.empty(
        (
            len(angles),
            len(frequencies),
            len(polarizations),
        ),
        dtype=float,
    )

    # Perfis gaussianos artificiais.
    # sigma em graus.
    sigmas = {
        0.9e9: 1.0,
        1.0e9: 0.8,
    }

    for fi, frequency in enumerate(frequencies):
        sigma = sigmas[frequency]

        profile = np.exp(-0.5 * (angles / sigma) ** 2)

        for pi in range(len(polarizations)):
            amplitude[
                :,
                fi,
                pi,
            ] = profile

    dataset = xr.Dataset(
        {
            "amplitude": (
                (
                    "angle",
                    "frequency",
                    "polarization",
                ),
                amplitude,
            ),
        },
        coords={
            "angle": angles,
            "frequency": frequencies,
            "polarization": polarizations,
        },
    )

    dataset.coords["angle"].attrs["units"] = "deg"

    dataset.coords["frequency"].attrs["units"] = "Hz"

    return Beam(dataset)


def test_simulate_from_maps_shape():
    simulator = TODSimulator(
        beam=None,
        gain=1.0,
    )

    result = simulator.simulate_from_maps(
        maps=[
            _map(10.0, 0.9),
            _map(20.0, 1.0),
        ],
        pointing_track=_pointing(),
        projector=DummyMapProjector(),
    )

    assert result["T_antenna"].dims == (
        "time",
        "frequency",
    )

    assert result["T_antenna"].shape == (
        3,
        2,
    )


def test_simulate_from_maps_frequency_in_hz():
    simulator = TODSimulator(
        beam=None,
        gain=1.0,
    )

    result = simulator.simulate_from_maps(
        maps=[
            _map(10.0, 0.9),
            _map(20.0, 1.0),
        ],
        pointing_track=_pointing(),
        projector=DummyMapProjector(),
    )

    np.testing.assert_allclose(
        result.coords["frequency"].values,
        [
            0.9e9,
            1.0e9,
        ],
    )


def test_simulate_from_maps_converts_uk_to_k():
    simulator = TODSimulator(
        beam=None,
        gain=1.0,
    )

    result = simulator.simulate_from_maps(
        maps=[
            _map(
                1_000_000.0,
                1.0,
            )
        ],
        pointing_track=_pointing(),
        projector=DummyMapProjector(),
    )

    np.testing.assert_allclose(
        result["T_antenna"].values,
        1.0,
    )


def test_simulate_from_maps_applies_gain():
    simulator = TODSimulator(
        beam=None,
        gain=2.0,
    )

    result = simulator.simulate_from_maps(
        maps=[
            _map(
                1_000_000.0,
                1.0,
            )
        ],
        pointing_track=_pointing(),
        projector=DummyMapProjector(),
    )

    np.testing.assert_allclose(
        result["T_antenna"].values,
        2.0,
    )


def test_simulate_from_maps_power():
    simulator = TODSimulator(
        beam=None,
        gain=1.0,
    )

    result = simulator.simulate_from_maps(
        maps=[
            _map(
                1_000_000.0,
                1.0,
            )
        ],
        pointing_track=_pointing(),
        projector=DummyMapProjector(),
    )

    from scipy.constants import k as k_B

    np.testing.assert_allclose(
        result["power"].values,
        k_B,
    )


def test_simulate_from_maps_requires_maps():
    simulator = TODSimulator(
        beam=None,
        gain=1.0,
    )

    with pytest.raises(ValueError):
        simulator.simulate_from_maps(
            maps=[],
            pointing_track=_pointing(),
            projector=DummyMapProjector(),
        )


def test_beam_fwhm_is_computed_automatically():
    beam = _beam_with_frequency_dependent_fwhm()

    simulator = TODSimulator(
        beam=beam,
        gain=1.0,
    )

    result = simulator._beam_fwhm_by_frequency()

    assert result is not None

    assert result.dims == ("frequency",)

    assert result.size == 2

    assert np.all(np.isfinite(result.values))

    assert np.all(result.values > 0)


def test_simulate_from_maps_uses_beam_fwhm():
    beam = _beam_with_frequency_dependent_fwhm()

    simulator = TODSimulator(
        beam=beam,
        gain=1.0,
    )

    projector = RecordingMapProjector()

    simulator.simulate_from_maps(
        maps=[
            _map(
                10.0,
                0.9,
            ),
            _map(
                20.0,
                1.0,
            ),
        ],
        pointing_track=_pointing(),
        projector=projector,
    )

    assert len(projector.fwhm_values) == 2

    assert all(value is not None for value in projector.fwhm_values)

    assert all(value > 0 for value in projector.fwhm_values)


def test_simulate_from_maps_can_disable_beam():
    beam = _beam_with_frequency_dependent_fwhm()

    simulator = TODSimulator(
        beam=beam,
        gain=1.0,
    )

    projector = RecordingMapProjector()

    simulator.simulate_from_maps(
        maps=[
            _map(
                10.0,
                0.9,
            )
        ],
        pointing_track=_pointing(),
        projector=projector,
        use_beam=False,
    )

    assert projector.fwhm_values == [None]
