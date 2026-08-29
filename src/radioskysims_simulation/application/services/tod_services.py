from __future__ import annotations

from collections.abc import Iterable

import xarray as xr

from radioskysims_sources.domain.satellite import Satellite
from radioskysims_simulation.domain.tod import TODSimulator


class TODService:
    """
    High-level service to generate Time-Ordered Data (TOD)
    from a collection of satellites.
    """

    def __init__(
        self,
        *,
        simulator: TODSimulator,
    ) -> None:
        self.simulator = simulator

    def run(
        self,
        *,
        satellites: Iterable[Satellite],
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
    ) -> xr.Dataset:
        """
        Generate TOD from all satellites.

        Parameters
        ----------
        satellites:
            Iterable of Satellite objects
        mueller:
            Mueller matrix with dims (polarization, stokes).
        combine_polarization:
            "sum" | "mean" | "separate" for final power.

        Returns
        -------
        xr.Dataset
            Combined TOD(time, frequency)
        """

        tod_list: list[xr.Dataset] = []

        for sat in satellites:
            ds_sat = self.simulator.simulate(
                sat=sat,
                mueller=mueller,
                combine_polarization=combine_polarization,
            )

            # tag satellite id
            ds_sat = ds_sat.assign_coords(satellite=sat.id)

            tod_list.append(ds_sat)

        if not tod_list:
            msg = "No satellites provided for TOD simulation"
            raise ValueError(msg)

        # --------------------------------------------------
        # Stack satellites → sum power
        # --------------------------------------------------
        ds = xr.concat(tod_list, dim="satellite")

        ds["power"] = ds["sat_signal"].sum(dim="satellite")

        ds.attrs.update(
            description="Combined Time-Ordered Data",
            contributions=["satellite"],
            n_satellites=len(tod_list),
        )

        return ds
