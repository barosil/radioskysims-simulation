from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import xarray as xr

from radioskysims_simulation.domain.tod import TODSimulator


class TODOrchestrator:
    """
    Application service responsible for orchestrating TOD contributions.

    The physical simulation itself remains delegated to ``TODSimulator``.

    Supported contribution paths:

    - discrete sources via ``simulate_from_source()``;
    - diffuse HEALPix maps via ``simulate_from_maps()``;
    - combined source + map TOD.
    """

    def __init__(
        self,
        *,
        simulator: TODSimulator,
    ) -> None:
        self._simulator = simulator

    def run_sources(
        self,
        *,
        sources: Iterable[Any],
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
    ) -> xr.Dataset:
        """
        Generate TOD from discrete SignalSource-compatible objects.
        """

        source_list = list(sources)

        if not source_list:
            msg = "run_sources requires at least one source."
            raise ValueError(msg)

        datasets: list[xr.Dataset] = []

        source_names: list[str] = []

        for index, source in enumerate(source_list):
            dataset = self._simulator.simulate_from_source(
                sat=source,
                mueller=mueller,
                combine_polarization=(combine_polarization),
            )

            source_name = (
                getattr(
                    source,
                    "name",
                    None,
                )
                or getattr(
                    source,
                    "id",
                    None,
                )
                or f"source_{index}"
            )

            datasets.append(dataset)

            source_names.append(str(source_name))

        combined = xr.concat(
            datasets,
            dim=xr.DataArray(
                source_names,
                dims="source",
                name="source",
            ),
        )

        combined.attrs.update(
            {
                "description": ("Time-Ordered Data from discrete sources"),
                "contributions": [
                    "source",
                ],
                "n_sources": len(source_names),
            }
        )

        return combined

    def run_maps(
        self,
        *,
        maps: Iterable[Any],
        pointing_track: xr.Dataset,
        projector: Any,
        use_beam: bool = True,
    ) -> xr.Dataset:
        """
        Generate TOD from diffuse HEALPix maps.
        """

        map_list = list(maps)

        if not map_list:
            msg = "run_maps requires at least one SkyMap."
            raise ValueError(msg)

        return self._simulator.simulate_from_maps(
            maps=map_list,
            pointing_track=pointing_track,
            projector=projector,
            use_beam=use_beam,
        )

    def run(
        self,
        *,
        sources: Iterable[Any] | None = None,
        maps: Iterable[Any] | None = None,
        pointing_track: xr.Dataset | None = None,
        projector: Any | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
        use_beam: bool = True,
    ) -> xr.Dataset:
        """
        Execute every requested TOD contribution.

        At least one of ``sources`` or ``maps`` must be provided.
        """

        source_list = list(sources) if sources is not None else []

        map_list = list(maps) if maps is not None else []

        if not source_list and not map_list:
            msg = "TODOrchestrator requires sources, maps, or both."
            raise ValueError(msg)

        source_tod: xr.Dataset | None = None
        map_tod: xr.Dataset | None = None

        # --------------------------------------------------
        # Discrete source contribution
        # --------------------------------------------------
        if source_list:
            source_tod = self.run_sources(
                sources=source_list,
                mueller=mueller,
                combine_polarization=(combine_polarization),
            )

        # --------------------------------------------------
        # Diffuse sky contribution
        # --------------------------------------------------
        if map_list:
            if pointing_track is None:
                msg = "pointing_track is required when maps are provided."
                raise ValueError(msg)

            if projector is None:
                msg = "projector is required when maps are provided."
                raise ValueError(msg)

            map_tod = self.run_maps(
                maps=map_list,
                pointing_track=pointing_track,
                projector=projector,
                use_beam=use_beam,
            )

        # --------------------------------------------------
        # Only one contribution requested
        # --------------------------------------------------
        if source_tod is None:
            assert map_tod is not None

            map_tod.attrs["contributions"] = ["skymap"]

            return map_tod

        if map_tod is None:
            source_tod.attrs["contributions"] = ["source"]

            return source_tod

        # --------------------------------------------------
        # Combined contribution
        # --------------------------------------------------
        return self.combine(
            source_tod=source_tod,
            map_tod=map_tod,
        )

    @staticmethod
    def combine(
        *,
        source_tod: xr.Dataset,
        map_tod: xr.Dataset,
    ) -> xr.Dataset:
        """
        Combine discrete-source and diffuse-map contributions.

        The physical quantities are additive in antenna temperature:

            T_total = T_sources + T_map

        and consequently:

            P_total = P_sources + P_map
        """

        source_temperature = source_tod["T_antenna"]

        # Source TOD may contain the explicit source dimension.
        if "source" in source_temperature.dims:
            source_temperature = source_temperature.sum(dim="source")

        source_power = source_tod["power"]

        if "source" in source_power.dims:
            source_power = source_power.sum(dim="source")

        source_temperature, map_temperature = xr.align(
            source_temperature,
            map_tod["T_antenna"],
            join="exact",
        )

        source_power, map_power = xr.align(
            source_power,
            map_tod["power"],
            join="exact",
        )

        total_temperature = source_temperature + map_temperature

        total_power = source_power + map_power

        result = xr.Dataset(
            data_vars={
                "T_antenna": total_temperature,
                "power": total_power,
            },
            attrs={
                "description": ("Combined Time-Ordered Data"),
                "contributions": [
                    "source",
                    "skymap",
                ],
            },
        )

        result["T_antenna"].attrs["units"] = "K"

        result["power"].attrs["units"] = "W/Hz"

        return result

    @staticmethod
    def combine_maps(
        *,
        map_tods: Iterable[xr.Dataset],
    ) -> xr.Dataset:
        """
        Combine multiple diffuse-map TOD contributions.

        All datasets must share exactly the same time/frequency grid.
        """

        datasets = list(map_tods)

        if not datasets:
            msg = "combine_maps requires at least one map TOD."
            raise ValueError(msg)

        temperature = datasets[0]["T_antenna"]
        power = datasets[0]["power"]

        for dataset in datasets[1:]:
            next_temperature, aligned_temperature = xr.align(
                temperature,
                dataset["T_antenna"],
                join="exact",
            )

            next_power, aligned_power = xr.align(
                power,
                dataset["power"],
                join="exact",
            )

            temperature = next_temperature + aligned_temperature

            power = next_power + aligned_power

        result = xr.Dataset(
            data_vars={
                "T_antenna": temperature,
                "power": power,
            },
            attrs={
                "description": ("Combined diffuse-map Time-Ordered Data"),
                "contributions": [
                    "skymap",
                ],
                "n_map_components": len(datasets),
            },
        )

        result["T_antenna"].attrs["units"] = "K"
        result["power"].attrs["units"] = "W/Hz"

        return result
