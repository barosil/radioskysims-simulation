from __future__ import annotations

import holoviews as hv
import hvplot.xarray  # noqa: F401  # registra .hvplot em xarray
import panel as pn
import xarray as xr
from radioskysims_viz import PanelApplication

from radioskysims_simulation.domain.simulation import SimulationResult

hv.extension("bokeh")


class TODPanel(PanelApplication):
    """Visualizador simples para resultados de TOD.

    Usa Panel + HoloViews/HVPlot para inspecionar um ``SimulationResult``
    (tipicamente vindo do :class:`SimOrchestrator`).
    (Refatorado para herdar de ``PanelApplication`` — Fase 5.)
    """

    def __init__(self, result: SimulationResult):  # type: ignore[override]
        self.result = result

        ds = result.dataset
        if "T_antenna" not in ds:
            msg = "SimulationResult.dataset não contém 'T_antenna'."
            raise ValueError(msg)

        # Agrega dimensões auxiliares para obter um mapa time x frequency
        da = ds["T_antenna"]
        reduce_dims = [d for d in da.dims if d not in ("time", "frequency")]
        if reduce_dims:
            da = da.mean(dim=reduce_dims)

        self._dataset = ds
        self._da = da

        super().__init__()

    def build_sidebar(self):
        """Constrói os controles (sidebar) do painel."""
        # Widgets básicos
        self.log_scale = pn.widgets.Checkbox(name="Escala log10(T)", value=False)
        self.var_select = pn.widgets.Select(
            name="Variável",
            options=["T_antenna", "power"],
            value="T_antenna",
        )
        self.sat_select = None
        if "satellite" in self._dataset.dims and self._dataset.dims["satellite"] > 1:
            sats = list(self._dataset.coords["satellite"].values)
            self.sat_select = pn.widgets.Select(
                name="Satélite",
                options=sats,
                value=sats[0],
            )

        # Callbacks
        self.log_scale.param.watch(self._update_plot, "value")
        self.var_select.param.watch(self._update_plot, "value")
        if self.sat_select is not None:
            self.sat_select.param.watch(self._update_plot, "value")

        controls = [self.var_select, self.log_scale]
        if self.sat_select is not None:
            controls.append(self.sat_select)

        return pn.Column(*controls, sizing_mode="fixed", width=250)

    def build_main(self):
        """Constrói a área principal (plot)."""
        self._plot_pane = pn.pane.HoloViews(self._make_plot(), sizing_mode="stretch_both")
        return self._plot_pane

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def _current_dataarray(self) -> xr.DataArray:
        ds = self._dataset
        var = self.var_select.value
        if var not in ds:
            msg = f"Variável '{var}' não encontrada no dataset."
            raise ValueError(msg)

        da = ds[var]
        if self.sat_select is not None and "satellite" in da.dims:
            da = da.sel(satellite=self.sat_select.value)

        reduce_dims = [d for d in da.dims if d not in ("time", "frequency")]
        if reduce_dims:
            da = da.mean(dim=reduce_dims)
        return da

    def _make_plot(self):
        da = self._current_dataarray()

        if self.log_scale.value:
            import numpy as np

            da_plot = xr.apply_ufunc(  # type: ignore[no-untyped-call]
                lambda x: np.where(x > 0, np.log10(x), x),
                da,
            )
            title = f"log10({self.var_select.value})"
        else:
            da_plot = da
            title = self.var_select.value

        return da_plot.hvplot.image(
            x="time",
            y="frequency",
            rasterize=True,
            cmap="viridis",
            colorbar=True,
            title=title,
        )

    def _update_plot(self, *_):  # type: ignore[override]
        self._plot_pane.object = self._make_plot()


if __name__ == "__main__":
    pn.extension()
    pn.state.notifications.info(
        "TODPanel deve ser instanciado com um SimulationResult. Importe TODPanel "
        "a partir de radioskysims_simulation.application.tod_panel e passe o resultado de uma simulação.",
        duration=5000,
    )
    pn.serve(pn.pane.Markdown("# TODPanel\nImporte e use a partir de Python."), show=True)
