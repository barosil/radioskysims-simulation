# radioskysims-simulation

Simulação do ecossistema **RadioSkySim** — o pacote integrador.

## Conteúdo

- `domain.tod` — `TODSimulator`, `SkyMapTODProjector`, modelos TOD
- `domain.simulation` — `Simulation`, `SimulationResult`, `SimulationContext`,
  summary, transits, pointing
- `domain.noise` / `domain.rfi` — modelos, simuladores e serviços de noise/RFI
- `domain.mapmaking` — modelos de mapmaking
- `application.services` — `SimulationService`, `TODOrchestrator`, `SimOrchestrator`,
  serviços de noise/rfi/tod/mapmaking
- `application.mapmaking` — `generate_from_tod`
- `application.ports` — ports de mapmaking/simulations
- `application.tod_panel` — `TODPanel` (subclasse de `PanelApplication`)
- `infrastructure.mapmaking`, `infrastructure.db` (simulation/mapmaking),
  `infrastructure.processing.resource_processor` (chunking/scheduler dask)
- `bootstrap` — builders por pacote + composition root de simulação

Depende de `core`, `viz`, `beam`, `skymap`, `sources`, `radiotelescope`.

## Testes

```bash
uv run --package radioskysims-simulation pytest
```
