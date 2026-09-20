from __future__ import annotations

from radioskysims_simulation.application.services.sim_orchestrator import (
    SimOrchestrator,
)
from radioskysims_simulation.infrastructure.processing.resource_processor import (
    ProcessingResources,
    ResourceAwareProcessor,
)


class DummyContext:
    local_sources_service = None
    radio_source_catalog_service = None
    tod_orchestrator = None
    galaxy_map_generator = None
    skymap_repository = None
    skymap_tod_projector = None
    config_loader = None
    logger = None
    neutral_hydrogen_map_generator = None
    cosmology_parameters = None


def test_sim_orchestrator_accepts_injected_processor() -> None:
    processor = ResourceAwareProcessor(
        resources=ProcessingResources(
            cpu_count=2,
            available_memory_mb=4096.0,
        )
    )

    orchestrator = SimOrchestrator(
        DummyContext(),  # type: ignore[arg-type]
        processor=processor,
    )

    assert orchestrator.processor is processor


def test_sim_orchestrator_reuses_injected_processor() -> None:
    processor = ResourceAwareProcessor(
        resources=ProcessingResources(
            cpu_count=2,
            available_memory_mb=4096.0,
        )
    )

    orchestrator = SimOrchestrator(
        DummyContext(),  # type: ignore[arg-type]
        processor=processor,
    )

    resolved = orchestrator._resolve_processor(
        max_batch_mb=None,
        cpu_count=None,
        memory_mb=None,
        dask_scheduler=None,
        dask_processes=None,
        processor_diagnostics=False,
    )

    assert resolved is processor
    assert orchestrator.processor is processor


def test_from_settings_accepts_processor(monkeypatch) -> None:
    processor = ResourceAwareProcessor(
        resources=ProcessingResources(
            cpu_count=2,
            available_memory_mb=4096.0,
        )
    )

    dummy_context = DummyContext()

    def fake_create_simulation_context(*, telescope, settings_file):
        assert telescope == "bingo"
        assert settings_file is None
        return dummy_context

    import radioskysims_simulation.bootstrap.composition as composition

    monkeypatch.setattr(
        composition,
        "create_simulation_context",
        fake_create_simulation_context,
    )

    orchestrator = SimOrchestrator.from_settings(
        "bingo",
        processor=processor,
    )

    assert orchestrator.processor is processor


def test_orchestrator_exposes_processor_dashboard_url() -> None:
    processor = ResourceAwareProcessor(
        resources=ProcessingResources(
            cpu_count=2,
            available_memory_mb=4096.0,
        )
    )

    processor._dashboard_url = "http://127.0.0.1:8787/status"

    orchestrator = SimOrchestrator(
        DummyContext(),  # type: ignore[arg-type]
        processor=processor,
    )

    assert orchestrator.dashboard_url == "http://127.0.0.1:8787/status"


def test_orchestrator_dashboard_url_is_none_without_processor() -> None:
    orchestrator = SimOrchestrator(
        DummyContext(),  # type: ignore[arg-type]
    )

    assert orchestrator.dashboard_url is None
