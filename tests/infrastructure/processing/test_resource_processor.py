from __future__ import annotations

from radioskysims_simulation.infrastructure.processing.resource_processor import (
    ProcessingResources,
    ResourceAwareProcessor,
)


class DummySource:
    pass


def test_processing_resources_respect_manual_limits() -> None:
    resources = ProcessingResources.detect(
        cpu_count=4,
        memory_mb=4096.0,
    )

    assert resources.cpu_count == 4
    assert resources.available_memory_mb == 4096.0
    assert resources.usable_cpu_count == 4
    assert resources.usable_memory_mb > 0


def test_processor_exposes_resources() -> None:
    resources = ProcessingResources(
        cpu_count=2,
        available_memory_mb=4096.0,
    )

    processor = ResourceAwareProcessor(resources=resources)

    assert processor.resources is resources
    assert processor.cpu_count == 2
    assert processor.max_batch_mb == resources.usable_memory_mb


def test_processor_dashboard_is_none_before_distributed_start() -> None:
    processor = ResourceAwareProcessor(
        resources=ProcessingResources(
            cpu_count=2,
            available_memory_mb=4096.0,
        ),
    )

    assert processor.dashboard_url is None


def test_processor_shutdown_is_safe_without_cluster() -> None:
    processor = ResourceAwareProcessor(
        resources=ProcessingResources(
            cpu_count=2,
            available_memory_mb=4096.0,
        ),
    )

    processor.shutdown()

    assert processor.dashboard_url is None


def test_processor_accepts_progress_callback() -> None:
    events: list[tuple[int, int]] = []

    def progress(current: int, total: int) -> None:
        events.append((current, total))

    processor = ResourceAwareProcessor(
        resources=ProcessingResources(
            cpu_count=2,
            available_memory_mb=4096.0,
        ),
        progress_callback=progress,
    )

    assert processor.progress_callback is progress


def test_processor_reports_progress_for_each_batch(monkeypatch) -> None:
    events: list[tuple[int, int]] = []

    def progress(current: int, total: int) -> None:
        events.append((current, total))

    processor = ResourceAwareProcessor(
        resources=ProcessingResources(
            cpu_count=1,
            available_memory_mb=4096.0,
        ),
        progress_callback=progress,
    )
    monkeypatch.setattr(
        processor,
        "_merge_results",
        lambda partials, _source_ids: partials[0],
    )

    class DummySimulationResult:
        def __init__(self) -> None:
            import xarray as xr

            self.dataset = xr.Dataset()

    class DummySimulation:
        def run(self) -> DummySimulationResult:
            return DummySimulationResult()

    from radioskysims_simulation.infrastructure.processing.resource_processor import (
        SourceBatch,
    )

    batches = [
        SourceBatch(
            sources=[],
            estimated_mb=1.0,
            batch_index=0,
        ),
        SourceBatch(
            sources=[],
            estimated_mb=1.0,
            batch_index=1,
        ),
        SourceBatch(
            sources=[],
            estimated_mb=1.0,
            batch_index=2,
        ),
    ]

    def simulation_factory(_sources):
        return DummySimulation()

    processor.run_batches(
        batches=batches,
        simulation_factory=simulation_factory,
    )

    assert events == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]


def test_tensor_estimate_accounts_for_four_polarizations() -> None:
    """A estimativa usa 4 polarizações (horn H/V COPOL/CROSS).

    Antes usava 2 fixo → subestimava a saída em 2× e impedia a decisão de
    streaming a disco (o dataset tentava caber inteiro em RAM).
    """
    from radioskysims_simulation.infrastructure.processing.resource_processor import (
        _tensor_mb,
    )

    n_time, n_freq = 10, 8
    expected_mb = n_time * n_freq * 4 * 8 / (1024 * 1024)
    assert _tensor_mb(n_time, n_freq, 1) == expected_mb


def test_run_batches_parallel_writes_one_nc_per_chunk(tmp_path, monkeypatch) -> None:
    """Streaming a disco: um ``batch_<i>.nc`` por chunk, executados em paralelo."""
    from types import SimpleNamespace

    import numpy as np
    import xarray as xr
    from radioskysims_simulation.infrastructure.processing.resource_processor import (
        SourceBatch,
    )

    processor = ResourceAwareProcessor(
        resources=ProcessingResources(cpu_count=2, available_memory_mb=4096.0),
    )
    merged_marker = object()
    monkeypatch.setattr(processor, "_open_merged", lambda files: merged_marker)

    class DummySimulation:
        def __init__(self, sources) -> None:
            self._sources = sources

        def run(self) -> SimpleNamespace:
            ds = xr.Dataset({"T_antenna": (("time", "source"), np.ones((2, 1)))})
            return SimpleNamespace(dataset=ds)

    batches = [SourceBatch(sources=[], estimated_mb=1.0, batch_index=i) for i in range(3)]
    outdir = tmp_path / "parts"

    result = processor.run_batches(
        batches=batches,
        simulation_factory=lambda srcs: DummySimulation(srcs),
        batch_output_dir=str(outdir),
    )

    assert result is merged_marker
    assert sorted(p.name for p in outdir.glob("batch_*.nc")) == [
        "batch_0000.nc",
        "batch_0001.nc",
        "batch_0002.nc",
    ]
    assert sorted(r.batch_index for r in processor._batch_results) == [0, 1, 2]
