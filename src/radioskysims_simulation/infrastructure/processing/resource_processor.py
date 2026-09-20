"""
Módulo: src/radioskysims/infrastructure/processing/resource_processor.py

Processador consciente de recursos que chunkifica a carga de trabalho
baseado em CPUs e memória disponível.

Responsabilidades:
- Detectar número de CPUs lógicos/disponíveis
- Estimar memória necessária por fonte (satélite, local, rádio)
- Dividir fontes em batches que cabem na memória disponível
- Gerenciar o scheduler dask (threads / synchronous / distributed)
- Logar o endereço do dashboard ao usar o scheduler ``distributed``
- Executar simulação em batches e consolidar resultados com merge seguro

Estratégia de execução (threads vs process)
------------------------------------------

- ``threads`` (padrão): workers em threads do mesmo processo. Indicado quando
  o custo dominante é numpy/C (libera o GIL) e o dataset já está em memória
  (T_antenna/power/tracks são arrays dask no mesmo processo). Vantagens:
  baixo overhead e compartilhamento de memória; desvantagem: contenção do GIL
  em código Python puro e menor isolamento.
- ``process`` (``dask_scheduler="distributed"`` com ``dask_processes=True``):
  workers em processos separados via ``LocalCluster``. Indicado quando há
  código Python puro GIL-ligado ou quando se quer isolamento (um worker não
  derruba os demais). Desvantagem: serialização/cópia de arrays entre
  processos (maior uso de memória) e overhead de comunicação.
- RECOMENDAÇÃO para o RadioSkySim: começar com ``threads`` (uso de memória
  contido, ideal para simulações grandes em RAM). Usar ``distributed`` com
  ``dask_processes=True`` quando a carga for GIL-ligada ou para monitorar o
  progresso pelo dashboard; ``dask_processes=False`` (1 worker, threads)
  combina o dashboard com o perfil de memória de threads.

Copyright BINGO Collaboration
Last modified: 2026-08-04
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import xarray as xr
from radioskysims_core.domain.common import FrequencyRange, TimeRange
from radioskysims_sources.domain.signal_source import SignalSource

from radioskysims_simulation.domain.simulation import Simulation, SimulationResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Estimativa de footprint de memória
# ---------------------------------------------------------------------------

# Margem de segurança multiplicativa sobre a estimativa teórica
_SAFETY_FACTOR = 3.0

# Memória reservada para o sistema/sobrecarga Python (MB)
_SYSTEM_RESERVE_MB = 512.0

# Bytes por elemento para float64
_F64_BYTES = 8

# Fatores de footprint por fonte, em unidades de "tensor principal"
# (time × frequency × source=1 × polarization=2 × pointing=1):
#   - 2 tensores de saída acumulados por fonte (T_antenna + power);
#   - ~6 tensores intermediários no pico de um único ``TODSimulator.simulate()``
#     (beam_response, beam_power, flux_at_observer, equivalent_temperature,
#     antenna_temperature, power) — tratado como constante por batch
#     (conservador).
_OUTPUT_TENSORS = 2
_PEAK_TENSORS = 6


def _n_time_steps(time_range: TimeRange) -> int:
    """Nº real de passos temporais do TimeRange (inclusive extremos).

    Usa a duração efetiva de ``time_range`` (e não 24h fixo, como a versão
    antiga). Alinha com ``TimeRange.to_datetime64()``:
    ``round(duration_s / step_s) + 1``.
    """
    try:
        start, end, step_s = time_range.to_tuple()
    except Exception:
        return 1
    if step_s <= 0:
        return 1
    duration_s = float((end - start).sec)
    if duration_s <= 0:
        duration_s = 24 * 3600.0  # fallback defensivo
    return max(1, int(round(duration_s / step_s)) + 1)


def _n_freq_steps(
    frequency_range: FrequencyRange | None,
    num_channels: int | None,
) -> int:
    """Nº de canais de frequência (num_channels ou derivado do range)."""
    n_freq = num_channels or 1
    if frequency_range is not None and n_freq == 1:
        # Deriva o nº de canais da grade real do range (ex.: 1 bin com
        # min == max → value.size == 1). Evita chamar `to_tuple()`, que
        # não existe em FrequencyRange (bug quando num_channels == 1).
        n_freq = max(1, int(frequency_range.value.size))
    return n_freq


def _tensor_mb(n_time: int, n_freq: int, n_sources: int = 1) -> float:
    """MB de UM tensor float64 (time, freq, source, polarization=2, pointing=1)."""
    shape = (n_time, n_freq, n_sources, 2, 1)
    bytes_ = int(np.prod(shape, dtype=np.int64)) * _F64_BYTES
    return float(bytes_) / (1024 * 1024)


@dataclass(frozen=True)
class ProcessingResources:
    """Recursos disponíveis detectados pelo sistema."""

    cpu_count: int
    available_memory_mb: float
    memory_safety_margin: float = _SAFETY_FACTOR
    system_reserve_mb: float = _SYSTEM_RESERVE_MB

    @property
    def usable_memory_mb(self) -> float:
        """Memória utilizável após reserva de sistema e margem de segurança."""
        return max(
            0.0,
            (self.available_memory_mb - self.system_reserve_mb) / self.memory_safety_margin,
        )

    @property
    def usable_cpu_count(self) -> int:
        """CPUs utilizáveis (pelo menos 1)."""
        return max(1, self.cpu_count)

    @classmethod
    def detect(
        cls,
        *,
        cpu_count: int | None = None,
        memory_mb: float | None = None,
        safety_margin: float | None = None,
    ) -> ProcessingResources:
        """Detecta recursos do sistema com fallbacks e overrides.

        Parameters
        ----------
        cpu_count:
            Sobrescreve a detecção automática de CPUs.
        memory_mb:
            Sobrescreve a detecção automática de memória (MB).
        safety_margin:
            Sobrescreve o fator de segurança padrão.

        Returns
        -------
        ProcessingResources
        """
        n_cpus = cpu_count
        if n_cpus is None:
            try:
                n_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                n_cpus = os.cpu_count() or 4

        mem_mb = memory_mb
        if mem_mb is None:
            try:
                import psutil

                mem = psutil.virtual_memory()
                mem_mb = float(mem.available) / (1024 * 1024)
            except ImportError:
                # Fallback: assume 4GB disponível
                mem_mb = 4096.0
                log.warning(
                    "psutil não disponível. Assumindo %.0f MB de memória "
                    "disponível. Instale psutil para detecção automática.",
                    mem_mb,
                )

        return cls(
            cpu_count=n_cpus,
            available_memory_mb=mem_mb,
            memory_safety_margin=safety_margin or _SAFETY_FACTOR,
        )


class MemoryEstimator:
    """Estima o consumo de memória de uma simulação baseada nas fontes."""

    @staticmethod
    def _estimate_n_time(
        time_range: TimeRange,
    ) -> int:
        """Retorna o número real de amostras temporais do TimeRange."""
        _, _, step_s = time_range.to_tuple()

        if step_s <= 0:
            return 1

        duration_s = (time_range.end_jd - time_range.start_jd) * 86400.0

        return max(
            1,
            int(round(duration_s / step_s)) + 1,
        )

    @staticmethod
    def estimate_source_mb(
        sources: Sequence[SignalSource],
        time_range: TimeRange,
        frequency_range: FrequencyRange | None,
        num_channels: int | None,
    ) -> float:
        """Estima a memória de pico para processar N fontes em um batch.

        O cálculo considera:
        - Nº real de passos temporais (duração efetiva do ``time_range``);
        - Nº de canais de frequência;
        - Saída acumulada por fonte (T_antenna + power);
        - Pico de arrays intermediários de um ``simulate()`` (≈6 tensores).

        Parameters
        ----------
        sources:
            Lista de fontes a processar.
        time_range:
            Intervalo temporal da simulação.
        frequency_range:
            Range de frequências.
        num_channels:
            Número de canais de frequência.

        Returns
        -------
        float
            Estimativa em MB.
        """
        n_time = _n_time_steps(time_range)
        n_freq = _n_freq_steps(frequency_range, num_channels)
        tensor_mb = _tensor_mb(n_time, n_freq, n_sources=1)

        n_sources = len(sources)

        # Saída (T+power) acumulada por fonte + pico intermediário (constante).
        total_mb = tensor_mb * (_OUTPUT_TENSORS * n_sources + _PEAK_TENSORS)
        return total_mb

    @staticmethod
    def estimate_single_source_mb(
        time_range: TimeRange,
        frequency_range: FrequencyRange | None = None,
        num_channels: int | None = None,
    ) -> float:
        """Estima a memória para uma única fonte (saída + pico de simulação)."""
        n_time = _n_time_steps(time_range)
        n_freq = _n_freq_steps(frequency_range, num_channels)
        tensor_mb = _tensor_mb(n_time, n_freq, n_sources=1)
        return tensor_mb * (_OUTPUT_TENSORS + _PEAK_TENSORS)

    @staticmethod
    def estimate_total_output_mb(
        sources: Sequence[SignalSource],
        time_range: TimeRange,
        frequency_range: FrequencyRange | None = None,
        num_channels: int | None = None,
    ) -> float:
        """Só a saída final (T_antenna + power) para TODAS as fontes.

        Útil para decidir se o dataset consolidado cabe em RAM ou precisa ser
        gravado/reamostrado em disco (streaming por batches).
        """
        n_time = _n_time_steps(time_range)
        n_freq = _n_freq_steps(frequency_range, num_channels)
        tensor_mb = _tensor_mb(n_time, n_freq, n_sources=1)
        return tensor_mb * _OUTPUT_TENSORS * max(1, len(sources))

        tensor_shape = (
            n_time,
            n_freq,
            1,
            2,
            1,
        )

        tensor_bytes = (
            np.prod(
                tensor_shape,
                dtype=np.int64,
            )
            * _F64_BYTES
        )

        # T_antenna + power + ~3x arrays intermediários
        total_bytes = tensor_bytes * 5

        return float(total_bytes) / (1024 * 1024)


class ResourceLimitError(RuntimeError):
    """Exceção levantada quando os recursos são insuficientes."""


T = TypeVar("T")


@dataclass
class SourceBatch:
    """Lote de fontes para processamento seguro em memória."""

    sources: list[SignalSource]
    estimated_mb: float
    batch_index: int


@dataclass
class BatchResult:
    """Resultado parcial de um batch de fontes."""

    dataset: xr.Dataset | None
    sources: list[SignalSource]
    batch_index: int
    success: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# ResourceAwareProcessor
# ---------------------------------------------------------------------------


class ResourceAwareProcessor(Generic[T]):
    """Processa simulações de forma consciente de recursos.

    Características:
    - Detecta CPUs e memória disponível
    - Divide o workload em batches seguros para memória
    - Configura o scheduler dask apropriadamente
    - Executa cada batch serialmente (evita contenção)
    - Consolida resultados parciais com merge seguro

    Parameters
    ----------
    resources:
        Recursos detectados ou configurados manualmente.
        Se ``None``, detecta automaticamente.
    max_batch_mb:
        Sobrescreve o limite de memória por batch.
    dask_scheduler:
        Scheduler dask a usar: ``"threads"`` (padrão), ``"synchronous"`` ou
        ``"distributed"``. ``"distributed"`` sobe um ``LocalCluster`` +
        ``Client`` e registra o endereço do dashboard em :attr:`dashboard_url`.
    dask_processes:
        Estratégia do scheduler distribuído: ``None`` (padrão) → workers em
        processos separados; ``True`` → processos; ``False`` → 1 worker com
        threads (menos overhead de memória). Ver docstring do módulo.
    logger:
        Logger para eventos do processador.
    """

    def __init__(
        self,
        resources: ProcessingResources | None = None,
        *,
        max_batch_mb: float | None = None,
        dask_scheduler: str = "threads",
        dask_processes: bool | None = None,
        diagnostics_enabled: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
        logger: Any | None = None,
    ) -> None:
        self._resources = resources or ProcessingResources.detect()
        self._max_batch_mb = max_batch_mb
        self._dask_scheduler = dask_scheduler
        self._dask_processes = dask_processes
        self._diagnostics_enabled = diagnostics_enabled
        self._progress_callback = progress_callback
        self._log = logger or log

        self._n_batches: int = 0
        self._batch_results: list[BatchResult] = []

        # Scheduler distribuído (dask.distributed) — dashboard opcional
        self._cluster: Any | None = None
        self._client: Any | None = None
        self._dashboard_url: str | None = None
        self._diagnostics: dict[str, Any] | None = None

    @staticmethod
    def _rss_mb() -> float | None:
        try:
            import psutil

            process = psutil.Process(os.getpid())

            return float(process.memory_info().rss) / (1024 * 1024)
        except Exception:
            return None

    @staticmethod
    def _logical_nbytes_mb(ds: xr.Dataset) -> float | None:
        try:
            return float(ds.nbytes) / (1024 * 1024)
        except Exception:
            return None

    @staticmethod
    def _array_backend(ds: xr.Dataset, var_name: str) -> str | None:
        try:
            if var_name not in ds:
                return None

            data = ds[var_name].data

            return f"{type(data).__module__}.{type(data).__name__}"
        except Exception:
            return None

    @staticmethod
    def _dask_tasks_approx(ds: xr.Dataset, var_name: str) -> int | None:
        try:
            if var_name not in ds:
                return None

            data = ds[var_name].data
            graph_fn = getattr(data, "__dask_graph__", None)
            if graph_fn is None:
                return 0

            graph = graph_fn()

            return len(graph) if graph is not None else 0
        except Exception:
            return None

    @staticmethod
    def _source_type_label(value: Any) -> str:
        """Normaliza source_type vindo de string/enum para rótulo serializável."""
        if value is None:
            return "unknown"

        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return str(enum_value)

        return str(value)

    @classmethod
    def _count_source_types_from_sources(
        cls,
        sources: Sequence[SignalSource],
    ) -> dict[str, int]:
        """Conta source_type diretamente dos objetos Source do batch."""
        counts: dict[str, int] = {
            "satellite": 0,
            "radiosource": 0,
            "localsource": 0,
        }
        try:
            for src in sources:
                label = cls._source_type_label(getattr(src, "source_type", None))
                counts[label] = counts.get(label, 0) + 1
        except Exception:
            # Telemetria não pode interromper a simulação.
            pass

        return counts

    @classmethod
    def _count_source_types_from_partial(
        cls,
        ds_partial: xr.Dataset,
    ) -> dict[str, int] | None:
        """Conta source_type da coord do dataset parcial para conferência."""
        counts: dict[str, int] = {
            "satellite": 0,
            "radiosource": 0,
            "localsource": 0,
        }
        try:
            if "source_type" not in ds_partial.coords:
                return None

            values = np.asarray(ds_partial.coords["source_type"].values)
            if values.size == 0:
                return counts

            labels, freqs = np.unique(values, return_counts=True)
            for label, freq in zip(labels.tolist(), freqs.tolist(), strict=False):
                key = cls._source_type_label(label)
                counts[key] = counts.get(key, 0) + int(freq)

            return counts
        except Exception:
            return None

    # ── Propriedades ──────────────────────────────────────────────

    @property
    def resources(self) -> ProcessingResources:
        """Recursos detectados/configurados."""
        return self._resources

    @property
    def progress_callback(self) -> Callable[[int, int], None] | None:
        """Callback opcional usado para reportar progresso dos batches."""

        return self._progress_callback

    @property
    def cpu_count(self) -> int:
        """Número de CPUs disponíveis."""
        return self._resources.usable_cpu_count

    @property
    def max_batch_mb(self) -> float:
        """Tamanho máximo de batch em MB."""
        if self._max_batch_mb is not None:
            return self._max_batch_mb
        return self._resources.usable_memory_mb

    # ── Particionamento ───────────────────────────────────────────

    def partition_sources(
        self,
        sources: Sequence[SignalSource],
        time_range: TimeRange,
        frequency_range: FrequencyRange | None = None,
        num_channels: int | None = None,
    ) -> list[SourceBatch]:
        """Divide fontes em batches cabíveis na memória disponível.

        Parameters
        ----------
        sources:
            Lista completa de fontes.
        time_range:
            Intervalo temporal.
        frequency_range:
            Range de frequências.
        num_channels:
            Número de canais.

        Returns
        -------
        list[SourceBatch]
            Lista de batches com estimativa de memória.

        Raises
        ------
        ResourceLimitError
            Se uma única fonte excede a memória disponível.
        """
        single_mb = MemoryEstimator.estimate_single_source_mb(
            time_range=time_range,
            frequency_range=frequency_range,
            num_channels=num_channels,
        )

        if single_mb > self.max_batch_mb:
            msg = (
                f"Cada fonte requer ~{single_mb:.0f} MB, mas o limite "
                f"do batch é {self.max_batch_mb:.0f} MB. Considere reduzir "
                "a resolução temporal, o número de canais ou o tempo de "
                "simulação."
            )
            raise ResourceLimitError(msg)

        # Agrupa fontes em batches
        batches: list[SourceBatch] = []
        current_batch: list[SignalSource] = []
        current_mb = 0.0

        for src in sources:
            src_mb = single_mb  # estimativa uniforme por simplicidade

            if current_mb + src_mb > self.max_batch_mb and current_batch:
                # Fecha lote atual
                batches.append(
                    SourceBatch(
                        sources=list(current_batch),
                        estimated_mb=current_mb,
                        batch_index=len(batches),
                    )
                )
                current_batch = [src]
                current_mb = src_mb
            else:
                current_batch.append(src)
                current_mb += src_mb

        # Último lote
        if current_batch:
            batches.append(
                SourceBatch(
                    sources=list(current_batch),
                    estimated_mb=current_mb,
                    batch_index=len(batches),
                )
            )

        self._log.info(
            "partition_sources",
            n_sources=len(sources),
            n_batches=len(batches),
            single_source_mb=round(single_mb, 1),
            max_batch_mb=round(self.max_batch_mb, 0),
            cpu_count=self.cpu_count,
        )

        return batches

    # ── Configuração do scheduler dask ─────────────────────────────

    def _configure_dask(self) -> None:
        """Configura o scheduler dask de acordo com recursos.

        - ``"synchronous"``: single-thread.
        - ``"threads"`` (padrão): threads com ``num_workers = cpu_count``.
        - ``"distributed"``: sobe um ``LocalCluster`` + ``Client`` e registra
          o endereço do dashboard em :attr:`dashboard_url`. Se ``distributed``
          não estiver instalado, cai para ``threads``.
        """
        if self._dask_scheduler == "distributed":
            if self._start_distributed():
                return
            self._log.warning(
                "distributed_unavailable",
                scheduler="threads (fallback)",
            )
            self._dask_scheduler = "threads"

        import dask

        if self.cpu_count <= 1 or self._dask_scheduler == "synchronous":
            dask.config.set(scheduler="synchronous")
            self._log.info("dask_scheduler", scheduler="synchronous")
        else:
            # threads: usa até cpu_count workers
            from dask.config import set as dask_set

            dask_set({"num_workers": self.cpu_count})
            dask.config.set(scheduler="threads")
            self._log.info(
                "dask_scheduler",
                scheduler="threads",
                num_workers=self.cpu_count,
            )

    def _start_distributed(self) -> bool:
        """Sobe um scheduler distribuído (LocalCluster + Client) com dashboard.

        Returns
        -------
        bool
            ``True`` se o cluster foi iniciado; ``False`` se ``distributed``
            não estiver instalado ou a inicialização falhar (fallback p/ threads).
        """
        try:
            from distributed import Client, LocalCluster
        except ImportError:
            return False

        processes = self._dask_processes
        if processes is None:
            processes = True  # padrão: workers em processos separados

        try:
            if processes:
                cluster = LocalCluster(
                    n_workers=max(1, self.cpu_count),
                    threads_per_worker=1,
                    processes=True,
                    dashboard_address=":0",
                )
            else:
                cluster = LocalCluster(
                    n_workers=1,
                    threads_per_worker=max(1, self.cpu_count),
                    processes=False,
                    dashboard_address=":0",
                )
            client = Client(cluster)
        except Exception as exc:
            self._log.warning("distributed_start_failed", error=str(exc))
            return False

        self._cluster = cluster
        self._client = client
        self._dashboard_url = client.dashboard_link
        n_workers = len(client.scheduler_info()["workers"])
        self._log.info(
            "dask_dashboard",
            scheduler="distributed",
            dashboard_url=self._dashboard_url,
            n_workers=n_workers,
            processes=processes,
        )
        # Mensagem visível (notebooks/CLI): endereço do dashboard para conexão
        print(
            f"\n🖥️  Dask Client ativo — dashboard em: {self._dashboard_url} "
            f"(workers={n_workers}, processes={processes})\n"
        )
        return True

    def _restore_dask(self) -> None:
        """Restaura o scheduler dask.

        Para ``"distributed"``, mantém o ``Client`` como scheduler padrão do
        processo — o dashboard continua acessível para o compute a jusante.
        Encerre com :meth:`shutdown`.
        """
        if self._dask_scheduler == "distributed":
            return
        import dask

        dask.config.set(scheduler="threads")

    # ── Scheduler distribuído / dashboard ────────────────────────

    @property
    def dashboard_url(self) -> str | None:
        """URL do dashboard do scheduler distribuído (None se não ativo)."""
        return self._dashboard_url

    def shutdown(self) -> None:
        """Encerra o cluster/client distribuído (se ativo) e restaura threads."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._cluster is not None:
            try:
                self._cluster.close()
            except Exception:
                pass
            self._cluster = None
        self._dashboard_url = None
        try:
            import dask

            dask.config.set(scheduler="threads")
        except Exception:
            pass

    def __enter__(self) -> ResourceAwareProcessor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.shutdown()
        return False

    # ── Execução ──────────────────────────────────────────────────

    def run_batches(
        self,
        batches: list[SourceBatch],
        simulation_factory: Callable[[list[SignalSource]], Simulation],
        *,
        batch_output_dir: str | None = None,
        part_engine: str = "h5netcdf",
    ) -> xr.Dataset:
        """Executa cada batch sequencialmente e consolida resultados.

        Parameters
        ----------
        batches:
            Batches de fontes a processar.
        simulation_factory:
            Função que recebe uma lista de fontes e retorna uma
            :class:`Simulation` pronta para executar.
        batch_output_dir:
            Diretório opcional para **streaming por batches**: quando
            fornecido, o dataset parcial de cada batch é gravado em um arquivo
            ``batch_<i>.nc`` e liberado da memória; ao final, os arquivos são
            reabertos de forma **lazy (dask)** via ``open_mfdataset``. Assim o
            dataset consolidado completo NUNCA é materializado em RAM — a
            memória fica limitada a um batch por vez (essencial para
            simulações cuja saída total excede a RAM, ex.: 24h @ 10 s × 5000
            canais). Se ``None``, consolida em memória como antes.
        part_engine:
            Backend xarray usado para gravar/ler os arquivos parciais
            (padrão ``"h5netcdf"``).

        Returns
        -------
        xr.Dataset
            Dataset consolidado com todas as fontes processadas (numpy quando
            ``batch_output_dir=None``; lazy/dask quando streaming a disco).
        """
        self._configure_dask()
        self._batch_results = []

        partials: list[xr.Dataset] = []
        part_files: list[str] = []
        self._diagnostics = None

        if self._diagnostics_enabled:
            self._diagnostics = {
                "rss_before_first_batch_mb": self._rss_mb(),
                "rss_after_last_merge_mb": None,
                "batches": [],
            }

        source_ids_all: list[str] = []

        try:
            for batch in batches:
                self._log.info(
                    "batch_start",
                    batch=batch.batch_index + 1,
                    total=len(batches),
                    n_sources=len(batch.sources),
                    estimated_mb=round(batch.estimated_mb, 0),
                )
                if self._progress_callback is not None:
                    self._progress_callback(
                        batch.batch_index + 1,
                        len(batches),
                    )
                rss_before_batch_mb = None
                source_type_counts = None
                if self._diagnostics is not None:
                    rss_before_batch_mb = self._rss_mb()
                    source_type_counts = self._count_source_types_from_sources(batch.sources)

                try:
                    sim = simulation_factory(batch.sources)
                    result = sim.run()
                    ds_partial = result.dataset

                    rss_after_simulation_run_mb = None
                    if self._diagnostics is not None:
                        rss_after_simulation_run_mb = self._rss_mb()

                    # Nomes REAIS das fontes que sobreviveram ao filtro do feixe
                    # (N×FWHM). Usar ``batch.sources`` (lista completa) e fatiar
                    # por posição quebraria o alinhamento quando o filtro
                    # descarta fontes — ex.: o Sol era renomeado para
                    # 'localsource' e as fontes recebiam nomes errados.
                    if "source" in ds_partial.coords and "source" in ds_partial.dims:
                        source_ids_all.extend(str(s) for s in ds_partial.coords["source"].values)
                    else:
                        for s in batch.sources:
                            name = getattr(s, "name", None) or getattr(
                                s, "source_id", f"src_{len(source_ids_all)}"
                            )
                            source_ids_all.append(str(name))

                    if self._diagnostics is not None:
                        self._diagnostics["batches"].append(
                            {
                                "batch_index": int(batch.batch_index),
                                "source_count": len(batch.sources),
                                "source_type_counts": source_type_counts,
                                "partial_source_type_counts": self._count_source_types_from_partial(
                                    ds_partial
                                ),
                                "rss_before_batch_mb": rss_before_batch_mb,
                                "rss_after_simulation_run_mb": rss_after_simulation_run_mb,
                                "partial_logical_nbytes_mb": self._logical_nbytes_mb(ds_partial),
                                "t_antenna_backend": self._array_backend(ds_partial, "T_antenna"),
                                "power_backend": self._array_backend(ds_partial, "power"),
                                "t_antenna_dask_tasks_approx": self._dask_tasks_approx(
                                    ds_partial, "T_antenna"
                                ),
                                "power_dask_tasks_approx": self._dask_tasks_approx(
                                    ds_partial, "power"
                                ),
                            }
                        )

                    if batch_output_dir is not None:
                        # Streaming: grava o parcial e libera a memória.
                        Path(batch_output_dir).mkdir(parents=True, exist_ok=True)
                        fpath = Path(batch_output_dir) / (f"batch_{batch.batch_index:04d}.nc")
                        self._write_partial(ds_partial, fpath, engine=part_engine)
                        part_files.append(str(fpath))
                        ds_partial = None  # permite GC liberar o array numpy
                        self._log.info(
                            "batch_written_to_disk",
                            batch=batch.batch_index + 1,
                            path=str(fpath),
                        )
                    else:
                        partials.append(ds_partial)

                    self._batch_results.append(
                        BatchResult(
                            # No streaming a disco o parcial já foi gravado e
                            # liberado — manter o numpy impediria o GC.
                            dataset=(None if batch_output_dir is not None else result.dataset),
                            sources=batch.sources,
                            batch_index=batch.batch_index,
                            success=True,
                        )
                    )
                    self._log.info(
                        "batch_success",
                        batch=batch.batch_index + 1,
                    )
                except Exception as exc:
                    self._log.exception(
                        "batch_failed",
                        batch=batch.batch_index + 1,
                        error=str(exc),
                    )
                    self._batch_results.append(
                        BatchResult(
                            dataset=None,
                            sources=batch.sources,
                            batch_index=batch.batch_index,
                            success=False,
                            error=str(exc),
                        )
                    )
                    # Continua para o próximo batch mesmo se um falhar

            # ── Consolida resultados parciais ──
            if part_files:
                self._log.info(
                    "opening_merged_from_disk",
                    n_parts=len(part_files),
                )
                return self._open_merged(part_files)

            if not partials:
                _errs = [str(r.error) for r in self._batch_results if r.error is not None]
                _detail = "; ".join(_errs[:3]) if _errs else "sem detalhes"
                msg = f"Todos os batches falharam: {_detail}"
                raise RuntimeError(msg)

            self._log.info(
                "consolidating_results",
                n_partials=len(partials),
            )
            merged = self._merge_results(partials, source_ids_all)

            if self._diagnostics is not None:
                self._diagnostics["rss_after_last_merge_mb"] = self._rss_mb()
                self._diagnostics["merged_logical_nbytes_mb"] = self._logical_nbytes_mb(merged)
                self._diagnostics["merged_t_antenna_backend"] = self._array_backend(
                    merged, "T_antenna"
                )
                self._diagnostics["merged_power_backend"] = self._array_backend(merged, "power")
                self._diagnostics["merged_t_antenna_dask_tasks_approx"] = self._dask_tasks_approx(
                    merged, "T_antenna"
                )
                self._diagnostics["merged_power_dask_tasks_approx"] = self._dask_tasks_approx(
                    merged, "power"
                )

            return merged

        finally:
            self._restore_dask()

    @staticmethod
    def _write_partial(
        ds: xr.Dataset,
        path: str | Path,
        *,
        engine: str = "h5netcdf",
    ) -> None:
        """Grava um dataset parcial (já em memória, numpy) em arquivo .nc.

        O dataset parcial é materializado no batch (saída do ``sim.run()``);
        a gravação é direta (sem dask). Remover um arquivo existente evita o
        lock residual de execuções anteriores (mesma precaução de
        ``SimulationResult.to_hdf5``).
        """
        p = Path(path)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
        p.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(p, engine=engine)

    @staticmethod
    def _open_merged(part_files: list[str]) -> xr.Dataset:
        """Reabre os arquivos parciais como UM dataset LAZY (dask).

        Usa ``open_mfdataset`` com concat aninhado ao longo de ``source`` e
        ``compat="override"`` (permitindo metadados distintos entre tipos de
        fonte). O resultado é lazy: cada chunk é lido sob demanda, então o
        dataset completo nunca entra em RAM. Lógica de limpeza de coordenadas
        espelha :meth:`_merge_results`.
        """
        keep_coords = {
            "time",
            "frequency",
            "source",
            "polarization",
            "pointing",
            "source_type",
            "angle_boresight",
            "pointing_ra_deg",
            "pointing_dec_deg",
            "pointing_alt_deg",
            "pointing_az_deg",
        }

        def _preprocess(ds: xr.Dataset) -> xr.Dataset:
            drop = [c for c in ds.coords if c not in ds.dims and c not in keep_coords]
            if drop:
                ds = ds.drop_vars(drop)
            return ds

        return xr.open_mfdataset(
            part_files,
            combine="nested",
            concat_dim="source",
            coords="minimal",
            compat="override",
            combine_attrs="drop_conflicts",
            preprocess=_preprocess,
            chunks="auto",
        )

    # ── Merge seguro ──────────────────────────────────────────────

    @staticmethod
    def _merge_results(
        partials: list[xr.Dataset],
        source_ids: list[str],
    ) -> xr.Dataset:
        """Consolida datasets parciais com merge seguro.

        Usa ``compat="override"`` para permitir que fontes de tipos
        diferentes (satélites, Sol, fontes de rádio) tenham metadados
        distintos (ex.: coordenada ``source`` dos tracks de rádio) sem
        causar ``MergeError``.

        Parameters
        ----------
        partials:
            Lista de datasets parciais (um por batch).
        source_ids:
            Lista completa de IDs de fonte na ordem correta.

        Returns
        -------
        xr.Dataset
        """
        # Limpa coordenadas não-dimension conflitantes antes do concat
        cleaned: list[xr.Dataset] = []
        for ds in partials:
            keep_coords = {
                "time",
                "frequency",
                "source",
                "polarization",
                "pointing",
                "source_type",
                "angle_boresight",
                "pointing_ra_deg",
                "pointing_dec_deg",
                "pointing_alt_deg",
                "pointing_az_deg",
            }
            drop_coords = [c for c in ds.coords if c not in ds.dims and c not in keep_coords]
            if drop_coords:
                ds = ds.drop_vars(drop_coords)
            cleaned.append(ds)

        raw = xr.concat(
            cleaned,
            dim="source",
            compat="override",
            coords="minimal",
            combine_attrs="drop_conflicts",
        )

        # Atribui IDs originais das fontes
        n_src = raw.sizes["source"]
        raw = raw.assign_coords(source=("source", source_ids[:n_src]))

        return raw

    # ── Resumo de execução ────────────────────────────────────────

    @property
    def summary(self) -> dict[str, Any]:
        """Resumo da última execução."""
        n_total = len(self._batch_results)
        n_ok = sum(1 for r in self._batch_results if r.success)
        n_fail = n_total - n_ok
        return {
            "batches": n_total,
            "succeeded": n_ok,
            "failed": n_fail,
            "errors": [r.error for r in self._batch_results if r.error is not None],
            "cpu_count": self.cpu_count,
            "max_batch_mb": round(self.max_batch_mb, 0),
            "dask_scheduler": self._dask_scheduler,
            "dashboard_url": self._dashboard_url,
        }

    @property
    def diagnostics(self) -> dict[str, Any] | None:
        """Diagnósticos de memória/graph por batch da última execução."""
        return self._diagnostics


# ═══════════════════════════════════════════════════════════════════
# Função utilitária para integração direta no SimOrchestrator
# ═══════════════════════════════════════════════════════════════════


def run_orchestrated_simulation(
    *,
    sources: list[SignalSource],
    time_range: TimeRange,
    frequency_range: FrequencyRange | None = None,
    num_channels: int | None = None,
    tod_simulator: Any = None,
    pointing_ra_deg: float | None = None,
    pointing_dec_deg: float | None = None,
    pointing_alt_deg: float | None = None,
    pointing_az_deg: float | None = None,
    mueller: xr.DataArray | None = None,
    combine_polarization: str = "sum",
    max_batch_mb: float | None = None,
    cpu_count: int | None = None,
    memory_mb: float | None = None,
    dask_scheduler: str = "threads",
    dask_processes: bool | None = None,
    logger: Any | None = None,
) -> SimulationResult:
    """Executa simulação multi-fonte com processamento consciente de recursos.

    Esta função integra o ``ResourceAwareProcessor`` diretamente no pipeline
    de simulação existente, permitindo que simulações com muitas fontes
    (satélites + Sol + fontes de rádio) sejam executadas em batches seguros.

    Parameters
    ----------
    sources:
        Lista completa de fontes a simular.
    time_range:
        Intervalo temporal.
    frequency_range:
        Range de frequências.
    num_channels:
        Número de canais de frequência.
    tod_simulator:
        Instância do ``TODSimulator``.
    pointing_ra_deg:
        RA do pointing (graus, preferencial).
    pointing_dec_deg:
        Dec do pointing (graus, preferencial).
    pointing_alt_deg:
        Altitude do pointing (graus, fallback).
    pointing_az_deg:
        Azimute do pointing (graus, fallback).
    mueller:
        Matriz de Mueller opcional.
    combine_polarization:
        Estratégia de combinação de polarização.
    pointing_ra_deg:
        RA do pointing (graus, preferencial).
    pointing_dec_deg:
        Dec do pointing (graus, preferencial).
    pointing_alt_deg:
        Altitude do pointing (graus, fallback).
    pointing_az_deg:
        Azimute do pointing (graus, fallback).
    max_batch_mb:
        Limite de memória por batch (opcional).
    cpu_count:
        Sobrescreve detecção de CPUs.
    memory_mb:
        Sobrescreve detecção de memória (MB).
    logger:
        Logger opcional.

    Returns
    -------
    SimulationResult
        Resultado consolidado.
    """
    resources = ProcessingResources.detect(
        cpu_count=cpu_count,
        memory_mb=memory_mb,
    )

    processor = ResourceAwareProcessor(
        resources=resources,
        max_batch_mb=max_batch_mb,
        dask_scheduler=dask_scheduler,
        dask_processes=dask_processes,
        logger=logger,
    )

    batches = processor.partition_sources(
        sources=sources,
        time_range=time_range,
        frequency_range=frequency_range,
        num_channels=num_channels,
    )

    def _make_simulation(batch_sources: list[SignalSource]) -> Simulation:
        return Simulation(
            sources=batch_sources,
            time_range=time_range,
            tod_simulator=tod_simulator,
            pointing_ra_deg=pointing_ra_deg,
            pointing_dec_deg=pointing_dec_deg,
            pointing_alt_deg=pointing_alt_deg,
            pointing_az_deg=pointing_az_deg,
            mueller=mueller,
            combine_polarization=combine_polarization,
        )

    dataset = processor.run_batches(
        batches=batches,
        simulation_factory=_make_simulation,
    )

    return SimulationResult(
        dataset=dataset,
        sources=sources,
        satellites=[s for s in sources if getattr(s, "source_type", "") == "satellite"],
        time_range=time_range,
    )
