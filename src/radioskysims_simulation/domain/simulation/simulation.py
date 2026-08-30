"""
Módulo: src/radioskysims/domain/simulation/simulation.py

Executor físico do pipeline de simulação.

Aceita qualquer fonte que implemente o protocolo SignalSource
(satélites, fontes locais, fontes de rádio) e executa:
    geometria -> beam -> TOD -> dataset

Copyright BINGO Collaboration
Last modified: 2026-07-28
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from astropy.time import Time

from radioskysims_core.domain.common import FrequencyRange, TimeRange
from radioskysims_sources.domain.signal_source import (
    SignalSource,
    compute_theta_boresight,
)
from radioskysims_simulation.domain.simulation.transits import (
    chunk_uniform,
    extract_transits,
)

if TYPE_CHECKING:
    from radioskysims_simulation.domain.tod import TODSimulator

logger = logging.getLogger(__name__)

_VALID_SOURCE_TYPES: frozenset[str] = frozenset(
    {
        "satellite",
        "localsource",
        "radiosource",
    }
)


def _time_range_from_attrs(attrs: dict[str, Any], ds: xr.Dataset) -> TimeRange | None:
    """Reconstroi um :class:`TimeRange` a partir dos attrs persistidos.

    Prioriza os campos ISO gravados por :meth:`SimulationResult._persistence_attrs`
    (``time_start``/``time_end``/``time_step_s``). Como fallback, deriva do eixo
    ``time`` do dataset. Retorna ``None`` se não for possível.
    """
    start = attrs.get("time_start")
    end = attrs.get("time_end")
    step = attrs.get("time_step_s")
    if start and end:
        try:
            return TimeRange.from_iso(
                str(start),
                str(end),
                resolution=float(step) if step else None,
            )
        except Exception:
            pass

    if "time" in ds.coords:
        t = ds.coords["time"].values
        if t.size >= 2:
            try:
                t_s = t.astype("datetime64[s]")
                start_s = np.datetime_as_string(t_s[0], unit="s")
                end_s = np.datetime_as_string(t_s[-1], unit="s")
                step_s = float(np.median(np.diff(t_s.astype(float))))
                return TimeRange.from_iso(start_s, end_s, resolution=step_s)
            except Exception:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════
# SimulationEvents — objetos no feixe, T_antenna médio, time-slice
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SimulationEvents:
    """Eventos de fontes no feixe durante a simulação (dataset de trânsitos).

    Atributos:
        time_start_utc: Início do intervalo (ISO).
        time_end_utc: Fim do intervalo (ISO).
        duration_s: Duração em segundos.
        objects_in_beam: Lista de nomes de fontes que atravessaram o feixe.
        t_antenna_mean: T_antenna médio nos trânsitos [K].
        beam_events: Dataset (time × frequency × source) apenas para
            instantes com fontes dentro do limiar do feixe — construído por
            chunks, sem materializar o dataset completo.
        transits: Dataset pequeno ``(transit,)`` com um registro por trânsito
            (fonte, tipo, início/fim, duração, T_antenna média/máx, ângulo mín).
    """

    time_start_utc: str
    time_end_utc: str
    duration_s: float
    objects_in_beam: list[str]
    t_antenna_mean: float
    beam_events: xr.Dataset
    transits: xr.Dataset | None = None

    @classmethod
    def from_dataset(
        cls,
        dataset: xr.Dataset,
        time_range: TimeRange,
        fwhm_threshold_deg: float | None = None,
        source_types: list[str] | None = None,
        **kwargs: Any,
    ) -> SimulationEvents:
        """Constrói SimulationEvents a partir de um Dataset (lazy ou não).

        Usa o ângulo de boresight presente no dataset para detectar fontes
        no feixe: se ``angle_boresight < N * FWHM``, a fonte é considerada
        em trânsito. O dataset é percorrido em **chunks ao longo do tempo** —
        cada chunk é computado isoladamente, de modo que o dataset completo
        nunca é materializado de uma vez na memória. Cada intervalo contíguo
        dentro do feixe vira um trânsito no dataset ``transits`` (pequeno);
        ``beam_events`` mantém o TOD apenas nos instantes de trânsito.

        Parameters
        ----------
        dataset:
            Dataset lazy (dask) ou materializado (numpy) com T_antenna.
        time_range:
            Intervalo temporal da simulação.
        fwhm_threshold_deg:
            Limiar angular do feixe (graus). Se fornecido, filtra
            fontes pelo ângulo de boresight.
        source_types:
            Se fornecido, filtra apenas estes tipos de fonte.
        **kwargs:
            Repassados para :func:`radioskysims_simulation.domain.simulation.transits.extract_transits`
            (ex.: ``chunk_s``, ``target_mb``).

        Returns
        -------
        SimulationEvents
        """
        tr_start, tr_end, _ = time_range.to_tuple()
        duration_s = (Time(tr_end) - Time(tr_start)).sec

        res = extract_transits(
            dataset,
            fwhm_threshold_deg=fwhm_threshold_deg,
            source_types=source_types,
            **kwargs,
        )

        return cls(
            time_start_utc=tr_start.isot,
            time_end_utc=tr_end.isot,
            duration_s=duration_s,
            objects_in_beam=res.objects_in_beam,
            t_antenna_mean=res.t_antenna_mean,
            beam_events=res.beam_events,
            transits=res.transits,
        )


# ═══════════════════════════════════════════════════════════════════
# SimulationOrchestratorResult — resultado composto do SimOrchestrator
# ═══════════════════════════════════════════════════════════════════


class SimulationOrchestratorResult:
    """Resultado composto retornado pelo SimOrchestrator.

    Atributos:
        simulation_result: Objeto SimulationResult com dataset lazy.
        config: Configurações utilizadas na simulação.
        events: SimulationEvents computado (não-lazy). Acessar esta
            propriedade dispara a materialização do dataset.
    """

    def __init__(
        self,
        simulation_result: SimulationResult,
        config: dict[str, Any],
        events: SimulationEvents | None = None,
    ) -> None:
        self.simulation_result = simulation_result
        self.config = config
        self._events = events

    @property
    def events(self) -> SimulationEvents:
        """Retorna SimulationEvents, computando sob demanda (por chunks).

        Contrato de materialização de ``events``:
        - o ``dataset`` principal do ``SimulationResult`` permanece lazy;
        - ``events`` é um produto derivado, materializado incrementalmente;
        - a extração percorre slices/chunks de ``time`` e computa apenas
          esses blocos;
        - não deve ocorrer ``Dataset.compute()`` sobre o dataset completo.

        Na primeira chamada, os trânsitos são extraídos por
        :func:`extract_transits` com processamento chunk-wise. Chamadas
        subsequentes retornam o objeto cacheado sem recomputar.
        """
        if self._events is None:
            self._events = SimulationEvents.from_dataset(
                dataset=self.simulation_result.dataset,
                time_range=self.simulation_result.time_range,
            )
        return self._events

    def compute(self, **kwargs: Any) -> xr.Dataset:
        """Atalho para ``simulation_result.compute(**kwargs)``.

        Parameters
        ----------
        **kwargs
            Repassados para ``SimulationResult.compute()``.
        """
        return self.simulation_result.compute(**kwargs)

    # ── Resumo estruturado (config + stats) ──────────────────

    def summary(self, *, compute_stats: bool = True) -> dict[str, Any]:
        """Retorna o resumo completo da simulação (dict aninhado).

        Inclui os blocos ``observer``, ``simulation`` e ``computation``.
        O bloco de computação (dims/chunks/tamanho) é sempre preenchido
        (leitura lazy segura). As estatísticas de céu (T_antenna/fluxo por
        tipo de fonte) são preenchidas apenas quando o dataset já está
        materializado (numpy) — ou via ``compute_stats``.

        Nota: a chamada **preenche e muta** ``self.config`` in-place
        (efeito colateral idempotente); evite compartilhar esse dict entre
        objetos se precisar de imutabilidade.
        """
        from radioskysims_simulation.domain.simulation.summary import (
            populate_computation,
            populate_sky_stats,
        )

        populate_computation(self.simulation_result, self.config)

        ds = self.simulation_result.dataset
        try:
            import dask.array as da

            is_dask = "T_antenna" in ds and isinstance(ds["T_antenna"].data, da.Array)
        except ImportError:
            is_dask = False

        if (not is_dask) and (self._events is not None or compute_stats):
            populate_sky_stats(self.simulation_result, self.config)

        return self.config

    def format_summary(self) -> str:
        """Resumo formatado para exibição (repr legível)."""
        from radioskysims_simulation.domain.simulation.summary import format_summary

        return format_summary(self.summary())

    def to_json(self, **kwargs: Any) -> str:
        """Serializa o resumo em JSON."""
        from radioskysims_simulation.domain.simulation.summary import to_json

        return to_json(self.summary(), **kwargs)

    # ── Persistência do resultado composto ───────────────────

    def save_result(
        self,
        filepath: str,
        *,
        fmt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Salva o resultado composto em arquivo (dataset + resumo JSON).

        Grava o dataset do ``SimulationResult`` (HDF5/Zarr) e um ``sidecar``
        ``<filepath>.json`` com o resumo completo (``to_json()``). Atualiza os
        flags de computação (``saved``/``filepath``) no ``config`` interno.

        Returns
        -------
        str
            Caminho do arquivo sidecar (resumo JSON).
        """
        self.simulation_result.save(filepath, fmt=fmt, **kwargs)
        comp = self.config.setdefault("computation", {})
        comp["saved"] = True
        comp["filepath"] = str(filepath)
        sidecar = str(filepath) + ".json"
        Path(sidecar).write_text(self.to_json(), encoding="utf-8")
        return sidecar

    @classmethod
    def load_result(
        cls,
        filepath: str,
        *,
        fmt: str | None = None,
        **kwargs: Any,
    ) -> SimulationOrchestratorResult:
        """Reconstrói um resultado composto persistido por ``save_result``.

        Carrega o dataset (HDF5/Zarr) e o ``sidecar`` JSON de resumo. Se o
        sidecar não existir, o ``config`` é inicializado vazio e preenchido sob
        demanda por ``summary()``/``format_summary()``.
        """
        sim_result = SimulationResult.load(filepath, fmt=fmt, **kwargs)
        config: dict[str, Any] = {}
        sidecar = str(filepath) + ".json"
        if Path(sidecar).exists():
            try:
                config = json.loads(Path(sidecar).read_text(encoding="utf-8"))
            except (ValueError, OSError):
                config = {}
        return cls(simulation_result=sim_result, config=config, events=None)

    def __repr__(self) -> str:
        return self.format_summary()


# ═══════════════════════════════════════════════════════════════════
# SimulationResult — resultado científico da simulação
# ═══════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SimulationResult:
    """
    Resultado de uma simulação científica com métodos seguros
    para materialização dos dados (compute, to_zarr, to_hdf5).

    Attributes
    ----------
    dataset : xr.Dataset
        Dataset lazy (dask) contendo TOD, metadados e coordenadas.
    sources : list[SignalSource]
        Fontes incluídas na simulação.
    satellites : list[Satellite]
        Satélites incluídos na simulação (deprecated, use ``sources``).
    time_range : TimeRange
        Intervalo temporal simulado.
    provenance : dict
        Metadados de proveniência (timestamp, pipeline, parâmetros).
    """

    dataset: xr.Dataset
    sources: list  # list[SignalSource]
    satellites: list  # list[Satellite] — kept for backward compat
    time_range: TimeRange
    freq_range: FrequencyRange | None = field(default=None)
    provenance: dict[str, Any] = field(default_factory=dict)

    def _repr_backend(self) -> str:
        """Retorna backend resumido sem materializar variáveis."""
        try:
            import dask.array as da
        except Exception:
            da = None  # type: ignore[assignment]

        has_dask = False
        has_numpy = False
        for var in self.dataset.data_vars.values():
            data = var.data
            if da is not None and isinstance(data, da.Array):
                has_dask = True
            else:
                has_numpy = True
            if has_dask and has_numpy:
                return "mixed"
        if has_dask:
            return "dask"
        return "numpy"

    def _repr_time_range(self) -> str | None:
        """Retorna intervalo temporal curto sem acessar .values em data vars."""
        if "time" not in self.dataset.coords:
            return None
        try:
            t_index = self.dataset.indexes.get("time")
            if t_index is not None and len(t_index) > 0:
                return f"{t_index[0]} -> {t_index[-1]}"
        except Exception:
            pass
        return f"n={int(self.dataset.sizes.get('time', 0))}"

    def _repr_frequency_range(self) -> str | None:
        """Retorna faixa de frequência curta sem materialização do dataset."""
        if "frequency" not in self.dataset.coords:
            return None
        if self.freq_range is not None:
            try:
                return f"{self.freq_range.min_mhz:g}-{self.freq_range.max_mhz:g} MHz"
            except Exception:
                pass
        return f"n={int(self.dataset.sizes.get('frequency', 0))}"

    def __post_init__(self) -> None:
        # ── Auto-deriva freq_range do dataset se não fornecido ──
        if self.freq_range is None and "frequency" in self.dataset.coords:
            freq_vals = self.dataset.coords["frequency"].values
            # A coordenada ``frequency`` do dataset TOD está em Hz — converte
            # para MHz antes de derivar o FrequencyRange (que opera em MHz).
            try:
                import astropy.units as u

                if not isinstance(freq_vals, u.Quantity):
                    freq_vals = u.Quantity(freq_vals, u.Hz, copy=False)
                self.freq_range = FrequencyRange.from_array(freq_vals)
            except Exception:
                self.freq_range = FrequencyRange.from_array(np.asarray(freq_vals, dtype=float))
        elif self.freq_range is None:
            self.freq_range = FrequencyRange(min_mhz=0.0, max_mhz=0.0, resolution_mhz=0.0)

        if not self.provenance:
            self.provenance["created_at"] = datetime.now(UTC).isoformat()
            self.provenance["pipeline"] = self.dataset.attrs.get("pipeline", "tod")
            self.provenance["n_sources"] = self.dataset.attrs.get("n_sources", len(self.sources))
            self.provenance["time_start"] = self.dataset.attrs.get("time_start", "")
            self.provenance["time_end"] = self.dataset.attrs.get("time_end", "")
            self.provenance["time_step_s"] = self.dataset.attrs.get("time_step_s", 0)

            # ── Proveniência detalhada das fontes ──────────────
            sat_details = []
            local_details = []
            radio_details = []

            for src in self.sources:
                meta = getattr(src, "metadata", {}) or {}
                stype = getattr(src, "source_type", getattr(src, "source_type_str", "unknown"))

                if stype == "satellite":
                    sat_details.append(
                        {
                            "name": getattr(src, "name", "unknown"),
                            "norad_id": meta.get("norad_id"),
                            "constellation": meta.get("constellation"),
                            "has_tle": bool(meta.get("tle_line1")),
                        }
                    )
                elif stype == "localsource":
                    local_details.append(
                        {
                            "name": getattr(src, "name", "unknown"),
                            "source_file": meta.get("source_file"),
                        }
                    )
                elif stype == "radiosource" or stype == "radiosource_group":
                    radio_details.append(
                        {
                            "name": getattr(src, "name", "unknown"),
                            "catalog": meta.get("catalog"),
                            "n_sources_in_group": meta.get("n_sources_in_group", 1),
                        }
                    )

            if sat_details:
                self.provenance["satellites"] = sat_details
                self.provenance["n_satellites"] = len(sat_details)
            if local_details:
                self.provenance["local_sources"] = local_details
                self.provenance["n_local_sources"] = len(local_details)
            if radio_details:
                self.provenance["radio_sources"] = radio_details
                self.provenance["n_radio_sources"] = sum(
                    r.get("n_sources_in_group", 1) for r in radio_details
                )

            # ── Git hash (opcional) ────────────────────────────
            try:
                import subprocess

                result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self.provenance["git_hash"] = result.stdout.strip()
            except Exception:
                pass

    def __repr__(self) -> str:
        dims = {k: int(v) for k, v in self.dataset.sizes.items()}
        backend = self._repr_backend()
        n_sources = int(self.dataset.sizes.get("source", len(self.sources)))

        fields = [
            f"dims={dims}",
            f"backend='{backend}'",
            f"sources={n_sources}",
        ]

        time_repr = self._repr_time_range()
        if time_repr is not None:
            fields.append(f"time='{time_repr}'")

        freq_repr = self._repr_frequency_range()
        if freq_repr is not None:
            fields.append(f"freq='{freq_repr}'")

        created_at = self.provenance.get("created_at") if self.provenance else None
        if created_at:
            fields.append(f"created_at='{created_at}'")

        return f"SimulationResult({', '.join(fields)})"

    def select_source_type(self, source_type: str) -> SimulationResult:
        """Retorna um novo resultado contendo apenas um tipo de fonte.

        A seleção filtra somente a dimensão ``source`` e preserva o restante
        do dataset, a proveniência e os metadados temporais/frequenciais.
        O resultado permanece lazy quando o dataset de origem também é lazy.
        """
        if source_type not in _VALID_SOURCE_TYPES:
            msg = (
                "source_type inválido: "
                f"{source_type!r}. Valores aceitos: {sorted(_VALID_SOURCE_TYPES)}"
            )
            raise ValueError(msg)

        source_type_coord = self.dataset.coords.get("source_type")
        if source_type_coord is None or "source" not in source_type_coord.dims:
            msg = "dataset não possui a coordenada source_type na dimensão source."
            raise ValueError(msg)

        mask = np.asarray(source_type_coord.values == source_type)
        if not bool(mask.any()):
            msg = f"Nenhuma fonte com source_type={source_type!r} existe neste resultado."
            raise ValueError(msg)

        selected_dataset = self.dataset.isel(source=mask)
        selected_sources = [
            src
            for src in self.sources
            if getattr(src, "source_type", getattr(src, "source_type_str", None)) == source_type
        ]
        selected_satellites = [
            src
            for src in self.satellites
            if getattr(src, "source_type", getattr(src, "source_type_str", None)) == source_type
        ]

        return SimulationResult(
            dataset=selected_dataset,
            sources=selected_sources,
            satellites=selected_satellites,
            time_range=self.time_range,
            freq_range=self.freq_range,
            provenance=dict(self.provenance),
        )

    # ── Materialização segura com verificação de memória ───────

    def _estimate_memory_mb(self) -> float:
        """Estima memória necessária para materializar o dataset.

        Calcula o footprint do array T_antenna + power em float64,
        incluindo overhead de coordenadas e metadados (~20%).
        """
        import dask.array as da

        total_bytes = 0
        for var_name in ("T_antenna", "power"):
            var = self.dataset[var_name]
            if isinstance(var.data, da.Array):
                # Para arrays dask: shape total × 8 bytes
                shape = var.shape
                n_elements = 1
                for s in shape:
                    n_elements *= s
                total_bytes += n_elements * 8
            else:
                total_bytes += var.size * 8

        # Overhead xarray + coord (~20%)
        total_bytes *= 1.2
        return float(total_bytes) / (1024 * 1024)

    def compute(self, **kwargs: Any) -> xr.Dataset:
        """Executa dask compute com proteção contra OOM.

        Antes de materializar, estima a memória necessária e compara
        com a memória disponível. Se o dataset exceder a memória RAM
        disponível, emite um aviso ao usuário — a computação ainda
        prossegue, mas o usuário pode abortar se desejar.

        Parameters
        ----------
        **kwargs
            Repassados para ``xr.Dataset.compute()``.

        Returns
        -------
        xr.Dataset
            Dataset materializado (numpy, não mais dask).

        Warns
        -----
        RuntimeWarning
            Se a memória estimada exceder a memória disponível.
        """
        import warnings

        import dask
        import psutil

        est_mb = self._estimate_memory_mb()
        avail_mb = float(psutil.virtual_memory().available) / (1024 * 1024)

        if est_mb > avail_mb:
            msg = (
                f"⚠️  Memória estimada necessária: {est_mb:.0f} MB, "
                f"mas apenas {avail_mb:.0f} MB estão disponíveis. "
                "A computação pode travar ou falhar por OOM. "
                "Considere:\n"
                "  - Reduzir a resolução temporal (TimeRange.resolution)\n"
                "  - Reduzir o número de canais de frequência\n"
                "  - Usar use_processor=True no SimOrchestrator\n"
                "  - Simular um intervalo de tempo menor"
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        # Usa o scheduler atual se já houver um Client distribuído ativo
        # (ex.: ResourceAwareProcessor com dask_scheduler="distributed" — o
        # dashboard do dask acompanha o compute). Caso contrário, garante
        # threads para evitar overhead de distributed.
        if dask.config.get("scheduler", None) == "dask.distributed":
            computed = self.dataset.compute(**kwargs)
        else:
            with dask.config.set(scheduler="threads"):
                computed = self.dataset.compute(**kwargs)

        # Materializa no próprio objeto: o resumo (``summary``/``format_summary``)
        # precisa enxergar o dataset como numpy para preencher as estatísticas de
        # computação (size_mb, memória) e os fluxos/estatísticas de céu. Referências
        # capturadas ANTES do compute (ex.: ``ds_orch``) continuam lazy — sem impacto.
        self.dataset = computed
        return computed

    # ── Persistência ──────────────────────────────────────────

    def _persistence_attrs(self) -> dict[str, Any]:
        """Attrs de persistência gravados no arquivo (não muta o original).

        Inclui a proveniência completa (JSON) e o intervalo temporal/frequência
        para reconstrução fiel no ``load``.
        """
        attrs: dict[str, Any] = dict(self.dataset.attrs)
        try:
            attrs["radioskysim_provenance"] = json.dumps(self.provenance, default=str)
        except Exception:
            pass
        try:
            tr_start, tr_end, tr_res = self.time_range.to_tuple()
            attrs["time_start"] = tr_start.isot
            attrs["time_end"] = tr_end.isot
            attrs["time_step_s"] = float(tr_res)
        except Exception:
            pass
        if self.freq_range is not None:
            attrs["freq_min_mhz"] = float(self.freq_range.min_mhz)
            attrs["freq_max_mhz"] = float(self.freq_range.max_mhz)
            attrs["freq_resolution_mhz"] = float(self.freq_range.resolution_mhz)
        return attrs

    def _dataset_for_persistence(self) -> xr.Dataset:
        """Cópia do dataset com os attrs de persistência (sem tocar o original)."""
        ds = self.dataset.copy(deep=False)
        ds.attrs = self._persistence_attrs()
        return ds

    def _dataset_is_dask(self) -> bool:
        """True se o dataset é lazy (arrays dask) — exige compute antes de gravar."""
        try:
            import dask.array as da
        except Exception:
            return False
        for name in ("T_antenna", "power"):
            var = self.dataset.get(name)
            if var is not None and isinstance(var.data, da.Array):
                return True
        return False

    def _materialize_if_dask(self, compute: bool) -> None:
        """Materializa o dataset lazy (dask) uma única vez antes de persistir.

        Evita a "trava" aparente do compute silencioso de um grafo grande e a
        re-computação em cascata (events, novos saves). O resultado fica
        cacheado em ``self.dataset``.
        """
        if compute and self._dataset_is_dask():
            logger.warning(
                "persist: dataset lazy (dask) será materializado antes da escrita "
                "(~%.0f MB). Cálculo feito uma única vez e reutilizado.",
                self._estimate_memory_mb(),
            )
            self.compute()

    def save(
        self,
        path: str,
        *,
        fmt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Salva o dataset + metadados em HDF5 (NetCDF) ou Zarr.

        O formato é inferido pela extensão do caminho (``.nc``, ``.h5``,
        ``.hdf5`` → HDF5; ``.zarr`` → Zarr) ou pelo argumento ``fmt``.

        Parameters
        ----------
        path:
            Caminho do arquivo (HDF5) ou diretório (Zarr).
        fmt:
            Formato explícito (``"hdf5"``, ``"nc"``, ``"zarr"``). Se omitido,
            é detectado pela extensão; default é HDF5.
        **kwargs:
            Repassados para ``to_hdf5``/``to_zarr``.
        """
        suffix = (fmt or Path(path).suffix.lstrip(".")).lower()
        if suffix == "zarr":
            self.to_zarr(path, **kwargs)
        else:
            self.to_hdf5(path, **kwargs)

    def to_zarr(
        self,
        path: str,
        *,
        compute: bool = True,
        mode: str = "w",
        chunk_s: int | None = None,
        target_mb: float = 64.0,
        zarr_format: int | None = 2,
        source_type: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Persiste o dataset em formato Zarr, respeitando a estrutura de chunks.

        O dataset é rechuncado de forma **uniforme** (tempo em chunks do
        tamanho pedido/capado; demais dimensões em um único chunk — requisito
        do Zarr de chunks uniformes exceto o último) e gravado via
        ``to_zarr(compute=False)`` (delayed do dask): o scheduler escreve cada
        chunk do Zarr como uma tarefa independente, sem carregar o dataset
        completo na memória. Nunca materializa o resultado inteiro.

        Encoding: o padrão é **Zarr v2** (``zarr_format=2``), o formato mais
        portável. O formato 3 da biblioteca ``zarr`` codifica coordenadas
        string (``source``, ``source_type``, ``polarization``) como
        ``StringDType``/``vlen-utf8`` e usa metadados consolidados — elementos
        que **não fazem parte da especificação v3**, gerando warnings e dados
        potencialmente ilegíveis por outras implementações. Com v2 essas
        coordenadas são gravadas com bytes ``|S``/``S`` (interoperável) e sem
        warnings.

        Parameters
        ----------
        path:
            Caminho do diretório Zarr.
        compute:
            Se ``True`` (padrão), executa a escrita por chunks. Se ``False``,
            retorna o delayed (``dask.delayed``) para execução posterior.
        mode:
            Modo de abertura do Zarr (``"w"``, ``"w-"``, ``"a"``).
        chunk_s:
            Nº de amostras de tempo por chunk. Se ``None``, estima de
            ``target_mb``.
        target_mb:
            Tamanho alvo por chunk (MB) quando ``chunk_s`` é ``None``.
        zarr_format:
            Formato Zarr (``2`` ou ``3``). Default ``2`` por portabilidade
            das coordenadas string (veja acima).
        source_type:
            Se fornecido, persiste apenas esse tipo de fonte.
        **kwargs:
            Repassados para ``xr.Dataset.to_zarr()``.

        Returns
        -------
        dask.delayed ou None
        """
        result = self.select_source_type(source_type) if source_type is not None else self
        ds = chunk_uniform(result._dataset_for_persistence(), chunk_s, target_mb=target_mb)
        encoding = dict(kwargs.pop("encoding", {}))
        for name, var in ds.variables.items():
            current = encoding.setdefault(name, {})
            if "chunks" in current:
                continue
            chunks = getattr(var.data, "chunks", None)
            if chunks:
                current["chunks"] = tuple(int(chunk[0]) for chunk in chunks)
            else:
                current["chunks"] = tuple(int(size) for size in var.shape)
        try:
            delayed = ds.to_zarr(
                path,
                compute=compute,
                mode=mode,
                zarr_format=zarr_format,
                encoding=encoding,
                **kwargs,
            )
        except TypeError as exc:
            if "zarr_format" not in str(exc):
                raise
            import zarr

            with zarr.config.set(default_zarr_format=zarr_format or 2):
                delayed = ds.to_zarr(
                    path,
                    compute=compute,
                    mode=mode,
                    encoding=encoding,
                    **kwargs,
                )
        if not compute:
            return delayed
        return None

    def to_hdf5(
        self,
        path: str,
        *,
        compute: bool = True,
        mode: str = "w",
        engine: str | None = None,
        chunk_s: int | None = None,
        target_mb: float = 64.0,
        source_type: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Persiste o dataset em formato NetCDF/HDF5 respeitando chunks.

        O dataset é rechuncado de forma **uniforme** (tempo em chunks do
        tamanho pedido/capado; demais dimensões em um único chunk) e gravado
        via ``to_netcdf(compute=False)`` (delayed do dask): o backend escreve
        cada chunk do array como uma tarefa independente
        (``dask.array.store``), sem carregar o dataset completo na memória.
        Nunca materializa o resultado inteiro.

        O engine padrão é ``"h5netcdf"`` — o backend ``scipy`` (NetCDF3) não
        suporta ``datetime64[ns]``/int64 (o TOD usa precisão de nanossegundos)
        e falha com ``could not safely cast array from int64 to int32``.

        Parameters
        ----------
        path:
            Caminho do arquivo .nc.
        compute:
            Se ``True`` (padrão), executa a escrita por chunks. Se ``False``,
            retorna o delayed (``dask.delayed``) para execução posterior.
        mode:
            Modo de escrita (``"w"`` sobrescreve).
        engine:
            Backend xarray (``"h5netcdf"``, ``"netcdf4"``, ``"scipy"``). Se
            ``None``, usa ``"h5netcdf"``.
        chunk_s:
            Nº de amostras de tempo por chunk. Se ``None``, estima de
            ``target_mb``.
        target_mb:
            Tamanho alvo por chunk (MB) quando ``chunk_s`` é ``None``.
        source_type:
            Se fornecido, persiste apenas esse tipo de fonte.
        **kwargs:
            Repassados para ``xr.Dataset.to_netcdf()``.

        Returns
        -------
        dask.delayed ou None
        """
        if engine is None:
            engine = "h5netcdf"
        path_p = Path(path)
        # mode="w" sobre arquivo existente pode travar (lock residual de outra
        # execução/kernel). Remove o arquivo antes de gravar para evitar o hang.
        if mode == "w" and path_p.exists():
            logger.info("to_hdf5: removendo arquivo existente %s (mode='w')", path_p)
            try:
                path_p.unlink()
            except OSError as exc:
                msg = (
                    f"Não foi possível remover {path_p} para regravar. "
                    "Feche o arquivo em outros processos/kernels e tente de novo: "
                    f"{exc}"
                )
                raise OSError(
                    msg
                ) from exc

        result = self.select_source_type(source_type) if source_type is not None else self
        ds = chunk_uniform(result._dataset_for_persistence(), chunk_s, target_mb=target_mb)
        try:
            delayed = ds.to_netcdf(
                path,
                compute=False,
                mode=mode,
                engine=engine,
                **kwargs,
            )
        except Exception as exc:
            msg = f"to_hdf5 falhou ao gravar {path} (engine={engine}): {exc}"
            raise type(exc)(msg) from exc

        if not compute:
            return delayed

        import dask

        if isinstance(delayed, list | tuple):
            dask.compute(*delayed)
        else:
            dask.compute(delayed)
        return None

    # ── Carga (reconstrução do objeto a partir do disco) ─────

    @classmethod
    def from_hdf5(cls, path: str, **kwargs: Any) -> SimulationResult:
        """Carrega um :class:`SimulationResult` persistido em NetCDF/HDF5.

        Reconstrui ``time_range``/``freq_range``/``provenance`` a partir dos
        attrs gravados no arquivo (fallback: coordenadas do dataset).
        """
        ds = xr.open_dataset(path, **kwargs)
        return cls._from_dataset(ds)

    @classmethod
    def from_zarr(cls, path: str, **kwargs: Any) -> SimulationResult:
        """Carrega um :class:`SimulationResult` persistido em Zarr.

        Reconstrui ``time_range``/``freq_range``/``provenance`` a partir dos
        attrs gravados no diretório (fallback: coordenadas do dataset).
        """
        try:
            ds = xr.open_zarr(path, **kwargs)
        except Exception:
            ds = cls._from_zarr_store(path)
        return cls._from_dataset(ds)

    @classmethod
    def load(cls, path: str, *, fmt: str | None = None, **kwargs: Any) -> SimulationResult:
        """Carrega um :class:`SimulationResult` detectando o formato.

        Extensões ``.zarr`` → Zarr; ``.nc``/``.h5``/``.hdf5`` → HDF5. O formato
        pode ser forçado via ``fmt``.

        Parameters
        ----------
        path:
            Caminho do arquivo (HDF5) ou diretório (Zarr).
        fmt:
            Formato explícito (``"hdf5"``/``"zarr"``). Se omitido, inferido da
            extensão; default é HDF5.
        **kwargs:
            Repassados para ``from_hdf5``/``from_zarr``.
        """
        suffix = (fmt or Path(path).suffix.lstrip(".")).lower()
        if suffix == "zarr":
            return cls.from_zarr(path, **kwargs)
        return cls.from_hdf5(path, **kwargs)

    @classmethod
    def _from_dataset(cls, ds: xr.Dataset) -> SimulationResult:
        """Reconstrói um ``SimulationResult`` a partir de um dataset carregado."""
        attrs = dict(ds.attrs)

        provenance: dict[str, Any] = {}
        prov_raw = attrs.get("radioskysim_provenance")
        if prov_raw:
            try:
                parsed = json.loads(prov_raw)
                if isinstance(parsed, dict):
                    provenance = parsed
            except (ValueError, TypeError):
                provenance = {}

        time_range = _time_range_from_attrs(attrs, ds)

        # ``freq_range`` é derivado pelo ``__post_init__`` a partir da coordenada
        # ``frequency`` (Hz → MHz), garantindo consistência com datasets em Hz.
        return cls(
            dataset=ds,
            sources=[],
            satellites=[],
            time_range=time_range,
            freq_range=None,
            provenance=provenance,
        )

    @staticmethod
    def _from_zarr_store(path: str) -> xr.Dataset:
        """Carrega um store Zarr diretamente quando ``xarray.open_zarr`` falha.

        O fallback cobre stores Zarr v3 escritos pelo backend local e reconstrói
        coordenadas/variáveis a partir dos arrays do grupo e do decode CF.
        """
        import zarr

        root = zarr.open_group(path, mode="r")

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        coords: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

        for name in root.array_keys():
            arr = root[name]
            dims = getattr(getattr(arr, "metadata", None), "dimension_names", None)
            if not dims:
                dims = tuple(arr.attrs.get("_ARRAY_DIMENSIONS", (name,)))
            values = np.asarray(arr[:])
            target = coords if values.ndim <= 1 else data_vars
            target[name] = (tuple(dims), values)

        return xr.decode_cf(xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(root.attrs)))


class Simulation:
    """
    Executor físico de uma simulação multi-fonte.

    Aceita qualquer objeto que implemente o protocolo SignalSource
    (satélites, fontes locais, fontes de rádio).

    Pipeline:
        geometria → beam → TOD → dataset consolidado

    O pointing pode ser especificado em coordenadas equatoriais
    (``pointing_ra_deg``, ``pointing_dec_deg``) ou horizontais
    (``pointing_alt_deg``, ``pointing_az_deg``), ou como uma track
    temporal (``pointing_track``) via Skyfield. RA/Dec é o modo
    preferencial para fontes astronômicas.
    """

    def __init__(
        self,
        sources: Iterable[SignalSource],
        time_range: TimeRange,
        tod_simulator: TODSimulator,
        *,
        pointing_ra_deg: float | None = None,
        pointing_dec_deg: float | None = None,
        pointing_alt_deg: float | None = None,
        pointing_az_deg: float | None = None,
        pointing_track: xr.Dataset | None = None,
        mueller: xr.DataArray | None = None,
        combine_polarization: str = "sum",
        beam_fwhm_deg: float | None = None,
        fwhm_factor: float = 3.0,
        # Novo: SimulationContext (alternativa a parâmetros soltos)
        context: object | None = None,
        # Legacy params (ignored, kept for backward compat)
        geometry: object | None = None,
        signal_factory: object | None = None,
        satellites: Iterable | None = None,
    ) -> None:
        _ = (geometry, signal_factory)  # legado, mantido p/ compatibilidade
        # Se context foi fornecido, extrai parâmetros dele
        if context is not None:
            ctx_sources = getattr(context, "sources", None)
            ctx_tr = getattr(context, "time_range", None)
            ctx_tel = getattr(context, "radiotelescope", None)

            if ctx_sources is not None:
                sources = ctx_sources
            if ctx_tr is not None:
                time_range = ctx_tr

            # Extrai observer e pointing do radiotelescope no context
            if ctx_tel is not None:
                obs = getattr(ctx_tel, "observer", None)
                antennas = getattr(ctx_tel, "antennas", None)
                if antennas and pointing_alt_deg is None and pointing_az_deg is None:
                    # Usa pointing da primeira antena como default
                    ant0 = antennas[0]
                    p = getattr(ant0, "pointing", None)
                    if p is not None:
                        pointing_alt_deg = getattr(p, "alt_deg", pointing_alt_deg)
                        pointing_az_deg = getattr(p, "az_deg", pointing_az_deg)

                # Se context tem observer e pointing alt/az, mas não temos
                # pointing_track, tenta construir pointing_track
                if pointing_track is None and obs is not None and time_range is not None:
                    if pointing_alt_deg is not None and pointing_az_deg is not None:
                        # Cria antena virtual para o pointing track
                        from radioskysims_radiotelescope.domain.radiotelescope.antenna.models import (
                            Antenna,
                            Pointing,
                        )
                        from radioskysims_simulation.domain.simulation.pointing import (
                            build_pointing_track_vectorized,
                        )

                        virtual_antenna = Antenna(
                            name="default",
                            pointing=Pointing(
                                alt_deg=pointing_alt_deg,
                                az_deg=pointing_az_deg,
                            ),
                        )
                        try:
                            pointing_track = build_pointing_track_vectorized(
                                virtual_antenna, time_range, obs
                            )
                        except Exception:
                            pass

            # Ordena fontes por RA (Passo 7)
            sort_fn = getattr(context, "sort_sources_by_ra", None)
            if callable(sort_fn):
                sort_fn()

        self._sources = list(sources)
        self._time_range = time_range
        self._tod_simulator = tod_simulator
        self._pointing_ra_deg = pointing_ra_deg
        self._pointing_dec_deg = pointing_dec_deg
        self._pointing_alt_deg = pointing_alt_deg
        self._pointing_az_deg = pointing_az_deg
        self._pointing_track = pointing_track
        self._mueller = mueller
        self._combine_polarization = combine_polarization
        self._beam_fwhm_deg = beam_fwhm_deg
        self._fwhm_factor = fwhm_factor
        self._context = context

        # Backward compat: se satellites foi passado, usa como sources
        if satellites is not None and not self._sources:
            self._sources = list(satellites)

    @property
    def satellites(self):
        """Retorna fontes do tipo satellite."""
        return [s for s in self._sources if getattr(s, "source_type", "") == "satellite"]

    @property
    def sources(self):
        return self._sources

    @property
    def time_range(self) -> TimeRange:
        return self._time_range

    def run(self) -> SimulationResult:
        """
        Executa a simulação completa.

        Para cada fonte no array, calcula:
        1. Ângulo de separação (theta_boresight) entre pointing e a fonte
        2. Resposta instrumental via TODSimulator
        3. Consolida em xarray.Dataset
        """
        if not self._sources:
            msg = "Simulation requires at least one source."
            raise ValueError(msg)

        # ── Limiar angular (N × FWHM) para descartar fontes fora do feixe ──
        fwhm_threshold_rad: float | None = None
        if self._beam_fwhm_deg is not None and self._beam_fwhm_deg > 0:
            fwhm_threshold_rad = np.deg2rad(self._beam_fwhm_deg * self._fwhm_factor)

        # ── Pointing-first: calcula ângulo de todas as fontes primeiro ──
        boresight_list: list[tuple[str, xr.DataArray | None, SignalSource]] = []
        for idx, source in enumerate(self._sources):
            src_name = getattr(source, "name", None) or f"src_{idx}"
            angle_boresight = self._compute_boresight(source)
            boresight_list.append((src_name, angle_boresight, source))

        # ── Filtra fontes fora do limiar N×FWHM ──
        if fwhm_threshold_rad is not None:
            _filtered: list[tuple[str, xr.DataArray | None, SignalSource]] = []
            n_skipped = 0
            for src_name, angle_boresight, source in boresight_list:
                if angle_boresight is not None:
                    # Se mesmo no ponto mais próximo a fonte está fora do cone
                    # N×FWHM, ela nunca entra no feixe → descarta.
                    min_angle = float(angle_boresight.min().values)
                    if min_angle > fwhm_threshold_rad:
                        n_skipped += 1
                        continue
                _filtered.append((src_name, angle_boresight, source))
            if n_skipped > 0:
                logger = getattr(self._tod_simulator, "_logger", None)
                if logger is not None:
                    logger.info(
                        "sources_filtered_by_beam",
                        total=len(boresight_list),
                        skipped=n_skipped,
                        threshold_deg=self._beam_fwhm_deg * self._fwhm_factor,
                    )
            boresight_list = _filtered

        if not boresight_list:
            msg = "Nenhuma fonte dentro do limiar angular (N×FWHM)."
            raise ValueError(msg)

        _cleaned_tod: list[xr.Dataset] = []
        source_ids: list[str] = []
        angle_list: list[xr.DataArray] = []

        for src_name, angle_boresight, source in boresight_list:
            # Converte source para interface esperada pelo TODSimulator
            sat_adapter = _SourceToTODAdapter(source, angle_boresight)

            ds_src = self._tod_simulator.simulate(
                sat=sat_adapter,
                angle=angle_boresight,
                mueller=self._mueller,
                combine_polarization="separate",
            )

            # Normaliza coordenada time para datetime64[ns] (consistência entre
            # fontes: satélites usam Float64/JD, locais/rádio usam datetime64).
            # Usa o grid UTC canônico para evitar ruído de ponto flutuante na
            # conversão TT→UTC via astropy, que causava duplicação do eixo
            # temporal no xr.concat (13 pts TT-offset + 13 pts UTC exatos).
            if np.issubdtype(ds_src.coords["time"].dtype, np.floating):
                from radioskysims_simulation.application.services.simulation_service import (
                    _compute_time_grid,
                )

                dt_vals = _compute_time_grid(self._time_range)
                ds_src = ds_src.assign_coords(time=dt_vals)
                # Normaliza também o angle_boresight, que herda o time Float64 dos tracks
                if angle_boresight is not None and np.issubdtype(
                    angle_boresight.coords["time"].dtype, np.floating
                ):
                    angle_boresight = angle_boresight.assign_coords(time=dt_vals)

            source_ids.append(src_name)

            _keep = {
                "time",
                "frequency",
                "polarization",
                "pointing",
            }

            _drop = [
                coord for coord in ds_src.coords if coord not in ds_src.dims and coord not in _keep
            ]

            _cleaned_tod.append(ds_src.drop_vars(_drop) if _drop else ds_src)
            if angle_boresight is not None:
                # Garante que angle_boresight não tenha coordenada "source"
                # (alguns tracks de rádio-fontes podem trazê-la, o que quebra o
                # xr.concat abaixo quando misturado com fontes sem essa coord).
                _ab = angle_boresight
                if "source" in _ab.coords and "source" not in _ab.dims:
                    _ab = _ab.reset_coords("source", drop=True)
                elif "source" in _ab.dims:
                    _ab = _ab.isel(source=0, drop=True)
                angle_list.append(_ab)

        # Concatena ao longo de dim "source" (genérica para qualquer tipo de
        # fonte: satélites, locais, rádio). Inclui source_type como coordenada
        # auxiliar para filtragem posterior.
        # NOTA: source_types reflete as fontes EFETIVAMENTE simuladas
        # (após filtro N×FWHM), não a lista original self._sources.

        _simulated_sources = [src for _, _, src in boresight_list]
        source_types = [getattr(source, "source_type", "unknown") for source in _simulated_sources]

        raw = xr.concat(
            _cleaned_tod,
            dim="source",
            compat="override",
            coords="minimal",
            combine_attrs="drop_conflicts",
        )

        T = raw["T_antenna"].reset_coords(drop=True)
        P = raw["power"].reset_coords(drop=True)

        # Cria dataset com coordenadas limpas, incluindo source_type
        dataset = xr.Dataset(
            data_vars={
                "T_antenna": T,
                "power": P,
            },
            coords={
                "time": raw.coords["time"],
                "frequency": raw.coords["frequency"],
                "source": ("source", source_ids),
                "source_type": ("source", source_types),
            },
            attrs=dict(raw.attrs),
        )

        if "polarization" in raw.dims:
            dataset = dataset.assign_coords(polarization=raw.coords["polarization"])
        else:
            dataset["T_antenna"] = dataset["T_antenna"].expand_dims(polarization=["UNDEFINED"])
            dataset["power"] = dataset["power"].expand_dims(polarization=["UNDEFINED"])

        dataset = dataset.expand_dims(pointing=[0])
        # Coordenadas do pointing: RA/Dec primário, alt-az como fallback
        if self._pointing_ra_deg is not None and self._pointing_dec_deg is not None:
            dataset = dataset.assign_coords(
                pointing_ra_deg=("pointing", [float(self._pointing_ra_deg)]),
                pointing_dec_deg=("pointing", [float(self._pointing_dec_deg)]),
            )

        # Força o eixo temporal final a usar o grid canônico do TimeRange.
        # Isso garante que o dataset preserve a resolução solicitada
        # (ex.: 600s) mesmo quando as fontes internas são interpoladas via JD.
        if "time" in dataset.coords:
            dataset = dataset.assign_coords(time=self._time_range.to_datetime64())
        if self._pointing_alt_deg is not None and self._pointing_az_deg is not None:
            dataset = dataset.assign_coords(
                pointing_alt_deg=("pointing", [float(self._pointing_alt_deg)]),
                pointing_az_deg=("pointing", [float(self._pointing_az_deg)]),
            )

        wanted = [
            d
            for d in ("time", "frequency", "polarization", "source", "pointing")
            if d in dataset["T_antenna"].dims
        ]
        dataset["T_antenna"] = dataset["T_antenna"].transpose(*wanted)
        dataset["power"] = dataset["power"].transpose(*wanted)

        if angle_list and len(angle_list) == len(boresight_list):
            angle_boresight = (
                xr.concat(angle_list, dim="source", coords="minimal", compat="override")
                .reset_coords(drop=True)
                .assign_coords(source=("source", source_ids))
            )

            if "time" in angle_boresight.dims and "time" in dataset.coords:
                if angle_boresight.sizes["time"] != dataset.sizes["time"]:
                    msg = (
                        "angle_boresight time dimension does not match "
                        "the simulation dataset time dimension."
                    )
                    raise ValueError(msg)

                angle_boresight = angle_boresight.assign_coords(time=dataset.coords["time"].values)

            dataset = dataset.assign_coords(
                angle_boresight=angle_boresight.expand_dims(pointing=[0])
            )

        # ── Passo 12: garante ordenação temporal apenas quando necessário ──
        if "time" in dataset.coords:
            try:
                time_values = dataset.coords["time"].values

                if time_values.size > 1 and not np.all(time_values[1:] >= time_values[:-1]):
                    dataset = dataset.sortby("time")

            except Exception:
                # Mantém compatibilidade com coordenadas temporais que não
                # suportem comparação vetorizada.
                dataset = dataset.sortby("time")

        tr_start, tr_end, tr_res = self._time_range.to_tuple()

        dataset.attrs.update(
            {
                "n_sources": len(_simulated_sources),
                "n_filtered": (len(self._sources) - len(_simulated_sources)),
                "time_start": tr_start.isot,
                "time_end": tr_end.isot,
                "time_step_s": tr_res,
                "pipeline": "tod",
            }
        )

        if self._beam_fwhm_deg is not None:
            dataset.attrs["beam_fwhm_deg"] = float(self._beam_fwhm_deg)

            dataset.attrs["fwhm_factor"] = float(self._fwhm_factor)

            dataset.attrs["beam_fwhm_threshold_deg"] = float(self._beam_fwhm_deg) * float(
                self._fwhm_factor
            )

        return SimulationResult(
            dataset=dataset,
            sources=_simulated_sources,
            satellites=[
                s
                for s in _simulated_sources
                if getattr(
                    s,
                    "source_type",
                    "",
                )
                == "satellite"
            ],
            time_range=self._time_range,
        )

    def _compute_boresight(self, source: SignalSource) -> xr.DataArray | None:
        """Calcula theta_boresight entre pointing e a fonte.

        Estratégia (por ordem de preferência):
        1. pointing_track (RA/Dec temporal) vs tracks com ra/dec
        2. pointing_ra_deg/pointing_dec_deg fixos vs tracks com ra/dec
        3. Fallback alt-az
        """
        try:
            tracks = source.tracks
        except (AttributeError, ValueError):
            return None

        # 1. Pointing track temporal (RA/Dec) — via Skyfield from_altaz
        if self._pointing_track is not None:
            if "ra" in self._pointing_track and "dec" in self._pointing_track:
                if "ra" in tracks and "dec" in tracks:
                    # Interpola pointing_track no grid temporal da fonte
                    pt_ra = self._pointing_track["ra"].interp(
                        time=tracks.coords["time"],
                        kwargs={"fill_value": "extrapolate"},
                    )
                    pt_dec = self._pointing_track["dec"].interp(
                        time=tracks.coords["time"],
                        kwargs={"fill_value": "extrapolate"},
                    )
                    return compute_theta_boresight(
                        tracks,
                        pointing_track_ra=pt_ra,
                        pointing_track_dec=pt_dec,
                    )

        # 2. Pointing RA/Dec fixo
        if self._pointing_ra_deg is not None and self._pointing_dec_deg is not None:
            if "ra" in tracks and "dec" in tracks:
                return compute_theta_boresight(
                    tracks,
                    pointing_ra_deg=self._pointing_ra_deg,
                    pointing_dec_deg=self._pointing_dec_deg,
                )

        # Fallback alt-az
        if self._pointing_alt_deg is not None and self._pointing_az_deg is not None:
            if "az" in tracks and "el" in tracks:
                return compute_theta_boresight(
                    tracks,
                    pointing_alt_deg=self._pointing_alt_deg,
                    pointing_az_deg=self._pointing_az_deg,
                )

        return None


class _SourceToTODAdapter:
    """Adapta qualquer SignalSource para a interface esperada pelo TODSimulator.

    O TODSimulator espera objetos com métodos:
        - psd() -> xr.DataArray(frequency)
        - angle() -> xr.DataArray(time)
        - range() -> xr.DataArray(time)
    """

    def __init__(
        self,
        source: SignalSource,
        angle_boresight: xr.DataArray | None = None,
    ) -> None:
        self._source = source
        self._angle_boresight = angle_boresight

    def psd(self) -> xr.DataArray:
        return self._source.psd()

    def angle(self) -> xr.DataArray:
        if self._angle_boresight is not None:
            return self._angle_boresight
        return self._source.tracks["theta"]

    def range(self) -> xr.DataArray:
        return self._source.tracks["range"]


# from __future__ import annotations

# import dask.array as da
# import xarray as xr
# from pydantic import BaseModel, ConfigDict

# from radioskysims_core.domain.common.models import TimeRange
# from radioskysims_sources.domain.satellite.factory import SatelliteFactory
# from radioskysims_sources.domain.satellite.models import Satellite, Track


# class SimulationResult(BaseModel):
#     """Resultado da simulação: um Dataset com efemérides de vários satélites."""

#     model_config = ConfigDict(arbitrary_types_allowed=True)
#     dataset: xr.Dataset


# class Simulation:
#     """
#     Orquestrador de simulações de satélites.
#     Recebe satélites de uma factory e agrega efemérides em um único Dataset.
#     """

#     def __init__(self, factory: SatelliteFactory):
#         self.factory = factory
#         self.satellites: list[Satellite] = []
#         self.results: xr.Dataset | None = None

#     def select_by_ids(self, ids: list[int]) -> None:
#         self.satellites = self.factory.create_from_norad_ids(ids)

#     def select_by_constellations(self, constellations: list[str]) -> None:
#         self.satellites = self.factory.create_from_constellations(constellations)

#     def run(self, tr: TimeRange) -> xr.Dataset:
#         """
#         Executa a simulação para o intervalo fornecido.
#         Retorna Dataset com posições e velocidades em uma dimensão `sat`.
#         """

#         positions = []
#         velocities = []
#         sat_names = []

#         for sat in self.satellites:
#             eph: Track = sat.load_ephemeris(tr)

#             # (3, N) → queremos (N, 3)
#             pos = eph.positions.transpose("time", "xyz")
#             vel = (
#                 eph.velocities.transpose("time", "xyz")
#                 if eph.velocities is not None
#                 else None
#             )

#             positions.append(pos.data)  # pega dask array
#             velocities.append(vel.data if vel is not None else da.zeros_like(pos.data))
#             sat_names.append(sat.name)

#         # Empilha em nova dimensão "sat"
#         positions_da = xr.DataArray(
#             da.stack(positions, axis=0),
#             dims=["sat", "time", "xyz"],
#             coords={"sat": sat_names, "time": tr.value, "xyz": ["x", "y", "z"]},
#             attrs={"units": "km"},
#         )

#         velocities_da = xr.DataArray(
#             da.stack(velocities, axis=0),
#             dims=["sat", "time", "xyz"],
#             coords={"sat": sat_names, "time": tr.value, "xyz": ["vx", "vy", "vz"]},
#             attrs={"units": "km/s"},
#         )

#         self.results = xr.Dataset(
#             {
#                 "positions": positions_da,
#                 "velocities": velocities_da,
#             }
#         )
#         return self.results
