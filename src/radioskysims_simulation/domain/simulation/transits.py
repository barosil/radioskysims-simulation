"""
Módulo: src/radioskysims/domain/simulation/transits.py

Extração de trânsitos a partir de um ``SimulationResult`` LAZY (dask), em
chunks ao longo do tempo, SEM materializar o dataset inteiro de uma vez.

Design
------
O dataset ``simulation_result`` (TOD) é sempre muito grande e **nunca** deve
ser computado integralmente na memória. A extração de eventos percorre o eixo
``time`` em chunks (cada chunk computado isoladamente, memória limitada ao
tamanho do chunk) e usa a coordenada ``angle_boresight`` para detectar, por
fonte, os intervalos contíguos em que a fonte está dentro do feixe (<= N x FWHM).
Cada intervalo corresponde a **um trânsito**; o resultado é um dataset pequeno
de trânsitos calculados (``dims=(transit,)``), acompanhado do TOD "in-beam"
(``beam_events``) apenas nos instantes de trânsito.

Chunks
------
- `chunk_time`: rechunca o dataset ao longo de ``time`` (dask) sem computar.
- `time_slices`: slices de tempo derivados dos chunks reais do dataset.
- Trânsitos que cruzam a fronteira entre dois chunks são fundidos (carry-over).

Copyright BINGO Collaboration
Last modified: 2026-08-04
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

# ═══════════════════════════════════════════════════════════════════
# Chunk helpers
# ═══════════════════════════════════════════════════════════════════


def _per_time_sample_bytes(dataset: xr.Dataset) -> int:
    """Bytes de ``T_antenna`` + ``power`` por amostra de tempo."""
    total = 0
    for var in ("T_antenna", "power"):
        if var not in dataset:
            continue
        da_var = dataset[var]
        per = int(np.dtype(da_var.dtype).itemsize)
        for d in da_var.dims:
            if d != "time":
                per *= int(dataset.sizes[d])
        total += per
    return total


def estimate_time_chunk(dataset: xr.Dataset, target_mb: float = 64.0) -> int:
    """Estima quantas amostras de tempo cabem em ~``target_mb`` por chunk.

    Parameters
    ----------
    dataset:
        Dataset (lazy ou não) com ``time``.
    target_mb:
        Tamanho alvo por chunk (em MB) para as variáveis numéricas
        (``T_antenna`` + ``power``).

    Returns
    -------
    int
        Número de amostras de tempo por chunk (>= 1).
    """
    per_sample = _per_time_sample_bytes(dataset)
    n_time = int(dataset.sizes.get("time", 1))
    if per_sample <= 0:
        return max(1, n_time)
    return max(1, min(n_time, int(target_mb * 1024 * 1024 / per_sample)))


def chunk_time(
    dataset: xr.Dataset,
    chunk_s: int | None = None,
    *,
    target_mb: float = 64.0,
) -> xr.Dataset:
    """Rechunca o dataset ao longo de ``time`` (dask) sem materializar.

    Se o dataset já é dask, apenas ajusta os chunks de tempo. Se é numpy,
    envolve em dask (sem cópia) para permitir escrita por chunks.

    ``chunk_s`` é um **limite superior** de amostras por chunk: se o valor
    pedido exceder o orçamento de memória (``target_mb``), ele é reduzido
    para que a escrita nunca materialize o dataset inteiro de uma vez (ex.:
    ``chunk_s=600`` num dataset com apenas ~145 amostras de tempo viraria um
    único chunk e recomputaria o grafo inteiro — o cap evita isso).
    """
    n_time = int(dataset.sizes.get("time", 1))
    # Orçamento de memória por chunk (segurança contra materialização total).
    budget = estimate_time_chunk(dataset, target_mb=target_mb)
    if chunk_s is None:
        chunk_s = budget
    else:
        chunk_s = int(chunk_s)
        chunk_s = min(chunk_s, budget)
    chunk_s = max(1, min(chunk_s, n_time))
    return dataset.chunk({"time": chunk_s})


def chunk_uniform(
    dataset: xr.Dataset,
    chunk_s: int | None = None,
    *,
    target_mb: float = 64.0,
) -> xr.Dataset:
    """Rechunca para escrita em Zarr: tempo uniforme + demais dims em um chunk.

    O Zarr exige **chunks uniformes** (exceto o último) em todas as dimensões.
    O dataset do pipeline pode vir com chunks irregulares em outras dimensões
    (ex.: ``frequency=(258, 123)`` ou ``source=(1, ..., 1, 112, 45)``) — isso
    quebra o ``to_zarr``. Esta função:

    - ``time`` → chunk uniforme de ``chunk_s`` (capado pelo orçamento
      ``target_mb``, como em :func:`chunk_time`);
    - demais dimensões → **um único chunk** (``-1`` = extensão total), sempre
      válido para o Zarr e mantendo a memória limitada ao chunk de tempo.

    Returns
    -------
    xr.Dataset
        Dataset rechuncado (lazy, sem materializar).
    """
    n_time = int(dataset.sizes.get("time", 1))
    budget = estimate_time_chunk(dataset, target_mb=target_mb)
    if chunk_s is None:
        chunk_s = budget
    else:
        chunk_s = int(chunk_s)
        chunk_s = min(chunk_s, budget)
    chunk_s = max(1, min(chunk_s, n_time))
    chunks = {d: -1 for d in dataset["T_antenna"].dims if d != "time"}
    chunks["time"] = chunk_s
    return dataset.chunk(chunks)


def time_slices(dataset: xr.Dataset) -> list[slice]:
    """Slices de tempo derivados dos chunks reais do dataset (na ordem)."""
    n_time = int(dataset.sizes.get("time", 1))
    chunks_t = None
    try:
        chunks_t = dataset["T_antenna"].data.chunks
        dims = dataset["T_antenna"].dims
        chunks_t = chunks_t[dims.index("time")]
    except (AttributeError, KeyError, IndexError, ValueError):
        chunks_t = None

    if chunks_t is None:
        return [slice(0, n_time)]

    slices: list[slice] = []
    start = 0
    for c in chunks_t:
        end = start + int(c)
        slices.append(slice(start, end))
        start = end
    # Garante cobertura completa (defensivo)
    if not slices or slices[-1].stop != n_time:
        slices.append(slice(start, n_time))
    return slices


# ═══════════════════════════════════════════════════════════════════
# Resultado da extração
# ═══════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class TransitsResult:
    """Resultado da extração chunk-wise de trânsitos.

    Attributes
    ----------
    transits:
        Dataset pequeno com ``dims=(transit,)`` — uma linha por trânsito
        (fonte, tipo, início/fim, duração, T_antenna média/máx, ângulo mín).
    beam_events:
        TOD ``(time, frequency, polarization, source, pointing)`` apenas
        nos instantes de trânsito, com ``T_antenna`` zerado fora do feixe.
    objects_in_beam:
        Fontes com pelo menos um trânsito.
    t_antenna_mean:
        Média ponderada (por duração) do T_antenna nos trânsitos [K].
    """

    transits: xr.Dataset
    beam_events: xr.Dataset
    objects_in_beam: list[str]
    t_antenna_mean: float


# ═══════════════════════════════════════════════════════════════════
# Acumulador de trânsito aberto (atravessa fronteiras de chunk)
# ═══════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class _OpenTransit:
    source: str
    source_type: str
    pointing: int
    start_time: np.datetime64
    end_time: np.datetime64
    n: int = 0
    t_sum: float = 0.0
    t_max: float = -np.inf
    angle_min_deg: float = np.inf

    def update(
        self, t_max: float, t_mean: float, n_seg: int, angle_min_deg: float, end_time: np.datetime64
    ) -> None:
        self.n += n_seg
        self.t_sum += t_mean * n_seg
        self.t_max = max(self.t_max, t_max)
        self.angle_min_deg = min(self.angle_min_deg, angle_min_deg)
        self.end_time = end_time

    @property
    def t_mean(self) -> float:
        return float(self.t_sum / self.n) if self.n else 0.0

    @property
    def duration_s(self) -> float:
        dt = self.end_time - self.start_time
        return float(dt / np.timedelta64(1, "s"))


def _row_from_open(o: _OpenTransit) -> dict[str, Any]:
    return {
        "source": o.source,
        "source_type": o.source_type,
        "pointing": o.pointing,
        "start_time": o.start_time,
        "end_time": o.end_time,
        "duration_s": o.duration_s,
        "t_antenna_mean": o.t_mean,
        "t_antenna_max": o.t_max,
        "angle_boresight_min_deg": o.angle_min_deg,
    }


def _finalize(open_t: _OpenTransit | None, rows: list[dict[str, Any]]) -> None:
    if open_t is not None:
        rows.append(_row_from_open(open_t))


# ═══════════════════════════════════════════════════════════════════
# Extração principal
# ═══════════════════════════════════════════════════════════════════


def _angle_threshold_deg(dataset: xr.Dataset, fwhm_threshold_deg: float | None) -> float | None:
    """Limiar angular (graus): argumento explícito ou derivado dos attrs."""
    if fwhm_threshold_deg is not None:
        return float(fwhm_threshold_deg)
    beam_fwhm = dataset.attrs.get("beam_fwhm_deg")
    if beam_fwhm is not None:
        try:
            return float(beam_fwhm) * float(dataset.attrs.get("fwhm_factor", 1.0))
        except Exception:
            return None
    return None


def extract_transits(
    dataset: xr.Dataset,
    *,
    fwhm_threshold_deg: float | None = None,
    source_types: list[str] | None = None,
    chunk_s: int | None = None,
    target_mb: float = 64.0,
) -> TransitsResult:
    """Extrai trânsitos por chunks de tempo, sem materializar o dataset todo.

        Contrato operacional
        --------------------
        - O dataset científico principal pode permanecer lazy (dask) durante toda
            a extração.
        - A função materializa somente os chunks necessários ao longo de ``time``
            (``chunk = ds.isel(time=...).compute()``).
        - Não deve haver ``Dataset.compute()`` sobre o dataset completo nesta rota.
        - ``transits``/``beam_events`` são produtos derivados materializados
            incrementalmente por chunk.

    Parameters
    ----------
    dataset:
        Dataset lazy (dask) ou numpy com ``T_antenna``, ``time``, ``source`` e
        (idealmente) a coordenada ``angle_boresight``.
    fwhm_threshold_deg:
        Limiar angular do feixe (graus). Se ``None``, deriva de
        ``beam_fwhm_deg`` x ``fwhm_factor`` nos attrs.
    source_types:
        Se fornecido, considera apenas estes tipos de fonte.
    chunk_s:
        Nº de amostras de tempo por chunk. Se ``None``, estima de ``target_mb``.
    target_mb:
        Tamanho alvo por chunk (MB) — usado quando ``chunk_s`` é ``None``.

    Returns
    -------
    TransitsResult
        Dataset pequeno de trânsitos + TOD in-beam (``beam_events``).
    """
    ds = chunk_time(dataset, chunk_s, target_mb=target_mb)

    # Filtra tipos de fonte (alinhado ao eixo ``source``).
    if source_types is not None and "source_type" in ds.coords:
        mask_src = np.isin(ds.coords["source_type"].values, source_types)
        ds = ds.sel(source=mask_src)

    has_angle = "angle_boresight" in ds.coords
    threshold_deg = _angle_threshold_deg(ds, fwhm_threshold_deg)
    use_angle = has_angle and threshold_deg is not None

    # Resolve dims uma única vez (ordem pode variar entre fontes).
    t_dims = ds["T_antenna"].dims
    freq_dims = [d for d in t_dims if d in ("frequency", "polarization")]

    rows: list[dict[str, Any]] = []
    open_transits: dict[tuple[str, int], _OpenTransit] = {}
    beam_parts: list[xr.Dataset] = []

    source_names: list[str] = [str(s) for s in ds.coords["source"].values]
    source_type_map: dict[str, str] = (
        {
            str(s): str(t)
            for s, t in zip(
                ds.coords["source"].values, ds.coords["source_type"].values, strict=False
            )
        }
        if "source_type" in ds.coords
        else dict.fromkeys(source_names, "unknown")
    )
    pointings = [int(p) for p in np.atleast_1d(ds.coords["pointing"].values)]

    for sl in time_slices(ds):
        chunk = ds.isel(time=sl)
        # Computa SOMENTE este chunk (memória limitada ao tamanho do chunk).
        cnp = chunk.compute()
        times = cnp.coords["time"].values

        if use_angle:
            angle_da = cnp.coords["angle_boresight"]
            if "pointing" not in angle_da.dims:
                angle_da = angle_da.expand_dims(pointing=pointings)
            in_beam = (angle_da <= np.deg2rad(threshold_deg)).compute()  # (time, source, pointing)
        else:
            # Fallback: fonte "ativa" se T_antenna > 0 em algum canal/pol.
            T_src = cnp["T_antenna"]
            in_beam = T_src.any(dim=freq_dims).compute()  # (time, source, pointing)

        # ── beam_events: instantes com alguma fonte no feixe ──
        active_times = in_beam.any(dim=["source", "pointing"])
        be = cnp.where(in_beam, other=0.0).isel(time=active_times)
        beam_parts.append(be)

        # ── Detecção de trânsitos por (fonte, pointing) ──
        for p in pointings:
            for s_name in source_names:
                key = (s_name, p)
                ib = in_beam.sel(source=s_name, pointing=p).values  # (t_chunk,) bool
                n_t = ib.size
                open_t = open_transits.pop(key, None)

                # Concatena estado aberto (fronteira de chunk anterior) com o
                # bit atual se o primeiro instante continua em trânsito.
                if open_t is not None:
                    lead = 0
                    while lead < n_t and ib[lead]:
                        lead += 1
                    if lead > 0:
                        seg_T = (
                            cnp["T_antenna"]
                            .sel(source=s_name, pointing=p)
                            .isel(time=slice(0, lead))
                        )
                        seg_angle = (
                            angle_da.sel(source=s_name, pointing=p).isel(time=slice(0, lead))
                            if use_angle
                            else None
                        )
                        open_t.update(
                            t_max=float(seg_T.max().values),
                            t_mean=float(seg_T.mean().values),
                            n_seg=lead,
                            angle_min_deg=(
                                float(seg_angle.min().values) if seg_angle is not None else 0.0
                            ),
                            end_time=times[lead - 1],
                        )
                    if lead >= n_t:
                        # Trânsito continua aberto além do chunk.
                        open_transits[key] = open_t
                        continue
                    # Fechou dentro do chunk.
                    _finalize(open_t, rows)
                    open_t = None
                    start_i = lead
                else:
                    start_i = 0

                i = start_i
                while i < n_t:
                    if not ib[i]:
                        i += 1
                        continue
                    j = i
                    while j < n_t and ib[j]:
                        j += 1
                    seg_T = cnp["T_antenna"].sel(source=s_name, pointing=p).isel(time=slice(i, j))
                    seg_angle = (
                        angle_da.sel(source=s_name, pointing=p).isel(time=slice(i, j))
                        if use_angle
                        else None
                    )
                    t_max = float(seg_T.max().values)
                    t_mean = float(seg_T.mean().values)
                    angle_min = float(seg_angle.min().values) if seg_angle is not None else 0.0
                    if j == n_t:
                        # Run chega ao fim do chunk → abre/continua trânsito.
                        open_t = _OpenTransit(
                            source=s_name,
                            source_type=source_type_map.get(s_name, "unknown"),
                            pointing=p,
                            start_time=times[i],
                            end_time=times[j - 1],
                            n=j - i,
                            t_sum=t_mean * (j - i),
                            t_max=t_max,
                            angle_min_deg=angle_min,
                        )
                        open_transits[key] = open_t
                        break
                    rows.append(
                        {
                            "source": s_name,
                            "source_type": source_type_map.get(s_name, "unknown"),
                            "pointing": p,
                            "start_time": times[i],
                            "end_time": times[j - 1],
                            "duration_s": float((times[j - 1] - times[i]) / np.timedelta64(1, "s")),
                            "t_antenna_mean": t_mean,
                            "t_antenna_max": t_max,
                            "angle_boresight_min_deg": angle_min,
                        }
                    )
                    i = j

    # Fecha trânsitos abertos no fim do último chunk.
    for _key, open_t in open_transits.items():
        _finalize(open_t, rows)

    # ── Monta o dataset de trânsitos (pequeno) ─────────────────
    n = len(rows)
    transits = xr.Dataset(
        data_vars={
            "t_antenna_mean": (
                "transit",
                np.asarray([r["t_antenna_mean"] for r in rows], dtype=float),
            ),
            "t_antenna_max": (
                "transit",
                np.asarray([r["t_antenna_max"] for r in rows], dtype=float),
            ),
            "duration_s": ("transit", np.asarray([r["duration_s"] for r in rows], dtype=float)),
        },
        coords={
            "transit": np.arange(n),
            "source": ("transit", np.asarray([r["source"] for r in rows], dtype=object)),
            "source_type": ("transit", np.asarray([r["source_type"] for r in rows], dtype=object)),
            "pointing": ("transit", np.asarray([r["pointing"] for r in rows], dtype=int)),
            "start_time": (
                "transit",
                np.asarray([r["start_time"] for r in rows], dtype="datetime64[ns]"),
            ),
            "end_time": (
                "transit",
                np.asarray([r["end_time"] for r in rows], dtype="datetime64[ns]"),
            ),
            "angle_boresight_min_deg": (
                "transit",
                np.asarray([r["angle_boresight_min_deg"] for r in rows], dtype=float),
            ),
        },
    )

    # ── Resumo ─────────────────────────────────────────────────
    objects_in_beam = sorted({r["source"] for r in rows})
    # Média ponderada pela duração (aprox. do nº de amostras em trânsito).
    if rows:
        weight_sum = sum(max(1.0, r["duration_s"]) for r in rows)
        t_antenna_mean = float(
            sum(r["t_antenna_mean"] * max(1.0, r["duration_s"]) for r in rows) / weight_sum
        )
    else:
        t_antenna_mean = 0.0

    beam_events = xr.concat(beam_parts, dim="time") if beam_parts else _empty_beam_events(ds)

    return TransitsResult(
        transits=transits,
        beam_events=beam_events,
        objects_in_beam=objects_in_beam,
        t_antenna_mean=t_antenna_mean,
    )


def _empty_beam_events(ds: xr.Dataset) -> xr.Dataset:
    """Dataset vazio (sem instantes de trânsito) com o mesmo esquema."""
    return ds.isel(time=slice(0, 0))
