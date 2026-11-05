"""\
Módulo: src/radioskysims/domain/simulation/summary.py

Construção e formatação do resumo estruturado de uma simulação
(Observer, Simulation/Sky, Computation). Alimenta o
``SimulationOrchestratorResult.summary()``/``format_summary()`` e a
serialização JSON persistida no repositório de simulações.

Copyright BINGO Collaboration
Last modified: 2026-08-03
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    from radioskysims_core.domain.common import FrequencyRange, Observer, TimeRange

SUMMARY_VERSION = "1.0"

# Tipos canônicos de fonte. Arquivos antigos podem trazer a coord
# ``source_type`` truncada (ex.: dtype ``<U9`` → "localsource" vira
# "localsour"), então normalizamos por prefixo.
_CANONICAL_SOURCE_TYPES: tuple[str, ...] = ("satellite", "localsource", "radiosource")


def _normalize_source_type(value: Any) -> str:
    """Mapeia ``source_type`` (inclusive truncado) ao rótulo canônico."""
    text = str(value).strip().lower()
    for canonical in _CANONICAL_SOURCE_TYPES:
        if text and canonical.startswith(text):
            return canonical
    return text


# ═══════════════════════════════════════════════════════════════════
# Helpers numéricos / serialização
# ═══════════════════════════════════════════════════════════════════


def _f(value: Any, default: float | None = None) -> float | None:
    """Converte para float seguro, mantendo None e descartando NaN/Inf."""
    if value is None:
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(fv):
        return default
    return fv


def _quantiles(values: Any) -> dict[str, float | None]:
    """min/med/max de um array 1-D, ignorando NaN/Inf."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "med": None, "max": None}
    return {
        "min": float(arr.min()),
        "med": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _to_json_safe(obj: Any) -> Any:
    """Converte estruturas numpy/datetime/astropy para tipos JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if hasattr(obj, "isoformat"):  # astropy Time
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if hasattr(obj, "to_value"):  # astropy Quantity/Angle
        try:
            return float(obj.to_value())
        except Exception:
            return str(obj)
    return obj


def json_safe(summary: dict[str, Any]) -> dict[str, Any]:
    """Retorna o resumo com todos os valores JSON-serializáveis."""
    return _to_json_safe(summary)


# ═══════════════════════════════════════════════════════════════════
# Construção do bloco estático (Observer + Simulation + Computation)
# ═══════════════════════════════════════════════════════════════════


def _observer_block(
    *,
    telescope_name: str,
    observer: Observer,
    frequency_range: FrequencyRange | None,
    num_channels: int | None,
    antennas: list[Any],
    beams: list[Any] | None = None,
) -> dict[str, Any]:
    """Bloco ``observer`` do resumo."""
    beams = beams or []

    # Localização
    lat_deg = lon_deg = height_m = None
    try:
        lat_deg = float(observer.lat.to_value(u.deg))
        lon_deg = float(observer.lon.to_value(u.deg))
    except Exception:
        pass
    try:
        height_m = float(observer.height.to_value(u.m))
    except Exception:
        pass

    # Faixa de frequência + canal central
    fr_cfg: dict[str, Any] = {
        "min_mhz": None,
        "max_mhz": None,
        "n_channels": num_channels,
        "fc_mhz": None,
    }
    if frequency_range is not None:
        fr_cfg = {
            "min_mhz": _f(frequency_range.min_mhz),
            "max_mhz": _f(frequency_range.max_mhz),
            "n_channels": num_channels,
            "fc_mhz": _f((frequency_range.min_mhz + frequency_range.max_mhz) / 2.0),
        }

    # Antenas
    antenna_cfgs: list[dict[str, Any]] = []
    for i, ant in enumerate(antennas):
        pointing = getattr(ant, "pointing", None)
        beam_spec = getattr(ant, "beam", None)

        beam_cfg: dict[str, Any] = {"fwhm_deg": None, "directivity_dbi": None}
        beam_obj = beams[i] if i < len(beams) else None
        if beam_obj is not None:
            try:
                from radioskysims_beam.domain.beam.properties import BeamProperties

                ds_props = BeamProperties.compute_properties(beam_obj.dataset)
                fwhm = _quantiles(np.asarray(ds_props["fwhm"].values, dtype=float))
                dbi = _quantiles(np.asarray(ds_props["directivity_db"].values, dtype=float))
                beam_cfg = {
                    "fwhm_deg": {"min": _f(fwhm["min"]), "max": _f(fwhm["max"])},
                    "directivity_dbi": {"min": _f(dbi["min"]), "max": _f(dbi["max"])},
                }
            except Exception:
                # Fallback: usa params() se disponível
                try:
                    params = beam_obj.params()
                    fwhm = _quantiles(np.asarray(params.fwhm, dtype=float))
                    beam_cfg = {
                        "fwhm_deg": {"min": _f(fwhm["min"]), "max": _f(fwhm["max"])},
                        "directivity_dbi": {
                            "min": _f(params.directivity_db),
                            "max": _f(params.directivity_db),
                        },
                    }
                except Exception:
                    beam_cfg = {"fwhm_deg": None, "directivity_dbi": None}

        antenna_cfgs.append(
            {
                "name": getattr(ant, "name", f"antenna_{i}"),
                "pointing": {
                    "alt_deg": _f(getattr(pointing, "alt_deg", None)),
                    "az_deg": _f(getattr(pointing, "az_deg", None)),
                },
                "polarizations": list(getattr(beam_spec, "polarizations", None) or []),
                "beam": beam_cfg,
            }
        )

    return {
        "name": telescope_name,
        "location": {"lat_deg": lat_deg, "lon_deg": lon_deg, "height_m": height_m},
        "frequency_range": fr_cfg,
        "antennas": antenna_cfgs,
    }


def _satellites_block(satellites: list[Any]) -> dict[str, Any]:
    """Bloco ``sky.satellites`` agrupado por constelação."""
    grouped: dict[str, dict[str, Any]] = {}
    for sat in satellites:
        const = getattr(sat, "constellation", None) or "unknown"
        entry = grouped.setdefault(
            const,
            {
                "name": const,
                "n_satellites": 0,
                "ids": [],
                "tles": [],
                "epochs": [],
            },
        )
        entry["n_satellites"] += 1

        norad = getattr(sat, "norad_id", None)
        if norad is not None:
            entry["ids"].append(int(norad))

        element = getattr(sat, "element", None)
        tle1 = getattr(element, "tle_line1", None)
        tle2 = getattr(element, "tle_line2", None)
        if tle1 and tle2:
            entry["tles"].append({"line1": tle1, "line2": tle2})

        epoch = getattr(sat, "epoch", None)
        if epoch is not None:
            iso = epoch.isoformat() if hasattr(epoch, "isoformat") else str(epoch)
            entry["epochs"].append(iso)

    for entry in grouped.values():
        epochs = []
        for iso in entry.pop("epochs", []):
            try:
                epochs.append(datetime.fromisoformat(iso))
            except ValueError:
                continue
        if epochs:
            mean_ts = sum(e.timestamp() for e in epochs) / len(epochs)
            entry["tle_mean_epoch"] = datetime.fromtimestamp(mean_ts, tz=UTC).isoformat()
        else:
            entry["tle_mean_epoch"] = None

    return grouped


def assemble_config(
    *,
    telescope_name: str,
    observer: Observer,
    frequency_range: FrequencyRange | None,
    antennas: list[Any],
    beams: list[Any] | None = None,
    num_channels: int | None = None,
    time_range: TimeRange,
    satellites: list[Any],
    include_satellites: bool = True,
    include_local_sources: bool = False,
    include_radio_sources: bool = False,
    radio_query: dict[str, Any] | None = None,
    start_utc: str | None = None,
    computing_time_s: float | None = None,
    products: list[str] | None = None,
    dimensions: dict[str, int] | None = None,
    chunks: dict[str, Any] | None = None,
    size_mb: float | None = None,
    delayed: bool = True,
) -> dict[str, Any]:
    """Monta o dicionário de configuração estático de uma simulação.

    Preenche os blocos ``observer``, ``simulation`` e ``computation``
    conforme o escopo definido no TODO (dados do observador, céu e
    computação). Os campos que dependem de materialização do dataset
    (ex.: estatísticas de T_antenna) ficam ``None`` e são preenchidos
    por :func:`populate_sky_stats` / :func:`populate_computation`.
    """
    # ── Intervalo temporal ────────────────────────────────────
    tr_start, tr_end, tr_res = time_range.to_tuple()
    time_cfg = {
        "start": tr_start.isot,
        "end": tr_end.isot,
        "step_sec": tr_res,
    }

    # ── Céu ───────────────────────────────────────────────────
    sky: dict[str, Any] = {
        "localsources": {},
        "radiosources": {},
        "satellites": _satellites_block(satellites) if include_satellites else {},
    }
    if include_satellites:
        sky["satellites_enabled"] = True
    else:
        sky["satellites_enabled"] = False
    if include_local_sources:
        sky["localsources"]["Sun"] = {
            "flux_jy": None,
            "t_antenna_k": None,
            "datafile": None,
        }
    if include_radio_sources:
        sky["radiosources"] = {
            "catalog": (radio_query or {}).get("catalog", "NVSS"),
            "query": {
                "ra_center_deg": (radio_query or {}).get("ra_center_deg"),
                "dec_center_deg": (radio_query or {}).get("dec_center_deg"),
                "search_radius_deg": (radio_query or {}).get("search_radius_deg"),
                "min_flux_jy": (radio_query or {}).get("min_flux_jy"),
            },
            "t_antenna_k": None,
        }

    # ── Computação ────────────────────────────────────────────
    computation: dict[str, Any] = {
        "start": start_utc,
        "computing_time_s": computing_time_s,
        "products": products or ["T_antenna", "power"],
        "dimensions": dimensions or {},
        "chunks": chunks or {},
        "size_mb": size_mb,
        "delayed": delayed,
        "memory": {"loaded_mb": None, "unloaded_mb": None},
        "saved": False,
        "registered": False,
        "filepath": None,
    }

    return {
        "version": SUMMARY_VERSION,
        "observer": _observer_block(
            telescope_name=telescope_name,
            observer=observer,
            frequency_range=frequency_range,
            num_channels=num_channels,
            antennas=antennas,
            beams=beams,
        ),
        "simulation": {
            "time_range": time_cfg,
            "sky": sky,
        },
        "computation": computation,
    }


# ═══════════════════════════════════════════════════════════════════
# Preenchimento de campos dependentes do dataset
# ═══════════════════════════════════════════════════════════════════


def populate_computation(result: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Preenche o bloco ``computation`` a partir do dataset (lazy ou não)."""
    ds = getattr(result, "dataset", None)
    if ds is None:
        return config

    comp = config["computation"]
    comp["products"] = [str(v) for v in ds.data_vars]
    comp["dimensions"] = {k: int(v) for k, v in ds.sizes.items()}

    T = ds.get("T_antenna")
    is_dask = False
    try:
        import dask.array as da

        is_dask = T is not None and isinstance(T.data, da.Array)
    except ImportError:
        is_dask = False

    comp["delayed"] = is_dask
    size_mb = _f(ds.nbytes / (1024 * 1024)) if not is_dask else None
    comp["size_mb"] = size_mb

    if is_dask and T is not None:
        comp["chunks"] = {d: T.data.chunks[i] for i, d in enumerate(T.dims)}
        # Estimativa de memória para materializar
        n_elems = 1
        for s in T.shape:
            n_elems *= s
        comp["memory"] = {
            "loaded_mb": 0.0,
            "unloaded_mb": _f(n_elems * 8 / (1024 * 1024)),
        }
    else:
        comp["chunks"] = "numpy"
        comp["memory"] = {
            "loaded_mb": size_mb,
            "unloaded_mb": size_mb,
        }

    return config


def populate_sky_stats(result: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Preenche estatísticas de céu (T_antenna por tipo, fluxo do Sol).

    As estatísticas são computadas por FATIA (source a source), então
    funcionam com dataset lazy (dask): apenas o(s) source(s) local/radio são
    materializados, nunca o TOD inteiro.
    """
    ds = getattr(result, "dataset", None)
    if ds is None or "T_antenna" not in ds:
        return config

    T = ds["T_antenna"]

    sky = config["simulation"]["sky"]

    if "source" in T.dims:
        source_names = ds.coords["source"].values
        has_types = "source_type" in ds.coords
        source_types = (
            [_normalize_source_type(t) for t in ds.coords["source_type"].values]
            if has_types
            else []
        )

        # Colapsa polarization/pointing mantendo (time, frequency, source)
        try:
            T_src = T.mean(dim=["polarization", "pointing"], skipna=True)
        except Exception:
            T_src = T

        # Agrega por tipo (quando a coordenada existe)
        if has_types:
            for stype, key in (
                ("localsource", "localsources"),
                ("radiosource", "radiosources"),
            ):
                mask = np.array([t == stype for t in source_types])
                if not mask.any():
                    continue
                idx = np.where(mask)[0]
                block = sky[key]
                if not isinstance(block, dict) or not block:
                    continue
                if key == "radiosources":
                    block["t_antenna_k"] = _quantiles(
                        np.asarray(T_src.isel(source=idx).values, dtype=float)
                    )
                else:
                    # localsources: bloco chaveado por nome de fonte (ex.: Sun) —
                    # preenche a estatística dentro de cada entrada de fonte.
                    for j in idx:
                        sname = str(source_names[j])
                        entry = block.get(sname)
                        if isinstance(entry, dict):
                            entry["t_antenna_k"] = _quantiles(
                                np.asarray(T_src.isel(source=j).values, dtype=float)
                            )

        # Fallback por NOME: garante a estatística de cada fonte local (ex.:
        # Sun) mesmo com ``source_type`` truncado/ausente.
        if isinstance(sky.get("localsources"), dict) and sky["localsources"]:
            name_to_idx = {str(n): j for j, n in enumerate(source_names)}
            for sname, entry in sky["localsources"].items():
                if isinstance(entry, dict) and entry.get("t_antenna_k") is None:
                    j = name_to_idx.get(str(sname))
                    if j is not None:
                        entry["t_antenna_k"] = _quantiles(
                            np.asarray(T_src.isel(source=j).values, dtype=float)
                        )

        # Sol: fluxo @ fc a partir do psd da fonte
        if sky["localsources"]:
            for sname, block in sky["localsources"].items():
                block = block if isinstance(block, dict) else {}
                if block.get("flux_jy") is None:
                    src = next(
                        (
                            s
                            for s in getattr(result, "sources", [])
                            if getattr(s, "name", None) == sname
                        ),
                        None,
                    )
                    if src is not None:
                        try:
                            psd = src.psd()
                            fc_hz = config["observer"]["frequency_range"]["fc_mhz"]
                            if fc_hz is not None:
                                fc_hz = float(fc_hz) * 1e6
                                flux_w = float(
                                    np.interp(
                                        fc_hz,
                                        np.asarray(psd.frequency.values, dtype=float),
                                        np.asarray(psd.values, dtype=float),
                                    )
                                )
                                block["flux_jy"] = {"value_jy": _f(flux_w / 1e-26)}
                        except Exception as exc:
                            _LOGGER.warning(
                                "Falha ao avaliar fluxo de '%s' no resumo: %s: %s",
                                sname,
                                type(exc).__name__,
                                exc,
                            )
                    else:
                        _LOGGER.warning(
                            "Fonte local '%s' não encontrada em result.sources "
                            "(%d fontes) — flux_jy ficará nulo no resumo.",
                            sname,
                            len(getattr(result, "sources", []) or []),
                        )

                # datafile: proveniência
                if block.get("datafile") is None:
                    for local in getattr(result, "provenance", {}).get("local_sources", []):
                        if local.get("name") == sname and local.get("source_file"):
                            block["datafile"] = local["source_file"]

    return config


# ═══════════════════════════════════════════════════════════════════
# Formatação legível
# ═══════════════════════════════════════════════════════════════════


def _fmt_ql(q: dict[str, Any] | None) -> str:
    """Formata {min, med, max}."""
    if not q:
        return "—"
    return f"min {_fmt(q.get('min'))} / med {_fmt(q.get('med'))} / max {_fmt(q.get('max'))}"


def _fmt_flux(v: Any) -> str:
    """Formata o fluxo de uma fonte (dict ``{value_jy}``, quantil ou escalar)."""
    if not v:
        return "—"
    if isinstance(v, dict):
        if "value_jy" in v:
            return f"{_fmt(v.get('value_jy'))} Jy"
        return _fmt_ql(v)
    return f"{_fmt(v)} Jy"


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _fmt_deg(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_mb(v: Any) -> str:
    if v is None:
        return "—"
    return f"{float(v):.1f} MB"


def format_summary(summary: dict[str, Any]) -> str:
    """Renderiza o resumo num texto formatado e legível."""
    obs = summary.get("observer", {})
    sim = summary.get("simulation", {})
    comp = summary.get("computation", {})

    lines: list[str] = []
    lines.append("═" * 72)
    lines.append("  RADIOSKYSIM — RESUMO DA SIMULAÇÃO")
    lines.append("═" * 72)

    # ── Observer ─────────────────────────────────────────────
    loc = obs.get("location", {})
    fr = obs.get("frequency_range", {})
    lines.append("")
    lines.append("┌─ OBSERVER ──────────────────────────────────────────────")
    lines.append(f"  Nome                    : {obs.get('name', '—')}")
    lines.append(
        f"  Localização (lat/lon)    : {_fmt_deg(loc.get('lat_deg'))} / "
        f"{_fmt_deg(loc.get('lon_deg'))}  (height {_fmt(loc.get('height_m'))} m)"
    )
    lines.append(
        f"  Frequency Range (MHz)    : {_fmt(fr.get('min_mhz'))} → "
        f"{_fmt(fr.get('max_mhz'))}  (n_channels={fr.get('n_channels', '—')})"
    )
    lines.append(f"  Central Frequency (MHz) : {_fmt(fr.get('fc_mhz'))}")

    for ant in obs.get("antennas", []):
        pt = ant.get("pointing", {})
        beam = ant.get("beam", {})
        lines.append(f"  Antenna                 : {ant.get('name', '—')}")
        lines.append(
            f"    Pointing (Alt/Az)      : {_fmt_deg(pt.get('alt_deg'))}° / "
            f"{_fmt_deg(pt.get('az_deg'))}°"
        )
        pols = ant.get("polarizations") or []
        lines.append(f"    Polarizations         : {pols if pols else '—'}")
        lines.append(f"    Beam FWHM (deg)       : {_fmt_ql(beam.get('fwhm_deg'))}")
        lines.append(f"    Directivity (dBi)     : {_fmt_ql(beam.get('directivity_dbi'))}")

    # ── Simulation ────────────────────────────────────────────
    tr = sim.get("time_range", {})
    sky = sim.get("sky", {})
    lines.append("")
    lines.append("┌─ SIMULATION ─────────────────────────────────────────────")
    lines.append(
        f"  Timerange (UTC)         : {tr.get('start', '—')} → "
        f"{tr.get('end', '—')}  (step {_fmt(tr.get('step_sec'))} s)"
    )

    local_sky = sky.get("localsources", {})
    if local_sky:
        lines.append("  Sky ─ Localsources:")
        for sname, sblock in local_sky.items():
            sblock = sblock if isinstance(sblock, dict) else {}
            lines.append(f"    - {sname}")
            lines.append(f"        Flux@fc (Jy)          : {_fmt_flux(sblock.get('flux_jy'))}")
            lines.append(f"        Antenna Temp (K)      : {_fmt_ql(sblock.get('t_antenna_k'))}")
            lines.append(f"        flux datafile        : {sblock.get('datafile') or '—'}")

    radio_sky = sky.get("radiosources", {})
    if radio_sky:
        rq = radio_sky.get("query", {})
        lines.append("  Sky ─ Radiosources:")
        lines.append(
            f"    - Catalog: {radio_sky.get('catalog', '—')}  "
            f"(ra {_fmt_deg(rq.get('ra_center_deg'))}°, dec {_fmt_deg(rq.get('dec_center_deg'))}°, "
            f"radius {_fmt_deg(rq.get('search_radius_deg'))}°)"
        )
        lines.append(f"        Antenna Temp (K)      : {_fmt_ql(radio_sky.get('t_antenna_k'))}")

    sat_sky = sky.get("satellites", {})
    if sat_sky:
        lines.append("  Sky ─ Satellites:")
        for const, sblock in sat_sky.items():
            lines.append(f"    - {const}: {sblock.get('n_satellites', 0)} satélites")
            ids = sblock.get("ids", [])
            if ids:
                lines.append(
                    f"        ids                : {ids[:12]}{' …' if len(ids) > 12 else ''}"
                )
            n_tles = len(sblock.get("tles", []))
            lines.append(f"        TLEs               : {n_tles}")
            lines.append(f"        tle mean epoch     : {sblock.get('tle_mean_epoch') or '—'}")

    # ── Computation ───────────────────────────────────────────
    dims = comp.get("dimensions", {})
    dims_str = ", ".join(f"{k}={v}" for k, v in dims.items()) or "—"
    mem = comp.get("memory", {})
    lines.append("")
    lines.append("┌─ COMPUTATION ───────────────────────────────────────────")
    lines.append(f"  Start                   : {comp.get('start') or '—'}")
    lines.append(f"  Computing time          : {_fmt(comp.get('computing_time_s'))} s")
    lines.append(f"  Products                : {comp.get('products') or '—'}")
    lines.append(f"  Dataset dimensions      : {dims_str}")
    lines.append(f"  Dataset chunks          : {comp.get('chunks') or '—'}")
    lines.append(f"  Dataset size            : {_fmt_mb(comp.get('size_mb'))}")
    lines.append(
        f"  Delayed / Computed      : {'delayed (dask)' if comp.get('delayed') else 'computed (numpy)'}"
    )
    lines.append(
        f"  Memory loaded/unloaded  : {_fmt_mb(mem.get('loaded_mb'))} / "
        f"{_fmt_mb(mem.get('unloaded_mb'))}"
    )
    lines.append(f"  Saved / Registered      : {comp.get('saved')} / {comp.get('registered')}")
    lines.append(f"  Filepath                : {comp.get('filepath') or '—'}")

    lines.append("")
    lines.append("═" * 72)
    return "\n".join(lines)


def to_json(summary: dict[str, Any], **kwargs: Any) -> str:
    """Serializa o resumo em JSON (indentado por padrão)."""
    kwargs.setdefault("indent", 2)
    return json.dumps(json_safe(summary), **kwargs)
