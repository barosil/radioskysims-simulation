"""
domain/rfi/models.py

Entidades de domínio para RFI (Radio Frequency Interference).

A entidade RFI modela fontes de interferência de radiofrequência e fornece
métodos para detectar e filtrar contaminar em Time-Ordered Data (TOD).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

from .exceptions import InvalidRFIError, RFIApplicationError
from .value_objects import FrequencyRange


class RFI:
    """
    Entidade de domínio que representa uma fonte de interferência de radiofrequência.

    Responsabilidades:
    - Armazenar características de interferência
    - Detectar frequências contaminadas
    - Criar máscaras booleanas de contaminar
    - Aplicar filtros a TOD

    Características:
    - Imutável (atributos privados)
    - Sem efeitos colaterais (não modifica TOD original)
    - Validações rigorosas de domínio
    - Sem dependências de infraestrutura

    Exemplo:
        fr1 = FrequencyRange(1.2, 1.4)
        fr2 = FrequencyRange(1.6, 1.8)
        rfi = RFI(frequency_ranges=[fr1, fr2], name="GSM Band")

        is_bad = rfi.is_contaminated(1.3)  # True
        is_good = rfi.is_contaminated(1.5)  # False

        mask = rfi.create_mask(frequencies)
        filtered_tod = rfi.apply_rfi_filter(tod, frequencies)
    """

    __slots__ = ("_description", "_frequency_ranges", "_name")

    def __init__(
        self,
        frequency_ranges: list[FrequencyRange] | set[FrequencyRange],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Cria uma instância de RFI.

        Args:
            frequency_ranges: Lista ou conjunto de FrequencyRange contaminar
            name: Nome opcional (ex: "GSM Band", "WiFi", "Radar")
            description: Descrição opcional da origem

        Raises:
            InvalidRFIError: Se frequency_ranges vazio ou inválido
        """
        self._validate(frequency_ranges)

        # Armazena como tupla imutável, deduplicada
        unique_ranges = self._deduplicate_ranges(frequency_ranges)
        self._frequency_ranges: tuple[FrequencyRange, ...] = tuple(
            sorted(unique_ranges, key=lambda r: r.f_min)
        )

        self._name = name
        self._description = description

    @staticmethod
    def _validate(frequency_ranges: Any) -> None:
        """Valida frequency_ranges."""
        # Tipo
        if not isinstance(frequency_ranges, list | set | tuple):
            msg = f"frequency_ranges deve ser list/set/tuple, não {type(frequency_ranges)}"
            raise InvalidRFIError(msg)

        # Não vazio
        if not frequency_ranges:
            msg = "frequency_ranges não pode estar vazio"
            raise InvalidRFIError(msg)

        # Todos são FrequencyRange
        for item in frequency_ranges:
            if not isinstance(item, FrequencyRange):
                msg = f"Todos os items devem ser FrequencyRange, não {type(item)}"
                raise InvalidRFIError(msg)

    @staticmethod
    def _deduplicate_ranges(
        ranges: list[FrequencyRange] | set[FrequencyRange],
    ) -> list[FrequencyRange]:
        """
        Remove duplicatas e mescla intervalos sobrepostos.

        Args:
            ranges: Intervalos de frequência

        Returns:
            Lista de intervalos não-sobrepostos
        """
        if not ranges:
            return []

        # Ordena por f_min
        sorted_ranges = sorted(ranges, key=lambda r: r.f_min)

        # Mescla sobrepostos
        merged = []
        current = sorted_ranges[0]

        for next_range in sorted_ranges[1:]:
            if current.overlaps_with(next_range):
                # Mescla em um intervalo maior
                current = FrequencyRange(
                    min(current.f_min, next_range.f_min),
                    max(current.f_max, next_range.f_max),
                )
            else:
                merged.append(current)
                current = next_range

        merged.append(current)
        return merged

    # ===================================================================
    # Propriedades Públicas
    # ===================================================================

    @property
    def frequency_ranges(self) -> tuple[FrequencyRange, ...]:
        """Intervalos de frequência contaminados (imutável)."""
        return self._frequency_ranges

    @property
    def name(self) -> str | None:
        """Nome opcional da fonte de RFI."""
        return self._name

    @property
    def description(self) -> str | None:
        """Descrição opcional da origem."""
        return self._description

    @property
    def total_contaminated_bandwidth(self) -> float:
        """Largura de banda total contaminada em GHz."""
        return sum(fr.bandwidth for fr in self._frequency_ranges)

    # ===================================================================
    # Métodos de Domínio
    # ===================================================================

    def is_contaminated(self, frequency: float) -> bool:
        """
        Verifica se uma frequência está contaminada por RFI.

        Args:
            frequency: Frequência em GHz

        Returns:
            True se a frequência está em algum intervalo contaminado
        """
        return any(frequency in fr for fr in self._frequency_ranges)

    def create_mask(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Cria máscara booleana de contaminar para um array de frequências.

        Args:
            frequencies: Array de frequências em GHz [N_freq]

        Returns:
            Array booleano [N_freq] onde True indica contaminar

        Raises:
            RFIApplicationError: Se frequencies inválido
        """
        if not isinstance(frequencies, np.ndarray):
            try:
                frequencies = np.asarray(frequencies)
            except Exception as e:
                msg = f"Não foi possível converter frequencies para ndarray: {e}"
                raise RFIApplicationError(msg) from e

        if frequencies.ndim != 1:
            msg = f"frequencies deve ser 1D, não {frequencies.ndim}D"
            raise RFIApplicationError(msg)

        if frequencies.size == 0:
            msg = "frequencies não pode estar vazio"
            raise RFIApplicationError(msg)

        # Cria máscara: True onde há RFI
        mask = np.zeros(frequencies.size, dtype=bool)
        for fr in self._frequency_ranges:
            mask |= (frequencies >= fr.f_min) & (frequencies <= fr.f_max)

        return mask

    def apply_rfi_filter(
        self,
        tod: Any,
        frequencies: np.ndarray | None = None,
        preserve_original: bool = True,
    ) -> Any:
        """
        Aplica filtro RFI ao Time-Ordered Data (TOD).

        Estratégia: Zera dados contaminados mantendo shape e estrutura.

        Args:
            tod: xarray.Dataset com dimensões (time, frequency, pixel) ou similar
            frequencies: Array de frequências em GHz [N_freq].
                        Se None, tenta extrair de tod.coords['frequency']
            preserve_original: Se True, não modifica tod original (recomendado)

        Returns:
            Dataset filtrado com mesma shape (dados contaminados = 0.0)

        Raises:
            RFIApplicationError: Se tod malformado ou operação falha
        """
        if xr is None:
            msg = "xarray não disponível. Instale com: pip install xarray"
            raise RFIApplicationError(msg)

        if not isinstance(tod, xr.Dataset):
            msg = f"tod deve ser xarray.Dataset, não {type(tod)}"
            raise RFIApplicationError(msg)

        # Extrai frequências
        if frequencies is None:
            if "frequency" not in tod.coords:
                msg = "frequencies não fornecida e 'frequency' não em tod.coords"
                raise RFIApplicationError(msg)
            frequencies = tod.coords["frequency"].values

        # Valida frequências
        if not isinstance(frequencies, np.ndarray):
            try:
                frequencies = np.asarray(frequencies)
            except Exception as e:
                msg = f"Não foi possível extrair frequências: {e}"
                raise RFIApplicationError(msg) from e

        # Cria máscara
        try:
            mask = self.create_mask(frequencies)
        except RFIApplicationError as e:
            msg = f"Falha ao criar máscara RFI: {e}"
            raise RFIApplicationError(msg) from e

        # Copia para não modificar original
        if preserve_original:
            tod = tod.copy(deep=True)

        # Aplica máscara: zera dados onde mask=True
        try:
            # Itera sobre variáveis de dados
            for var_name in tod.data_vars:
                var = tod[var_name]

                # Encontra dimensão de frequência
                if "frequency" in var.dims:
                    # Zera dados contaminados ao longo do eixo frequency
                    var.values[..., mask, :] = 0.0
        except Exception as e:
            msg = f"Erro ao aplicar filtro RFI: {e}"
            raise RFIApplicationError(msg) from e

        return tod

    # ===================================================================
    # Comparação e Representação
    # ===================================================================

    def __eq__(self, other: object) -> bool:
        """Igualdade por valor."""
        if not isinstance(other, RFI):
            return NotImplemented
        return (
            self._frequency_ranges == other._frequency_ranges
            and self._name == other._name
        )

    def __hash__(self) -> int:
        """Hash para usar em conjuntos."""
        return hash((self._frequency_ranges, self._name))

    def __repr__(self) -> str:
        ranges_str = ", ".join(str(r) for r in self._frequency_ranges)
        name_str = f" ({self._name})" if self._name else ""
        return f"RFI[{ranges_str}]{name_str}"

    def __str__(self) -> str:
        ranges_str = ", ".join(str(r) for r in self._frequency_ranges)
        name_str = f": {self._name}" if self._name else ""
        return f"RFI{name_str} contaminadas: {ranges_str}"
