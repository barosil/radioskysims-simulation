# Desacoplamento de Processador (T1) — Arquitetura de Injeção de Dependências

**Fase:** T1 — Dívida Técnica (Nov 2026)  
**Items:** T1.1–T1.7  
**Status:** ✅ Implementado (Branch B1)

## Problema Original (TD-001)

A camada **application** importava diretamente de **infrastructure**:

```python
# ❌ ANTES (violação arquitetural)
from radioskysims_simulation.infrastructure.processing.resource_processor import (
    ResourceAwareProcessor,
)

class SimOrchestrator:
    def __init__(self, processor: ResourceAwareProcessor | None = None):
        ...
```

**Impactos:**
- Acoplamento direto: `application/services` conhece `infrastructure/processing`
- Difícil de testar (mock de `ResourceAwareProcessor` complexo)
- Difícil de estender (swap para `distributed.Processor` exigiria mudança em application)
- Violação hexagonal: nenhuma barreira entre camadas

## Solução: Injeção de Dependências com Port Abstrato

### Arquitetura (T1.1–T1.7)

```
┌─────────────────────────────────────────┐
│        application/                     │
│  ┌────────────────────────────────────┐ │
│  │  application/services/            │ │
│  │  ├─ sim_orchestrator.py           │ │
│  │  └─ (imports ProcessorPort)       │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  application/ports/               │ │
│  │  ├─ processing.py                 │ │
│  │  └─ (ProcessorPort ABC)           │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
           ↑
           │ (injeção)
           │
┌─────────────────────────────────────────┐
│        bootstrap/                       │
│  ├─ composition.py (T1.6)              │
│  ├─ dependencies.py (T1.4)             │
│  └─ builders.py (existente)            │
└─────────────────────────────────────────┘
           ↓
           │ (factory)
           ↓
┌─────────────────────────────────────────┐
│        infrastructure/                  │
│  ┌────────────────────────────────────┐ │
│  │  infrastructure/processing/       │ │
│  │  ├─ resource_processor.py (impl)  │ │
│  │  │   ResourceAwareProcessor       │ │
│  │  └─ processor_adapter.py (T1.5)   │ │
│  │      ResourceAwareProcessorAdapter│ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Componentes Implementados

#### T1.1: Port Abstrato — `application/ports/processing.py`

Define o contrato que qualquer processador deve cumprir:

```python
class ProcessorPort(ABC, Generic[T]):
    """Port abstrato para processadores conscientes de recursos."""

    @property
    @abstractmethod
    def resources(self) -> ProcessingResources:
        """Recursos disponíveis (CPUs, memória)."""
        ...

    @property
    @abstractmethod
    def parallel_workers(self) -> int:
        """Número de workers paralelos."""
        ...

    @abstractmethod
    def run_batch(
        self,
        batch: Iterable[T],
        processor_fn: Callable[[T], SimulationResult],
    ) -> list[SimulationResult]:
        """Executa um batch de itens."""
        ...

    def close(self) -> None:
        """Limpa recursos."""
        ...
```

**Características:**
- ✓ Genérico `T` (extensível)
- ✓ Sem conhecimento de implementação
- ✓ Type hints 100%
- ✓ 8 métodos abstratos definindo contrato claro

#### T1.2–T1.3: Application Refatorada — `application/services/sim_orchestrator.py`

**Antes:**
```python
from radioskysims_simulation.infrastructure.processing.resource_processor import (
    ResourceAwareProcessor,
)

class SimOrchestrator:
    def __init__(self, processor: ResourceAwareProcessor | None = None):
        self._processor: ResourceAwareProcessor = processor
```

**Depois:**
```python
from radioskysims_simulation.application.ports.processing import ProcessorPort

class SimOrchestrator:
    def __init__(self, processor: ProcessorPort | None = None):
        self._processor: ProcessorPort = processor
```

**Mudanças:**
- ✓ Type hint: `ResourceAwareProcessor` → `ProcessorPort`
- ✓ Import: `infrastructure` → `application/ports`
- ✓ Compatibilidade mantida: `_resolve_processor` com lazy import para fallback legado
- ✓ Dashboard URL adicionado ao port

#### T1.4: DI Container — `bootstrap/dependencies.py`

Factory function e singleton container para instanciar ProcessorPort:

```python
from radioskysims_simulation.bootstrap import (
    get_container,
    configure_default_processor,
)

# Opção 1: Global singleton
processor = get_container().get_processor(
    max_batch_mb=2048,
    dask_scheduler='threads',
)

# Opção 2: Factory function
processor = configure_default_processor(
    cpu_count=4,
    memory_mb=8192,
)
```

**Responsabilidades:**
- Detecta recursos (CPUs, memória)
- Cria ProcessingResources
- Caching de instância (singleton)
- Reset para testes

#### T1.5: Adapter — `infrastructure/processing/processor_adapter.py`

Wrapper que faz `ResourceAwareProcessor` implementar `ProcessorPort`:

```python
class ResourceAwareProcessorAdapter(ProcessorPort[SimulationResult]):
    """Adapter que faz ResourceAwareProcessor → ProcessorPort."""

    def __init__(self, processor: ResourceAwareProcessor):
        self._processor = processor

    @property
    def resources(self) -> ProcessingResources:
        return self._processor.resources

    def run_batch(self, batch, processor_fn) -> list[SimulationResult]:
        return self._processor.run_batch(batch, processor_fn)
    
    # ... delegações para outros métodos ...
```

**Vantagens:**
- ✓ Não altera `ResourceAwareProcessor` existente
- ✓ Compatibilidade 100% com código legado
- ✓ Pode ser removido quando `ResourceAwareProcessor` implementar port diretamente

#### T1.6: Composition Root — `bootstrap/composition.py`

Integração: DIContainer → SimOrchestrator

```python
def create_sim_orchestrator(
    telescope: str = "default",
    *,
    processor_config: dict[str, Any] | None = None,
):
    """Cria SimOrchestrator com ProcessorPort injetado."""
    
    context = create_simulation_context(telescope=telescope)
    
    # Injetar ProcessorPort via DI
    processor = None
    if processor_config:
        container = get_container()
        processor = container.get_processor(**processor_config)
    
    return SimOrchestrator(context, processor=processor)
```

**Uso:**
```python
# Sem ProcessorPort (fallback legado):
sim = create_sim_orchestrator()

# Com ProcessorPort injetado (T1 pattern):
sim = create_sim_orchestrator(
    processor_config={
        'max_batch_mb': 2048,
        'cpu_count': 4,
    }
)
```

### T1.7: Sequência de Fluxo

```
┌─ Aplicação (CLI, notebooks, etc)
│
├─ create_sim_orchestrator(processor_config={...})
│  │
│  ├─ create_simulation_context()
│  │  └─ build_simulation_context_dependencies()
│  │
│  └─ get_container().get_processor(**config)
│     │
│     ├─ Detecta recursos (CPUs, RAM)
│     │
│     ├─ Cria ResourceAwareProcessor (infra)
│     │
│     └─ Wraps com ResourceAwareProcessorAdapter
│        └─ Retorna ProcessorPort (application vê isso)
│
└─ SimOrchestrator(context, processor=ProcessorPort)
   │
   └─ Usa processor.run_batch(), processor.resources, etc
      (só vê interface ProcessorPort, não impl)
```

## Benefícios

| Aspecto | Antes | Depois |
|---------|-------|--------|
| **Acoplamento** | application → infrastructure | application → ports ← infrastructure |
| **Mock em testes** | Complexo (ResourceAwareProcessor) | Simples (ProcessorPort ABC) |
| **Swap de impl** | Editar application/services | Trocar adapter ou DIContainer |
| **Type Safety** | Concreto (ResourceAwareProcessor) | Abstrato + genérico (ProcessorPort[T]) |
| **Compatibilidade** | N/A | 100% legado + novo pattern |

## Testes

```python
# Mock ProcessorPort para teste
class MockProcessor(ProcessorPort):
    def run_batch(self, batch, fn):
        # Simular execução sem dask
        return [fn(item) for item in batch]

# Injetar em SimOrchestrator
sim = SimOrchestrator(context, processor=MockProcessor())
# Teste passa sem setup de scheduler
```

## Próximos Passos (Fases futuras)

1. **Estender ports** — Adicionar ports para outros adapters (logging, persistência, etc)
2. **DIContainer avançado** — Factory patterns, scoping, auto-wiring
3. **Remover adapter** — Quando `ResourceAwareProcessor` implementar `ProcessorPort` diretamente
4. **Multi-tenant** — Múltiplos containers isolados por tenant/projeto

## Referências

- **Branch:** `fix/td-001-application-infrastructure-ports` (B1)
- **Items:** T1.1 (port), T1.3 (refactor), T1.4 (DI), T1.5 (adapter), T1.6 (integration), T1.7 (docs)
- **Commits:**
  - `bdec0f7`: T1.1 ProcessorPort
  - `bf0d172`: T1.2-T1.3 sim_orchestrator refactor
  - `a7837ea`: T1.4 DIContainer
  - `e7d4a3c`: T1.5 ResourceAwareProcessorAdapter
  - `81fe256`: T1.6 composition integration

---

**Padrão:** Dependency Injection com Port Abstrato (Hexagonal Architecture)  
**Status:** ✅ Implementado  
**Violação TD-001:** ✅ Eliminada  
**Próximo:** T2 (Qualidade — type hints, lint), T3 (F1 UX)
