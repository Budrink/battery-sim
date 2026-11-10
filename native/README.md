# Исходники C++ / CUDA

Начало чтения: [один шаг симуляции](core/src/simulation/simulation.cpp),
затем [порядок физических стадий](core/src/simulation/mass_step.cuh).
Подробная схема зависимостей — в [архитектуре](../docs/architecture.md).
Для работы с интерфейсом: [карта UI и аналогии с C#](ui/README.md).

```text
native/
├── core/                    вычислительное ядро, без UI и JSON-сценариев
│   ├── include/battery_native/
│   └── src/                 те же подсистемы для реализации
│       ├── model/           общие типы и состояние
│       ├── simulation/      шаг, транзакции, GPU-буферы, наблюдения
│       ├── thermal/         тепловая сетка и теплоперенос
│       ├── electrical/      электрический расчёт и джоулево тепло
│       ├── chemistry/       R1/R2/R3, состав, фазовые превращения
│       ├── mechanics/       оболочки, повреждения, раскрытие
│       ├── gas/
│       │   ├── transport/   массовая сеть, давление, перенос, память связей
│       │   └── mixing/      перемешивание и газ–поверхность
│       └── io/              сохранение и восстановление состояния
├── run/                     загрузка и параметры конкретного запуска
│   ├── src/scenario/        объекты, оболочки, тепловые связи, газовая сеть
│   └── src/contracts/       проверка документа и сборка батарея/опыт → native-сценарий
├── job/                     контракт задач и worker, не физика
├── ui/                      опциональная библиотека ImGui-приложения
│   └── src/                 application/, platform/, viewer/, editor/, shared/
├── visualization/
│   ├── include/battery_native/
│   └── src/
│       ├── data/            геометрия отображения и чтение записей
│       ├── fields/          подготовка отображаемых полей
│       └── render/          OpenGL/CUDA; shaders/ в include
├── apps/                    тонкие точки входа viewer, headless, worker; calibration_monitor
├── tools/                   probes/, gas/, benchmarks/, recording/
├── tests/                   тесты сгруппированы по тем же подсистемам
└── experiments/             изолированные исследовательские реализации
```

В публичных заголовках дополнительно есть `numerics/` (общие численные
ограничители) и `gas/boundary/` (внешняя газовая граница): отдельного `.cpp`
этим законам пока не требуется. Публичные include содержат подсистему:

```cpp
#include "battery_native/simulation/simulation.hpp"
#include "battery_native/gas/transport/gas_direction_memory.hpp"
```

Локальные CUDA-фрагменты `.cuh` — части существующих translation units.
Общий адаптер `CudaGasTransport` не нужно путать с отдельным алгоритмом Kawasaki.
Рабочие структуры — в `gas_work_types.cuh`, зависимости газовых модулей —
в `gas_device_context.cuh`. Порядок стадий и параметры launch не менялись.

Новый тест клади в `tests/<подсистема>/`; диагностический probe — в
`tools/probes/`, его вспомогательный код — в `tools/probes/support/`.
Новые исходники регистрируются в соответствующем `cmake/*.cmake`.
Не добавляй плоские forwarding-заголовки в корень: архитектурный тест
проверяет структуру, границы зависимостей и существование CMake-путей.

Цели сборки и команды запуска прежние: `battery_sim_viewer`,
`battery_sim_headless`, `battery_sim_worker`, `battery_sim_casing_probe` и др.
Preset `server` собирает только runtime/worker/CLI, `core` — только ядро.
Preset `dev` + build `viewer` подключает `battery_ui` и графику.
Публичная точка входа UI — `ui/include/battery_ui/application.hpp`;
API вычисления — `core/include/battery_native/simulation/simulation.hpp`.
JSON-контракт, схемы сценариев и данные опытов не менялись. Worker намеренно
получает новый build fingerprint: строгий контракт не выдаёт другую сборку
за тот же бинарник даже при чистом перемещении исходников.
