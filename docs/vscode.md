# Работа в VS Code (Linux + NVIDIA)

Открой `battery-sim.code-workspace` через File → Open Workspace from File
или командой `code battery-sim.code-workspace`. При запросе разреши доверие
этому репозиторию и установи рекомендованные CMake Tools и C/C++.

- **Ctrl+Shift+B** — настроить и собрать визуализатор и монитор калибровки.
- **F5** — собрать и запустить выбранную конфигурацию из Run and Debug.
- **Ctrl+F5** — то же без отладчика.
- Первая конфигурация `laboratory editor` открывает окно лаборатории на
  синтетическом шаблоне, не на экспериментальной калибровке.
- `cascade replay` и `exhaust replay` показывают сохранённые траектории с 1700 с.
  Они не запускают новый расчёт. Space — пауза.
- `debug viewer` — не оптимизированный host C++, удобный для точек останова.
  CUDA device-код обычный GDB не отлаживает; для этого нужен отдельный CUDA debugger.
- `headless smoke test` — короткая проверка CUDA без OpenGL и интерфейса.
- Tasks → `BatterySim: build server` — worker/CLI без графических зависимостей.
- Tasks → `BatterySim: build core` — только библиотека вычислительного ядра.

Все пути относительно проекта. Ничего не требуется копировать в `/tmp`.
Сборки лежат в `build/dev`, `build/debug`, `build/headless`, `build/server`, `build/core`; старые бинарники
и записи не перезаписываются. `dev` оптимизирован, но содержит символы отладки.
Настройка перед сборкой намеренно явная: открытие папки само по себе не запускает
CUDA-проверки и скачивание зависимостей. F5 выполняет её автоматически.
В CMake Tools можно выбрать configure preset `dev`, build preset `viewer`.

## Требования при переносе на другую машину

CMake ≥ 3.24, Ninja, совместимый с CUDA GCC/G++, CUDA Toolkit, драйвер NVIDIA,
GDB и библиотеки разработки OpenGL/X11 для desktop. Server/core графических
библиотек не требуют. Архитектура CUDA выбирается по местной
видеокарте (`native`); конфигурация требует доступного GPU. GLFW, ImGui, ImPlot,
GLM и JSON скачиваются CMake при первой настройке desktop — нужен интернет.
В server/core из этих зависимостей нужен только JSON.
`scripts/ide-cmake.sh` находит свежий CMake также в `~/.local/bin`, даже если
VS Code запущен из GNOME с укороченным PATH. Системные пакеты он не устанавливает.
Пути к компиляторам при необходимости можно переопределить через
`CMakeUserPresets.json` (не хранится в Git).

Записи replay содержат большие бинарные кадры, которые **не входят в Git**.
На этой машине они есть; после нового клонирования их нужно перенести отдельно
вместе с manifest.json и относительной структурой каталогов.

## Команды без IDE

```sh
./scripts/ide-cmake.sh --preset dev
./scripts/ide-cmake.sh --build --preset viewer
./build/dev/battery_sim_viewer examples/reaction-chain-demo.json
```

Полный набор целей: `--build --preset all-dev`; затем `ctest --preset dev`.
Известные прежние ошибки тестов описаны в `docs/refactor-checks-20261028.md`;
IDE-конфигурация не скрывает их и не меняет физику для прохождения тестов.

Справка: [задачи VS Code](https://code.visualstudio.com/docs/debugtest/tasks),
[CMake Tools](https://code.visualstudio.com/docs/cpp/cmake-linux),
[CMake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html).
