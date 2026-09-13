# Практическая работа №1

В данной работе реализовано сложение векторов на CUDA с использованием паттерна проектирования **Data + View** и идиомы **RAII**. 
Проект написан на стандарте **C++20** и собирается с помощью **CMake** и **Ninja**.

---

## 📂 Структура репозитория

- `core/` — Основная библиотека `core` (Header-only для CUDA-шаблонов)
  - `include/Data.cuh` — Класс `Data` реализует выделение/освобождение памяти на устройстве через RAII, а также копирование с хоста на устройство и обратно.
  - `include/VectorView.cuh` — Класс `VectorView`, который передается в CUDA-кёрнел по значению. Не владеет ресурсами, тривиально-копируемый (методы помечены `__host__ __device__`).
  - `include/Vector.cuh` — Фасад, объединяющий агрегацию `Data` через `std::shared_ptr` (позволяя копировать вектор без глубокого копирования данных) и композицию с `VectorView`. Включает перегруженный `operator+`.
  - `include/kernel_vecadd.cuh` — Само CUDA-ядро, выполняющее сложение векторов.
- `tests/` — Модульные тесты Google Test
  - Сравнивают результаты вычислений нашей реализации CUDA с эталоном `Eigen::VectorXf` для заданных размеров N. Оценка точности проводится при помощи `isApprox(1e-6)`.
- `benchmarks/` — Бенчмарки Google Benchmark
  - Замеряют время выполнения `operator+` для CUDA (используя CUDA Events API для чистого замера времени ядра) и для Eigen.
- `reports/` — Сопроводительные материалы, не необходимые для сборки функционала.
  - `work1.pdf` — исходное задание.
  - `report.tex`, `report.docx`, `Report.html` — варианты отчёта.
  - `generate_report.py`, `create_report_docx.py`, `scripts/plot.py` — генераторы отчёта и графиков.
- `Dockerfile` — Готовый образ для сборки и запуска на машине с GPU.
- `run.sh` — Bash-скрипт для сборки и запуска всего цикла в 1 команду.

---

## 🛠 Зависимости (Dependencies)

Для локального запуска вам понадобятся:
1. **NVIDIA CUDA Toolkit** (включает `nvcc`). На Mac не поддерживается, необходим Linux/Windows ПК с видеокартой NVIDIA.
2. **CMake** (версия 3.20+)
3. **Ninja** (опционально, но рекомендуется для быстрой сборки)
4. Библиотеки **Eigen3**, **GoogleTest** и **Google Benchmark** (настроены на автоматическое скачивание через `FetchContent` внутри CMake, устанавливать их вручную в систему не нужно!).
5. **Python 3** с библиотеками `pandas` и `plotly` (только если вы хотите сгенерировать отчётные графики).

---

## 🚀 Как собрать и запустить

### Способ 1: Использование Docker (Рекомендуемый)

Если у вас есть доступ к серверу с поддержкой NVIDIA Container Toolkit, вы можете собрать всё в изоляции:
```bash
docker build -t gvs_work1 .
docker run --gpus all gvs_work1
```

### Способ 2: Локальная сборка (На Linux / Windows)

```bash
# 1. Создаем папку сборки
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# 2. Компилируем проект
cmake --build build

# 3. Запускаем модульные тесты
./build/tests/test_vector

# 4. Запускаем бенчмарки и сохраняем результаты в JSON
./build/benchmarks/bench_vector --benchmark_out=reports/benchmark_results.json --benchmark_out_format=json

# 5. Генерируем графики (требуется python, pandas, plotly, kaleido)
pip install pandas plotly kaleido
python reports/scripts/plot.py reports/benchmark_results.json
python reports/generate_report.py reports/benchmark_results.json
python reports/create_report_docx.py
```

Вы также можете использовать файл `run.sh`, который объединяет все эти команды.

---

## 📊 Отчет

Я подготовил 2 варианта отчета:
1. **Word-отчет (`reports/report.docx`)**: создается скриптом `reports/create_report_docx.py` с оформлением по СТУ 7.5–07–2021. После запуска бенчмарков он включает фактические графики производительности.
2. **HTML-отчет (`reports/Report.html`)**: после запуска бенчмарков он пересоздается с фактическими графиками производительности и фрагментами кода. Его можно открыть в браузере и сохранить как PDF.
2. **LaTeX-отчет (`report.tex`)**: Классический шаблон, который вы можете загрузить на Overleaf. Чтобы вставить в него ваши собственные реальные графики, раскомментируйте блоки `\includegraphics` после запуска `plot.py`.
