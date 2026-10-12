# Практическая работа №3

Умножение матриц на CUDA с использованием **разделяемой памяти (Shared Memory)** и паттерна **Стратегия**.

---

## Запуск из существующего контейнера `gvs`

Контейнер `gvs` уже запущен и монтирует весь репозиторий в `/app`.

### 1. Подключение к контейнеру (в терминале PowerShell на хосте):

```powershell
docker exec -it gvs bash
```

*(Если контейнер остановлен: `docker start -ai gvs`)*

---

### 2. Сборка (внутри контейнера):

```bash
cd /app/work3

# Настройка сборки в /tmp/build3 (один раз)
cmake -B /tmp/build3 -S . -G Ninja -DCMAKE_BUILD_TYPE=Release

# Компиляция (после каждого изменения исходного кода)
cmake --build /tmp/build3
```

---

### 3. Запуск:

```bash
# Только тесты (343 комбинации размеров)
/tmp/build3/tests/test_matrix

# Только бенчмарки (Shmem vs Naive vs CPU)
mkdir -p reports/temp
/tmp/build3/benchmarks/bench_matrix --benchmark_out=reports/temp/benchmark_results.json --benchmark_out_format=json

# Графики сложности и ускорения (после бенчмарков)
python3 reports/scripts/plot.py reports/temp/benchmark_results.json

# Отчет HTML
python3 reports/generate_report.py reports/temp/benchmark_results.json

# Отчет DOCX по СТУ 7.5-07-2021
python3 reports/create_report_docx.py

# Или всё сразу одной командой (сборка + тесты + бенчмарки + графики + отчеты):
bash run.sh
```

---

## Структура компонентов

- `core/include/kernel_matmul_shmem.cuh` — блочное CUDA-ядро с тайлингом $16 \times 16$, разделяемой памятью `__shared__` и барьерами `__syncthreads()`.
- `core/include/kernel_matmul_naive.cuh` — наивное ядро из работы №2 (для сравнения производительности).
- `core/include/Matrix.cuh` — фасад с поддержкой `MatmulStrategy::SharedMemory` и `MatmulStrategy::Naive`.
- `tests/src/test_matrix.cu` — тесты Google Test ($m, k, n \in \{1, 2, 3, 127, 128, 129, 512\}$).
- `benchmarks/src/bench_matrix.cu` — бенчмарки Google Benchmark с CUDA Events API ($n \in \{16, \dots, 1024\}$).
- `reports/temp/` — сгенерированные данные и отчёты (игнорируются git).
