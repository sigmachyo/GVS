# Практическая работа №2

Умножение матриц на CUDA с использованием паттерна **Data + View**.

---

## Способ 1: Использовать существующий контейнер `gvs` (Рекомендуется)

Контейнер `gvs` уже монтирует весь репозиторий в `/app`, поэтому `work2` уже доступен внутри:

```powershell
# Подключиться к уже запущенному контейнеру gvs
docker exec -it gvs bash

# Внутри контейнера перейти в work2:
cd /app/work2
```

---

## Способ 2: Отдельный образ и контейнер `gvs_work2`

Если нужен изолированный контейнер только под work2:

**1. Собрать образ (из корня репозитория GVS):**
```powershell
docker build -t gvs_work2 -f work2/Dockerfile work2/
```

**2. Запустить интерактивный контейнер:**
```powershell
docker run --gpus all -it --name gvs_work2_env -v ${PWD}/work2:/app gvs_work2 bash
```

**3. Вернуться к нему в следующий раз:**
```powershell
docker start -ai gvs_work2_env
```

---

## Сборка внутри контейнера

```bash
# Настройка (один раз)
cmake -B /tmp/build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release

# Компиляция (после каждого изменения кода)
cmake --build /tmp/build
```

---

## Запуск

```bash
# Только тесты
/tmp/build/tests/test_matrix

# Только бенчмарки
mkdir -p reports/temp
/tmp/build/benchmarks/bench_matrix --benchmark_out=reports/temp/benchmark_results.json --benchmark_out_format=json

# Графики (после бенчмарков)
python3 reports/scripts/plot.py reports/temp/benchmark_results.json

# Отчёт HTML
python3 reports/generate_report.py reports/temp/benchmark_results.json

# Отчёт DOCX
python3 reports/create_report_docx.py

# Всё сразу
bash run.sh
```
