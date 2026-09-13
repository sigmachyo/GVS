#!/bin/bash
set -e

echo "=== Сборка проекта с CMake и Ninja ==="
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

echo "=== Запуск тестов ==="
./build/tests/test_vector

echo "=== Запуск бенчмарков ==="
./build/benchmarks/bench_vector --benchmark_out=reports/benchmark_results.json --benchmark_out_format=json

echo "=== Построение графиков (нужен python, pandas, plotly) ==="
python3 reports/scripts/plot.py reports/benchmark_results.json
python3 reports/generate_report.py reports/benchmark_results.json
python3 reports/create_report_docx.py

echo "=== Готово! Теперь вы можете скомпилировать report.tex ==="

