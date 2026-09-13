#!/bin/bash
set -e

echo "=== Сборка проекта с CMake и Ninja ==="
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

echo "=== Запуск тестов ==="
./build/tests/test_vector

echo "=== Запуск бенчмарков ==="
./build/benchmarks/bench_vector --benchmark_out=benchmark_results.json --benchmark_out_format=json

echo "=== Построение графиков (нужен python, pandas, plotly) ==="
python3 scripts/plot.py benchmark_results.json
python3 generate_report.py benchmark_results.json

echo "=== Готово! Теперь вы можете скомпилировать report.tex ==="

