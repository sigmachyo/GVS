#!/bin/bash
set -e

BUILD_DIR="/tmp/build3"

echo "=== Сборка проекта work3 (CMake + Ninja) ==="
cmake -B "$BUILD_DIR" -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR"

echo "=== Запуск Google Test ==="
"$BUILD_DIR/tests/test_matrix"

echo "=== Запуск Google Benchmark ==="
mkdir -p reports/temp
"$BUILD_DIR/benchmarks/bench_matrix" --benchmark_out=reports/temp/benchmark_results.json --benchmark_out_format=json

echo "=== Построение графиков и отчетов ==="
python3 reports/scripts/plot.py reports/temp/benchmark_results.json
python3 reports/generate_report.py reports/temp/benchmark_results.json
python3 reports/create_report_docx.py

echo "=== Готово! Результаты сохранены в reports/temp/ ==="
