FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY reports/requirements.txt /tmp/report-requirements.txt
RUN pip3 install --no-cache-dir --upgrade cmake && \
    pip3 install --no-cache-dir -r /tmp/report-requirements.txt

WORKDIR /app
COPY . /app

CMD ["/bin/bash", "-c", "\
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build && \
    ./build/tests/test_matrix && \
    ./build/benchmarks/bench_matrix --benchmark_out=reports/temp/benchmark_results.json --benchmark_out_format=json && \
    python3 reports/scripts/plot.py reports/temp/benchmark_results.json && \
    python3 reports/generate_report.py reports/temp/benchmark_results.json && \
    python3 reports/create_report_docx.py \
    "]
