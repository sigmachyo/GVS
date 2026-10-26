import json
import math
import sys
from pathlib import Path


def time_in_milliseconds(benchmark):
    unit = benchmark.get("time_unit", "ns")
    value = benchmark["real_time"]
    factors = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    return value * factors.get(unit, 1.0)


def load_measurements(path):
    with open(path, "r", encoding="utf-8") as source:
        benchmarks = json.load(source).get("benchmarks", [])

    eigen = {}
    wmma = {}
    shmem = {}

    for benchmark in benchmarks:
        name = benchmark.get("name", "")
        if name.startswith("BM_EigenMatrixMultiplication/"):
            size = int(name.split("/")[1])
            eigen[size] = time_in_milliseconds(benchmark)
        elif name.startswith("BM_CUDAMatrixMultiplication_WMMA/"):
            size = int(name.split("/")[1])
            wmma[size] = time_in_milliseconds(benchmark)
        elif name.startswith("BM_CUDAMatrixMultiplication_Shmem/"):
            size = int(name.split("/")[1])
            shmem[size] = time_in_milliseconds(benchmark)

    sizes = sorted(set(wmma.keys()) & set(shmem.keys()))
    if not sizes:
        sizes = sorted(wmma.keys())

    return sizes, eigen, wmma, shmem


benchmark_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
sizes, eigen_map, wmma_map, shmem_map = load_measurements(benchmark_path)

wmma_times  = [wmma_map[s] for s in sizes if s in wmma_map]
shmem_times = [shmem_map[s] for s in sizes if s in shmem_map]
eigen_times = [eigen_map[s] for s in sizes if s in eigen_map]


def generate_svg(x_vals, y_lines, labels, colors, title, ylabel, is_log_y=True):
    width, height = 800, 500
    pad_left, pad_right, pad_top, pad_bottom = 80, 40, 60, 60
    graph_w = width - pad_left - pad_right
    graph_h = height - pad_top - pad_bottom

    log_x = [math.log10(v) for v in x_vals]
    min_x, max_x = min(log_x), max(log_x)
    if min_x == max_x:
        max_x += 1.0

    all_y = [v for line in y_lines for v in line if v > 0]
    if not all_y:
        return ""
    log_y = [math.log10(v) if is_log_y else v for v in all_y]
    min_y, max_y = min(log_y), max(log_y)
    if min_y == max_y:
        max_y += 1.0

    def transform_x(val):
        return pad_left + (val - min_x) / (max_x - min_x) * graph_w

    def transform_y(val):
        return height - pad_bottom - (val - min_y) / (max_y - min_y) * graph_h

    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#fff; border:1px solid #ddd; font-family:sans-serif;">\n'
    svg += f'<text x="{width/2}" y="35" text-anchor="middle" font-size="18" font-weight="bold">{title}</text>\n'
    svg += f'<text x="25" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 25 {height/2})">{ylabel}</text>\n'
    svg += f'<text x="{width/2}" y="{height - 15}" text-anchor="middle" font-size="14">N, log10</text>\n'

    svg += f'<line x1="{pad_left}" y1="{height-pad_bottom}" x2="{width-pad_right}" y2="{height-pad_bottom}" stroke="#333" stroke-width="2"/>\n'
    svg += f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height-pad_bottom}" stroke="#333" stroke-width="2"/>\n'

    for i, line in enumerate(y_lines):
        pts = []
        for j, val in enumerate(line):
            cx = transform_x(log_x[j])
            cy = transform_y(math.log10(val) if is_log_y else val)
            pts.append(f"{cx},{cy}")
            svg += f'<circle cx="{cx}" cy="{cy}" r="4" fill="{colors[i]}"/>\n'

        svg += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[i]}" stroke-width="2"/>\n'

        leg_y = pad_top + i * 22
        svg += f'<line x1="{pad_left + 20}" y1="{leg_y}" x2="{pad_left + 45}" y2="{leg_y}" stroke="{colors[i]}" stroke-width="2"/>\n'
        svg += f'<text x="{pad_left + 50}" y="{leg_y + 4}" font-size="12">{labels[i]}</text>\n'

    svg += '</svg>'
    return svg


# Линии сложности
lines_complexity = []
labels_complexity = []
colors_complexity = []

if eigen_times:
    lines_complexity.append(eigen_times)
    labels_complexity.append('Eigen (CPU, half)')
    colors_complexity.append('#1f77b4')
if shmem_times:
    lines_complexity.append(shmem_times)
    labels_complexity.append('CUDA Shared Memory (work3)')
    colors_complexity.append('#ff7f0e')
if wmma_times:
    lines_complexity.append(wmma_times)
    labels_complexity.append('CUDA WMMA (work4)')
    colors_complexity.append('#2ca02c')

svg_complexity = generate_svg(sizes, lines_complexity, labels_complexity, colors_complexity,
                              "Real Complexity", "Time, ms (log10)", is_log_y=True)

# Линии ускорения
lines_speedup = []
labels_speedup = []
colors_speedup = []

if shmem_times and wmma_times:
    sp_shmem = [sh / wm for sh, wm in zip(shmem_times, wmma_times)]
    lines_speedup.append(sp_shmem)
    labels_speedup.append('Ускорение WMMA vs Shmem (GPU)')
    colors_speedup.append('#d62728')

if eigen_times and wmma_times:
    sp_cpu = [cp / wm for cp, wm in zip(eigen_times, wmma_times)]
    lines_speedup.append(sp_cpu)
    labels_speedup.append('Ускорение WMMA vs Eigen (CPU)')
    colors_speedup.append('#9467bd')

svg_speedup = generate_svg(sizes, lines_speedup, labels_speedup, colors_speedup,
                           "Speedup (CUDA WMMA)", "Speedup (log10)", is_log_y=True)

# Таблица результатов
table_rows = ""
for s in sizes:
    e_val = f"{eigen_map.get(s, 0.0):.4f}" if s in eigen_map else "-"
    sh_val = f"{shmem_map.get(s, 0.0):.4f}" if s in shmem_map else "-"
    w_val = f"{wmma_map.get(s, 0.0):.4f}" if s in wmma_map else "-"
    sp_val = f"{(shmem_map[s] / wmma_map[s]):.2f}x" if (s in shmem_map and s in wmma_map and wmma_map[s] > 0) else "-"
    table_rows += f"<tr><td>{s}x{s}</td><td>{e_val}</td><td>{sh_val}</td><td>{w_val}</td><td><b>{sp_val}</b></td></tr>\n"

html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Отчет по практической работе №4</title>
<style>
    body {{ max-width: 900px; margin: 0 auto; font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; color: #222; }}
    pre {{ background: #f8f9fa; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 13px; border: 1px solid #e2e8f0; }}
    h1, h2, h3 {{ color: #1a202c; }}
    .svg-container {{ text-align: center; margin: 30px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: center; font-size: 14px; }}
    th {{ background: #f1f5f9; }}
</style>
</head>
<body>
    <h1>Отчет по практической работе №4</h1>
    <p><strong>Тема:</strong> Умножение матриц на CUDA с использованием тензорных ядер (WMMA API)</p>
    <p><strong>Дисциплина:</strong> Высокопроизводительные вычислительные системы (GVS)</p>

    <h2>1. Цель работы</h2>
    <p>Получить навыки программирования тензорных ядер CUDA. Реализовать ядро умножения матриц с использованием CUDA WMMA API (Warp Matrix Multiply-Accumulate) для типа данных <code>__half</code> (FP16) с аккумуляцией в <code>float</code> (FP32).</p>

    <h2>2. Результаты вычислительного эксперимента</h2>
    <table>
        <thead>
            <tr>
                <th>Размер (N×N)</th>
                <th>Eigen CPU (мс)</th>
                <th>CUDA Shmem (мс)</th>
                <th>CUDA WMMA (мс)</th>
                <th>Ускорение WMMA vs Shmem</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>

    <div class="svg-container">
        {svg_complexity}
    </div>
    <div class="svg-container">
        {svg_speedup}
    </div>

    <h2>3. Анализ и сравнение с теоретической сложностью</h2>
    <p>Все три алгоритма (наивный с разделяемой памятью и WMMA) выполняют $O(N^3)$ арифметических операций. Однако WMMA использует аппаратные тензорные ядра (Tensor Cores), которые выполняют операцию D = A * B + C для матриц 16×16 за одну аппаратную инструкцию на уровне варпа.</p>
    <p>Тензорные ядра обрабатывают данные в формате FP16 (__half), что вдвое сокращает объём данных по сравнению с FP32 и удваивает пропускную способность шины памяти. Аккумуляция результата выполняется в FP32 для сохранения точности.</p>

    <h2>4. Ключевые фрагменты кода</h2>
    <h3>kernel_matmul_wmma.cuh</h3>
<pre><code>__global__ void kernel_matmul_wmma(const MatrixView&lt;const __half&gt; a,
                                   const MatrixView&lt;const __half&gt; b,
                                   MatrixView&lt;__half&gt; c) {{
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * WMMA_N;
    if (warpM >= c.nrows() || warpN >= c.ncols()) return;

    wmma::fragment&lt;wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float&gt; acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int t = 0; t &lt; (a.ncols() + WMMA_K - 1) / WMMA_K; ++t) {{
        wmma::fragment&lt;wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major&gt; a_frag;
        wmma::fragment&lt;wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major&gt; b_frag;
        // загрузка фрагментов и mma_sync ...
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }}
    // сохранение результата через store_matrix_sync
}}</code></pre>
</body>
</html>
"""

output_dir = Path(__file__).parent / "temp"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "Report.html"
with output_path.open("w", encoding="utf-8") as f:
    f.write(html)
print(f"{output_path} generated!")
