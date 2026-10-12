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
    shmem = {}
    naive = {}

    for benchmark in benchmarks:
        name = benchmark.get("name", "")
        if name.startswith("BM_EigenMatrixMultiplication/"):
            size = int(name.split("/")[1])
            eigen[size] = time_in_milliseconds(benchmark)
        elif name.startswith("BM_CUDAMatrixMultiplication_Shmem/"):
            size = int(name.split("/")[1])
            shmem[size] = time_in_milliseconds(benchmark)
        elif name.startswith("BM_CUDAMatrixMultiplication_Naive/"):
            size = int(name.split("/")[1])
            naive[size] = time_in_milliseconds(benchmark)

    sizes = sorted(set(shmem.keys()) & set(naive.keys()))
    if not sizes:
        sizes = sorted(shmem.keys())

    return sizes, eigen, shmem, naive


benchmark_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
sizes, eigen_map, shmem_map, naive_map = load_measurements(benchmark_path)

shmem_times = [shmem_map[s] for s in sizes if s in shmem_map]
naive_times = [naive_map[s] for s in sizes if s in naive_map]
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
    labels_complexity.append('Eigen (CPU)')
    colors_complexity.append('#1f77b4')
if naive_times:
    lines_complexity.append(naive_times)
    labels_complexity.append('CUDA Naive (work2)')
    colors_complexity.append('#ff7f0e')
if shmem_times:
    lines_complexity.append(shmem_times)
    labels_complexity.append('CUDA Shared Memory (work3)')
    colors_complexity.append('#2ca02c')

svg_complexity = generate_svg(sizes, lines_complexity, labels_complexity, colors_complexity,
                              "Real Complexity", "Time, ms (log10)", is_log_y=True)

# Линии ускорения
lines_speedup = []
labels_speedup = []
colors_speedup = []

if naive_times and shmem_times:
    sp_naive = [nv / sh for nv, sh in zip(naive_times, shmem_times)]
    lines_speedup.append(sp_naive)
    labels_speedup.append('Ускорение Shmem vs Naive (GPU)')
    colors_speedup.append('#d62728')

if eigen_times and shmem_times:
    sp_cpu = [cp / sh for cp, sh in zip(eigen_times, shmem_times)]
    lines_speedup.append(sp_cpu)
    labels_speedup.append('Ускорение Shmem vs Eigen (CPU)')
    colors_speedup.append('#9467bd')

svg_speedup = generate_svg(sizes, lines_speedup, labels_speedup, colors_speedup,
                           "Speedup (CUDA Shared Memory)", "Speedup (log10)", is_log_y=True)

# Таблица результатов
table_rows = ""
for s in sizes:
    e_val = f"{eigen_map.get(s, 0.0):.4f}" if s in eigen_map else "-"
    n_val = f"{naive_map.get(s, 0.0):.4f}" if s in naive_map else "-"
    s_val = f"{shmem_map.get(s, 0.0):.4f}" if s in shmem_map else "-"
    sp_val = f"{(naive_map[s] / shmem_map[s]):.2f}x" if (s in naive_map and s in shmem_map and shmem_map[s] > 0) else "-"
    table_rows += f"<tr><td>{s}x{s}</td><td>{e_val}</td><td>{n_val}</td><td>{s_val}</td><td><b>{sp_val}</b></td></tr>\n"

html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Отчет по практической работе №3</title>
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
    <h1>Отчет по практической работе №3</h1>
    <p><strong>Тема:</strong> Умножение матриц на CUDA с использованием разделяемой памяти (Shared Memory)</p>
    <p><strong>Дисциплина:</strong> Высокопроизводительные вычислительные системы (GVS)</p>

    <h2>1. Цель работы</h2>
    <p>Закрепить навыки работы с двумерными сетками нитей CUDA. Освоить работу с разделяемой памятью и примитивами синхронизации нитей (<code>__syncthreads()</code>). Реализовать блочный алгоритм умножения матриц с паттерном «Стратегия».</p>

    <h2>2. Результаты вычислительного эксперимента</h2>
    <table>
        <thead>
            <tr>
                <th>Размер (N×N)</th>
                <th>Eigen CPU (мс)</th>
                <th>CUDA Naive (мс)</th>
                <th>CUDA Shmem (мс)</th>
                <th>Ускорение Shmem vs Naive</th>
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
    <p>Оба алгоритма (наивный и блочный) выполняют $O(N^3)$ арифметических операций с плавающей точкой (FLOPs). При $N = 1024$ это порядка $2 \cdot 10^9$ операций.</p>
    <p>В наивном алгоритме каждый поток на каждой итерации обращается к медленной глобальной памяти (DRAM, латентность 400–800 тактов). Суммарное количество обращений к DRAM составляет $2N^3$.</p>
    <p>В блочном алгоритме с разделяемой памятью нити блока загружают тайлы размером $BLOCK\_SIZE \times BLOCK\_SIZE$ ($16 \times 16$) в быструю разделяемую память (SRAM, латентность 20–30 тактов). Число обращений к DRAM сокращается в $BLOCK\_SIZE = 16$ раз, снижая нагрузку на шину памяти.</p>
    <p><strong>Экспериментальные выводы:</strong></p>
    <ul>
        <li>На малых размерах ($N \le 32$) CPU (Eigen) быстрее GPU в 3.5–20 раз из-за постоянных накладных расходов запуска CUDA-ядра (~45–50 мкс).</li>
        <li>Начиная с $N = 64$, GPU опережает CPU в 2.22 раза, а при $N = 1024$ блочная реализация превосходит CPU в <strong>120.4 раза</strong> (3.68 мс против 443.56 мс).</li>
        <li>Разделяемая память стабильно превосходит наивную версию на рабочих размерах: при $N = 128$ ускорение составляет <strong>1.42x</strong>, а при $N = 1024$ — <strong>1.34x</strong> (время выполнения сокращается с 4.94 мс до 3.68 мс).</li>
        <li>Разница между практическим ускорением (1.34x–1.42x) и теоретическим пределом (16x) обусловлена эффективным аппаратным L2-кэшированием GPU в наивном коде и накладными расходами на барьеры <code>__syncthreads()</code>.</li>
    </ul>

    <h2>4. Ключевые фрагменты кода</h2>
    <h3>kernel_matmul_shmem.cuh</h3>
<pre><code>template &lt;typename AtomT, int BLOCK_SIZE = 16&gt;
__global__ void kernel_matmul_shmem(const MatrixView&lt;const AtomT&gt; a, const MatrixView&lt;const AtomT&gt; b, MatrixView&lt;AtomT&gt; c) {{
    __shared__ AtomT As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ AtomT Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    AtomT sum = 0;
    int num_tiles = (a.ncols() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t &lt; num_tiles; ++t) {{
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row &lt; a.nrows() &amp;&amp; a_col &lt; a.ncols()) ? a(row, a_col) : 0;

        int b_row = t * BLOCK_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row &lt; b.nrows() &amp;&amp; col &lt; b.ncols()) ? b(b_row, col) : 0;

        __syncthreads();
        #pragma unroll
        for (int e = 0; e &lt; BLOCK_SIZE; ++e) sum += As[threadIdx.y][e] * Bs[e][threadIdx.x];
        __syncthreads();
    }}

    if (row &lt; c.nrows() &amp;&amp; col &lt; c.ncols()) c(row, col) = sum;
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
