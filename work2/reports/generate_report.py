import json
import math
import sys
from pathlib import Path


def time_in_milliseconds(benchmark):
    unit = benchmark.get("time_unit", "ns")
    value = benchmark["real_time"]
    factors = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    return value * factors[unit]


def load_measurements(path):
    with open(path, "r", encoding="utf-8") as source:
        benchmarks = json.load(source).get("benchmarks", [])

    measurements = {"BM_EigenMatrixMultiplication": {}, "BM_CUDAMatrixMultiplication": {}}
    for benchmark in benchmarks:
        name = benchmark.get("name", "")
        for benchmark_name in measurements:
            if name.startswith(benchmark_name + "/"):
                size = int(name.split("/")[1])
                measurements[benchmark_name][size] = time_in_milliseconds(benchmark)

    eigen = measurements["BM_EigenMatrixMultiplication"]
    cuda = measurements["BM_CUDAMatrixMultiplication"]
    sizes = sorted(set(eigen) & set(cuda))
    if not sizes:
        raise ValueError("No matching Eigen/CUDA measurements found")

    return sizes, [eigen[size] for size in sizes], [cuda[size] for size in sizes]


benchmark_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
N, cpu_time, gpu_time = load_measurements(benchmark_path)
speedup = [cpu / gpu for cpu, gpu in zip(cpu_time, gpu_time)]


def generate_svg(x_vals, y_lines, labels, title, ylabel, is_log_y=True):
    width, height = 800, 500
    pad_left, pad_right, pad_top, pad_bottom = 80, 40, 60, 60
    graph_w = width - pad_left - pad_right
    graph_h = height - pad_top - pad_bottom

    log_x = [math.log10(v) for v in x_vals]
    min_x, max_x = min(log_x), max(log_x)

    all_y = [v for line in y_lines for v in line]
    if is_log_y:
        log_y = [math.log10(v) if v > 0 else 0 for v in all_y]
    else:
        log_y = all_y

    min_y, max_y = min(log_y), max(log_y)

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

    colors = ["#1f77b4", "#ff7f0e"]
    for i, line in enumerate(y_lines):
        pts = []
        for j, val in enumerate(line):
            cx = transform_x(log_x[j])
            cy = transform_y(math.log10(val) if is_log_y else val)
            pts.append(f"{cx},{cy}")
            svg += f'<circle cx="{cx}" cy="{cy}" r="4" fill="{colors[i]}"/>\n'

        svg += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[i]}" stroke-width="2"/>\n'

        leg_y = pad_top + i * 20
        svg += f'<line x1="{pad_left + 20}" y1="{leg_y}" x2="{pad_left + 40}" y2="{leg_y}" stroke="{colors[i]}" stroke-width="2"/>\n'
        svg += f'<text x="{pad_left + 45}" y="{leg_y + 4}" font-size="12">{labels[i]}</text>\n'

    svg += '</svg>'
    return svg


svg_complexity = generate_svg(N, [cpu_time, gpu_time], ['Eigen (CPU)', 'CUDA (GPU)'], "Real Complexity", "Time, ms (log10)", is_log_y=True)
svg_speedup = generate_svg(N, [speedup], ['Speedup (CUDA / Eigen)'], "Speedup: CUDA vs Eigen", "Speedup (log10)", is_log_y=True)

html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Отчет по практической работе №2</title>
<style>
    body {{ max-width: 900px; margin: 0 auto; font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
    pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 14px; border: 1px solid #ddd; }}
    h1, h2, h3 {{ color: #333; }}
    .svg-container {{ text-align: center; margin: 30px 0; }}
</style>
</head>
<body>
    <h1>Отчет по практической работе №2</h1>
    <p><strong>Дисциплина:</strong> GVS</p>

    <h2>1. Цель работы</h2>
    <p>Освоить двумерные сетки нитей CUDA и реализовать наивное умножение матриц C = A * B с использованием паттерна Data + View.</p>

    <h2>2. Результаты измерений</h2>
    <div class="svg-container">
        {svg_complexity}
    </div>
    <div class="svg-container">
        {svg_speedup}
    </div>

    <h2>3. Интерпретация результатов</h2>
    <p>Теоретическая сложность наивного умножения матриц кубична O(N³).</p>
    <p>Графики построены по фактическим результатам Google Benchmark из файла <code>{benchmark_path}</code>.</p>

    <h2>4. Фрагменты исходного кода</h2>
    <h3>MatrixView.cuh</h3>
<pre><code>template &lt;typename AtomT&gt;
class MatrixView {{
    AtomT* data_; int nrows_; int ncols_;
public:
    __host__ __device__ MatrixView(AtomT* data, int nrows, int ncols);
    __host__ __device__ AtomT&amp; operator()(int i, int j) {{ return data_[i * ncols_ + j]; }}
}};</code></pre>

    <h3>kernel_matmul_naive.cuh</h3>
<pre><code>template &lt;typename AtomT&gt;
__global__ void kernel_matmul_naive(const MatrixView&lt;const AtomT&gt; a, const MatrixView&lt;const AtomT&gt; b, MatrixView&lt;AtomT&gt; c) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row &lt; c.nrows() &amp;&amp; col &lt; c.ncols()) {{
        AtomT sum = 0;
        for (int k = 0; k &lt; a.ncols(); ++k) sum += a(row, k) * b(k, col);
        c(row, col) = sum;
    }}
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
