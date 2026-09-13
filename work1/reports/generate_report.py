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

    measurements = {"BM_EigenVectorAddition": {}, "BM_CUDAVectorAddition": {}}
    for benchmark in benchmarks:
        name = benchmark.get("name", "")
        for benchmark_name in measurements:
            if name.startswith(benchmark_name + "/"):
                size = int(name.split("/")[1])
                measurements[benchmark_name][size] = time_in_milliseconds(benchmark)

    eigen = measurements["BM_EigenVectorAddition"]
    cuda = measurements["BM_CUDAVectorAddition"]
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
    svg += f'<text x="{width/2}" y="{height - 15}" text-anchor="middle" font-size="14">N (log10)</text>\n'

    # Draw grid and axes
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
        
        # Legend
        leg_y = pad_top + i * 20
        svg += f'<line x1="{pad_left + 20}" y1="{leg_y}" x2="{pad_left + 40}" y2="{leg_y}" stroke="{colors[i]}" stroke-width="2"/>\n'
        svg += f'<text x="{pad_left + 45}" y="{leg_y + 4}" font-size="12">{labels[i]}</text>\n'
        
    svg += '</svg>'
    return svg

svg_complexity = generate_svg(N, [cpu_time, gpu_time], ['Eigen (CPU)', 'CUDA (GPU)'], "Real Complexity", "Time, ms (log10)", is_log_y=True)
svg_speedup = generate_svg(N, [speedup], ['Speedup (GPU / CPU)'], "Speedup: CUDA vs Eigen", "Speedup (log10)", is_log_y=True)

html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Отчет по практической работе №1</title>
<style>
    body {{ max-width: 900px; margin: 0 auto; font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
    pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 14px; border: 1px solid #ddd; }}
    h1, h2, h3 {{ color: #333; }}
    .svg-container {{ text-align: center; margin: 30px 0; }}
</style>
</head>
<body>
    <h1>Отчет по практической работе №1</h1>
    <p><strong>Дисциплина:</strong> GVS</p>
    
    <h2>1. Цель работы</h2>
    <p>Освоить базовые навыки программирования CUDA: работу с одномерными сетками нитей и динамической памятью устройства. Изучить паттерн проектирования data + view.</p>

    <h2>2. Результаты измерений</h2>
    <div class="svg-container">
        {svg_complexity}
    </div>
    <div class="svg-container">
        {svg_speedup}
    </div>

    <h2>3. Интерпретация результатов</h2>
    <p>Теоретическая сложность сложения векторов линейна O(N).</p>
    <p>Графики построены по фактическим результатам Google Benchmark из файла <code>{benchmark_path}</code>. Для сложения векторов ожидается линейная асимптотика O(N), а различия на малых размерах объясняются постоянными накладными расходами запуска CUDA-ядра.</p>

    <h2>4. Фрагменты исходного кода</h2>
    <h3>Data.cuh (фрагмент)</h3>
<pre><code>template &lt;typename AtomT&gt;
class Data {{
private:
    std::size_t size_;
    AtomT* data_;
    void allocate(std::size_t size) {{
        if (size > 0) cudaMalloc(&amp;data_, size * sizeof(AtomT));
    }}
public:
    explicit Data(std::size_t size) : size_(size), data_(nullptr) {{ allocate(size_); }}
    ~Data() {{ if (data_) cudaFree(data_); }}
    // ... move, copy, RAII ...
}};</code></pre>

    <h3>VectorView.cuh</h3>
<pre><code>template &lt;typename AtomT&gt;
class VectorView {{
private:
    AtomT* data_;
    std::size_t size_;
public:
    __host__ __device__ VectorView(AtomT* data, std::size_t size) : data_(data), size_(size) {{}}
    __host__ __device__ std::size_t size() const {{ return size_; }}
    __host__ __device__ AtomT&amp; operator[](std::size_t n) {{ return data_[n]; }}
}};</code></pre>

    <h3>Vector.cuh</h3>
<pre><code>template &lt;typename AtomT&gt;
class Vector {{
private:
    std::shared_ptr&lt;Data&lt;AtomT&gt;&gt; data_;
    VectorView&lt;AtomT&gt; view_;
public:
    explicit Vector(std::size_t size) 
        : data_(std::make_shared&lt;Data&lt;AtomT&gt;&gt;(size)),
          view_(data_->data(), size) {{}}
    // ...
}};
</code></pre>

    <h3>kernel_vecadd.cuh</h3>
<pre><code>template &lt;typename AtomT&gt;
__global__ void kernel_vecadd(VectorView&lt;AtomT&gt; lhs, VectorView&lt;AtomT&gt; rhs, VectorView&lt;AtomT&gt; out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out.size()) {{
        out[i] = lhs[i] + rhs[i];
    }}
}}</code></pre>
</body>
</html>
"""

output_path = Path(__file__).with_name("Report.html")
with output_path.open("w", encoding="utf-8") as f:
    f.write(html)
print(f"{output_path} generated!")

