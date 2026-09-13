import json
import plotly.graph_objects as go
import pandas as pd
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("json_file", help="Google Benchmark JSON output file")
    args = parser.parse_args()

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        print("No benchmarks found in JSON")
        return

    eigen_data = {"N": [], "Time": []}
    cuda_data = {"N": [], "Time": []}

    for b in benchmarks:
        name = b["name"]
        if "BM_EigenVectorAddition" in name:
            try:
                n = int(name.split("/")[1])
                time_ms = b["real_time"] if b.get("time_unit") == "ms" else b["real_time"] / 1e6
                # google benchmark time is usually in ns or ms. Let's assume it's in the time_unit field.
                unit = b.get("time_unit", "ns")
                if unit == "ns":
                    time_ms = b["real_time"] / 1e6
                elif unit == "us":
                    time_ms = b["real_time"] / 1e3
                elif unit == "ms":
                    time_ms = b["real_time"]
                elif unit == "s":
                    time_ms = b["real_time"] * 1e3
                    
                eigen_data["N"].append(n)
                eigen_data["Time"].append(time_ms)
            except:
                pass
        elif "BM_CUDAVectorAddition" in name:
            try:
                n = int(name.split("/")[1])
                unit = b.get("time_unit", "ns")
                if unit == "ns":
                    time_ms = b["real_time"] / 1e6
                elif unit == "us":
                    time_ms = b["real_time"] / 1e3
                elif unit == "ms":
                    time_ms = b["real_time"]
                elif unit == "s":
                    time_ms = b["real_time"] * 1e3
                    
                cuda_data["N"].append(n)
                cuda_data["Time"].append(time_ms)
            except:
                pass

    df_eigen = pd.DataFrame(eigen_data).sort_values("N")
    df_cuda = pd.DataFrame(cuda_data).sort_values("N")

    # Plot Real Complexity
    fig_complexity = go.Figure()
    fig_complexity.add_trace(go.Scatter(x=df_eigen["N"], y=df_eigen["Time"], mode='lines+markers', name='Eigen Vector Addition (CPU)'))
    fig_complexity.add_trace(go.Scatter(x=df_cuda["N"], y=df_cuda["Time"], mode='lines+markers', name='CUDA Vector Addition (GPU)'))
    
    fig_complexity.update_layout(
        title="Real Complexity",
        xaxis_title="N",
        yaxis_title="Time, ms",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )
    fig_complexity.write_html("real_complexity.html")
    fig_complexity.write_image("real_complexity.png")

    # Plot Speedup
    if not df_eigen.empty and not df_cuda.empty:
        merged = pd.merge(df_eigen, df_cuda, on="N", suffixes=('_eigen', '_cuda'))
        merged["Speedup"] = merged["Time_eigen"] / merged["Time_cuda"]
        
        fig_speedup = go.Figure()
        fig_speedup.add_trace(go.Scatter(x=merged["N"], y=merged["Speedup"], mode='lines+markers', name='Speedup'))
        
        fig_speedup.update_layout(
            title="Speedup: CUDA Vector Addition (GPU) vs Eigen Vector Addition (CPU)",
            xaxis_title="N",
            yaxis_title="Speedup",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white"
        )
        fig_speedup.write_html("speedup.html")
        fig_speedup.write_image("speedup.png")

if __name__ == "__main__":
    main()

