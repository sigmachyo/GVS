add_executable(bench_matrix src/bench_matrix.cu)
target_link_libraries(bench_matrix PRIVATE core benchmark::benchmark_main Eigen3::Eigen)
