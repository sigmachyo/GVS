add_executable(bench_vector src/bench_vector.cu)
target_link_libraries(bench_vector PRIVATE core benchmark::benchmark_main Eigen3::Eigen)

