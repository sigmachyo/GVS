add_executable(test_matrix src/test_matrix.cu)
target_link_libraries(test_matrix PRIVATE core gtest_main Eigen3::Eigen)
gtest_discover_tests(test_matrix)
