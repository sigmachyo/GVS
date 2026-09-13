add_executable(test_vector src/test_vector.cu)
target_link_libraries(test_vector PRIVATE core gtest_main Eigen3::Eigen)
gtest_discover_tests(test_vector)

