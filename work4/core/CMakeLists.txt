add_library(core INTERFACE)

target_include_directories(core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(core INTERFACE cxx_std_17)
