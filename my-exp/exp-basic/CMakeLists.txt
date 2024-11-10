add_executable(exp-ctx exp-ctx.c)
target_link_libraries(exp-ctx PRIVATE ggml)

add_executable(exp-ctx-cxx exp-ctx-cxx.cpp)
target_link_libraries(exp-ctx-cxx PRIVATE ggml)

add_executable(exp-backend exp-backend.cpp)
target_link_libraries(exp-backend PRIVATE ggml)
