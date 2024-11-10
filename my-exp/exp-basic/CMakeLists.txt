
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  MESSAGE("\nCurrent compiler is Clang\n")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  MESSAGE("\nCurrent compiler is GNU, version ${CMAKE_CXX_COMPILER_VERSION}, ${CMAKE_CXX_COMPILER}\n")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  MESSAGE("\nCurrent compiler is Intel\n")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  MESSAGE("\nCurrent compiler is MSVC\n")
endif()

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()

add_executable(exp-ctx exp-ctx.c)
target_link_libraries(exp-ctx PRIVATE ggml)

add_executable(exp-ctx-cxx exp-ctx-cxx.cpp)
target_link_libraries(exp-ctx-cxx PRIVATE ggml)

add_executable(exp-backend exp-backend.cpp)
target_link_libraries(exp-backend PRIVATE ggml)



