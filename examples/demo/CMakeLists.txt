
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  MESSAGE("\nCurrent compiler is Clang")
  MESSAGE("Supporting Variable-Length Arrays\n")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  MESSAGE("\nCurrent compiler is GNU, version ${CMAKE_CXX_COMPILER_VERSION}, ${CMAKE_CXX_COMPILER}")
  MESSAGE("Not supporting Variable-Length Arrays\n")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  MESSAGE("\nCurrent compiler is Intel\n")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  MESSAGE("\nCurrent compiler is MSVC\n")
endif()

set(TEST_TARGET demo-ctx)
add_executable(${TEST_TARGET} demo-ctx.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

set(TEST_TARGET demo-backend)
add_executable(${TEST_TARGET} demo-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()