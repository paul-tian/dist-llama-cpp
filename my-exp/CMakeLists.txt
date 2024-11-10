# dependencies

find_package(Threads REQUIRED)

# third-party

# ...


if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  MESSAGE("\nCurrent compiler is Clang\n")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  MESSAGE("\nCurrent compiler is GNU, version ${CMAKE_CXX_COMPILER_VERSION}, ${CMAKE_CXX_COMPILER}\n")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  MESSAGE("\nCurrent compiler is Intel\n")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  MESSAGE("\nCurrent compiler is MSVC\n")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(exp-basic)
    if (EXP_RPC)
        add_subdirectory(exp-rpc)
    endif()
endif()
