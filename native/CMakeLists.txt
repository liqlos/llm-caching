cmake_minimum_required(VERSION 3.20)

project(pcss_llama_bridge LANGUAGES C CXX)

set(LLAMA_CPP_ROOT "" CACHE PATH "Prefix containing a llama.cpp C API install")

add_executable(pcss_llama_bridge pcss_llama_bridge.cpp)
target_compile_features(pcss_llama_bridge PRIVATE cxx_std_17)
target_compile_options(pcss_llama_bridge PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
)

if(CMAKE_DL_LIBS)
    target_link_libraries(pcss_llama_bridge PRIVATE ${CMAKE_DL_LIBS})
endif()

# Prefer an exported CMake package only when callers have not pinned a
# dedicated llama.cpp prefix. A globally discoverable Homebrew package must not
# override an explicit verifier artifact root.
if(NOT LLAMA_CPP_ROOT)
    find_package(llama CONFIG QUIET)
endif()

if(TARGET llama::llama)
    target_link_libraries(pcss_llama_bridge PRIVATE llama::llama)
elseif(TARGET llama)
    target_link_libraries(pcss_llama_bridge PRIVATE llama)
else()
    if(NOT LLAMA_CPP_ROOT)
        message(FATAL_ERROR
            "LLAMA_CPP_ROOT is required for the legacy verifier bridge. "
            "Build the pinned server with scripts/build.sh; do not fall back to Homebrew.")
    endif()

    find_path(LLAMA_CPP_INCLUDE_DIR
        NAMES llama.h
        HINTS "${LLAMA_CPP_ROOT}/include"
        REQUIRED
    )
    find_library(LLAMA_CPP_LIBRARY
        NAMES llama
        HINTS "${LLAMA_CPP_ROOT}/lib"
        REQUIRED
    )

    target_include_directories(pcss_llama_bridge PRIVATE "${LLAMA_CPP_INCLUDE_DIR}")
    target_link_libraries(pcss_llama_bridge PRIVATE "${LLAMA_CPP_LIBRARY}")

    if(APPLE)
        get_filename_component(LLAMA_CPP_LIBRARY_DIR "${LLAMA_CPP_LIBRARY}" DIRECTORY)
        set_target_properties(pcss_llama_bridge PROPERTIES
            BUILD_RPATH "${LLAMA_CPP_LIBRARY_DIR}"
        )
    endif()
endif()
