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

# Prefer an exported CMake package when the installed llama.cpp provides one.
find_package(llama CONFIG QUIET)

if(TARGET llama::llama)
    target_link_libraries(pcss_llama_bridge PRIVATE llama::llama)
elseif(TARGET llama)
    target_link_libraries(pcss_llama_bridge PRIVATE llama)
else()
    if(NOT LLAMA_CPP_ROOT)
        set(LLAMA_CPP_ROOT "/opt/homebrew/opt/llama.cpp")
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
