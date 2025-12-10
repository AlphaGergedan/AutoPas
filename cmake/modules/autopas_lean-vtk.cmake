set(VTKIO_BUNDLED_DOC "Uses a lean header-only library for writing vtk outputs.")
option(VTKIO_BUNDLED ${VTKIO_BUNDLED_DOC} ON)

message(STATUS "LeanVTK - using bundled version 1.0")

# Enable FetchContent CMake module
include(FetchContent)

# Build LeanVTK and make the cmake targets available
FetchContent_Declare(
        LeanVTK
        URL ${AUTOPAS_SOURCE_DIR}/libs/lean-vtk.zip
        URL_HASH MD5=d8d9a98e415b5ed7c505082e4f6379c1
)
FetchContent_MakeAvailable(LeanVTK)
