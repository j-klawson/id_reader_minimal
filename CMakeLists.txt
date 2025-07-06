cmake_minimum_required(VERSION 3.10)
project(IDCardReader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add source files
set(SOURCE_FILES
    src/main.cpp
    src/detect_id_card.cpp
)

# Add executable
add_executable(detect_id_card ${SOURCE_FILES})

# Link libraries
target_link_libraries(detect_id_card ${OpenCV_LIBS})

# Include directories
target_include_directories(detect_id_card PRIVATE ${OpenCV_INCLUDE_DIRS} src)

# Copy assets to the build directory
file(COPY src/assets/haarcascade_frontalface_default.xml DESTINATION ${CMAKE_BINARY_DIR}/assets)
