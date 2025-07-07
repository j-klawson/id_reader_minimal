 cmake_minimum_required(VERSION 3.10)
project(IDCardReader LANGUAGES CXX)

enable_testing()
find_package(GTest REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Generate C++ header from XML using xxd
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/haarcascade_data.h
    COMMAND xxd -i ${CMAKE_SOURCE_DIR}/src/assets/haarcascade_frontalface_default.xml | sed "s/_Users_keith_code_id_reader_minimal_src_assets_haarcascade_frontalface_default_xml/gHaarCascadeData/g" | sed "s/_Users_keith_code_id_reader_minimal_src_assets_haarcascade_frontalface_default_xml_len/gHaarCascadeData_len/g" > ${CMAKE_BINARY_DIR}/haarcascade_data.h
    DEPENDS ${CMAKE_SOURCE_DIR}/src/assets/haarcascade_frontalface_default.xml
    COMMENT "Converting haarcascade_frontalface_default.xml to C++ array"
)

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
target_include_directories(detect_id_card PRIVATE ${OpenCV_INCLUDE_DIRS} src ${CMAKE_BINARY_DIR})

# Add the generated header as a source file for the executable
target_sources(detect_id_card PRIVATE ${CMAKE_BINARY_DIR}/haarcascade_data.h)

# Remove the file copy command as it's no longer needed
# file(COPY src/assets/haarcascade_frontalface_default.xml DESTINATION ${CMAKE_BINARY_DIR}/assets)

add_subdirectory(tests)
