cmake_minimum_required(VERSION 3.10)

project(IDCardReaderTests LANGUAGES CXX)

add_executable(id_reader_tests test_main.cpp)

target_link_libraries(id_reader_tests GTest::gtest_main)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_test(NAME OntarioIDCardTest COMMAND id_reader_tests)
