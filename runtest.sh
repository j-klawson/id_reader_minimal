#!/bin/bash

# Clean previous build
rm -rf build

# Create build directory and navigate into it
mkdir -p build
cd build

# Configure and build the project
cmake ..
cmake --build .

# Navigate back to the root directory
cd ..

# Run the test
./bin/detect_id_card --debug ./tests/test_images/dl-ontario-front.jpg
