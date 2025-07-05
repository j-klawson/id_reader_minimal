CXX = g++
CXXFLAGS = -std=c++17 -Wall
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`
TARGET = detect_id_card_with_portrait

all: $(TARGET)

$(TARGET): detect_id_card_with_portrait.cpp
	$(CXX) $(CXXFLAGS) detect_id_card_with_portrait.cpp -o $(TARGET) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET) *.jpg
