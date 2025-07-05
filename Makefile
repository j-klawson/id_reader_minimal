CXX = g++
CXXFLAGS = -std=c++17 -Wall
PKG_CFLAGS = `pkg-config --cflags opencv4`
PKG_LIBS = `pkg-config --libs opencv4`

SRCS = main.cpp detect_id_card.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = detect_id_card

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(PKG_LIBS)

%.o: %.cpp detect_id_card.h
	$(CXX) $(CXXFLAGS) $(PKG_CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) *.jpg debug.log
