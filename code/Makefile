ccsrc = $(wildcard *.cpp)
obj = $(ccsrc:.cpp=.o)

CXX = g++

CPPFLAGS = -std=c++11

LDFLAGS = -lm

nn: $(obj)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) nn
