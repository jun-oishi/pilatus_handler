
CPP = g++
EIGEN =  $(shell pkg-config --cflags-only-I eigen3)
PYBIND = $(shell python3 -m pybind11 --includes)
PY_EXT = $(shell python3.11-config --extension-suffix)
CPPFLAGS = -O3 -Wall -shared -std=c++23 -fPIC $(EIGEN)
SRCDIR = src/SpectraSpark

all: example


example: $(SRCDIR)/cppmod/example.o $(SRCDIR)/cppmod/wrapper.cpp
	$(CPP) $(CPPFLAGS) $(PYBIND) $(SRCDIR)/cppmod/wrapper.cpp $(SRCDIR)/cppmod/example.o  -o $(SRCDIR)/cppmod/example$(PY_EXT)

$(SRCDIR)/cppmod/example.o: $(SRCDIR)/cppmod/example.cpp
	$(CPP) $(CPPFLAGS) -c $(SRCDIR)/cppmod/example.cpp -o $(SRCDIR)/cppmod/example.o