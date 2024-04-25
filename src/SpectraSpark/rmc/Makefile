
# CPP = g++
CPP = icpx
OMP = -qopenmp
EIGEN =  $(shell pkg-config --cflags-only-I eigen3)
PYBIND = $(shell python3 -m pybind11 --includes)
PY_EXT = $(shell python3.11-config --extension-suffix)
CPPFLAGS = -Wall -std=c++23 -fPIC $(EIGEN)
SRCDIR = src/SpectraSpark

all: example

clean:
	rm -f *.o *.so *.pyc *.out

wrap: rmc.o wrap.cpp rmc.hpp
	$(CPP) $(CPPFLAGS) -shared $(PYBIND) wrap.cpp rmc.o -o rmc$(PY_EXT)

rmc.o: rmc.cpp rmc.hpp
	$(CPP) $(CPPFLAGS) -c rmc.cpp -o rmc.o

runfit: runfit.out
	./runfit.out

runfit.out: rmc.o runfit.cpp
	$(CPP) $(CPPFLAGS) rmc.o runfit.cpp -o runfit.out

rmc2d.o: rmc2d.cpp rmc2d.hpp
	$(CPP) $(CPPFLAGS) -O0 $(OMP) -c rmc2d.cpp -o rmc2d.o

rmc2d_util.o: rmc2d_util.cpp rmc2d_util.hpp
	$(CPP) $(CPPFLAGS) $(OMP) -c rmc2d_util.cpp -o rmc2d_util.o

rmc2d_test.out: rmc2d_test.cpp rmc2d.o rmc2d_util.o
	$(CPP) $(CPPFLAGS) $(OMP) rmc2d_test.cpp rmc2d.o rmc2d_util.o \
								-o rmc2d_test.out