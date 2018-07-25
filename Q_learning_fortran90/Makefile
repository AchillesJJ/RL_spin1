MKLPATH = /opt/intel/mkl/lib/intel64
MKLINCLUDE = /opt/intel/mkl/include

LINKFLAGS =  -Wl,--start-group  $(MKLPATH)/libmkl_intel_lp64.a $(MKLPATH)/libmkl_intel_thread.a $(MKLPATH)/libmkl_core.a -Wl,--end-group -qopenmp -lpthread

F90FLAGS = -O2 $(LINKFLAGS)

FC = ifort -mcmodel=large
LN = ifort -mcmodel=large

FFLAGS = -c
LFLAGS =

OBJECTS = main_mod.o function.o ED.o Watkins_Q_learning.o
result: $(OBJECTS) main.o
	$(LN) $(LFLAGS) -o result main.o $(OBJECTS) $(F90FLAGS)

ED.o: ED.f90
	$(FC) $(FFLAGS) -o ED.o ED.f90

function.o: function.f90
	$(FC) $(FFLAGS) -o function.o function.f90

Watkins_Q_learning.o: Watkins_Q_learning.f90
	$(FC) $(FFLAGS) -o Watkins_Q_learning.o Watkins_Q_learning.f90

main_mod.o: main_mod.f90
	$(FC) $(FFLAGS) -o main_mod.o main_mod.f90

main.o: main.f90
	$(FC) $(FFLAGS) -o main.o main.f90

clean:
	rm result *.o*  *.mod *.e*

delete:
	rm -rf ./check/*.dat