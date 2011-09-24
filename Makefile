F90 = gfortran
F90FLAGS = -O

fspsq: fspsq.o fspsq.f90
	$(F90) -o fspsq fspsq.o

%.o : %.f90
	$(F90) $(F90FLAGS) -o $@ -c $<

