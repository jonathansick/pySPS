F90 = gfortran
F90FLAGS = -O3 -march=native -cpp

PROGS = fspsq

COMMON = sps_vars.o nrtype.o sps_utils.o nr.o nrutil.o compsp.o \
ssp_gen.o getmags.o gasdev.o spline.o splint.o ran.o ran_state.o \
locate.o qromb.o polint.o trapzd.o tridag.o sps_setup.o ran2.o \
pz_convol.o get_tuniv.o imf.o imf_weight.o add_dust.o getspec.o \
add_bs.o mod_hb.o add_remnants.o getindx.o velbroad.o mod_agb.o \
write_isochrone.o sfhstat.o

all: $(PROGS)

clean:
	rm -rf *.o *.mod *.MOD *~

fspsq: fspsq.o $(COMMON)
	$(F90) -o fspsq fspsq.o $(COMMON)

fspsq.o: sps_vars.o nrtype.o nr.o sps_utils.o

sps_utils.o: nrutil.o nr.o sps_vars.o

%.o: %.f90
	$(F90) $(F90FLAGS) -o $@ -c $<

%.o: fsps/%.f90
	$(F90) $(F90FLAGS) -o $@ -c $<

%.o: fsps/nr/%.f90
	$(F90) $(F90FLAGS) -o $@ -c $<
