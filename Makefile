F90 = gfortran
F90FLAGS = -O3 -march=native

PROGS = fspsq

COMMON = $(SPS_HOME)src/sps_vars.o $(SPS_HOME)src/nrtype.o \
$(SPS_HOME)src/sps_utils.o $(SPS_HOME)src/nr.o $(SPS_HOME)src/nrutil.o \
$(SPS_HOME)src/compsp.o $(SPS_HOME)src/ssp_gen.o $(SPS_HOME)src/getmags.o \
$(SPS_HOME)src/gasdev.o $(SPS_HOME)src/spline.o $(SPS_HOME)src/splint.o \
$(SPS_HOME)src/ran.o $(SPS_HOME)src/ran_state.o \
$(SPS_HOME)src/locate.o $(SPS_HOME)src/qromb.o $(SPS_HOME)src/polint.o \
$(SPS_HOME)src/trapzd.o $(SPS_HOME)src/tridag.o $(SPS_HOME)src/sps_setup.o \
$(SPS_HOME)src/ran2.o $(SPS_HOME)src/pz_convol.o $(SPS_HOME)src/get_tuniv.o \
$(SPS_HOME)src/imf.o $(SPS_HOME)src/imf_weight.o $(SPS_HOME)src/add_dust.o \
$(SPS_HOME)src/getspec.o $(SPS_HOME)src/add_bs.o $(SPS_HOME)src/mod_hb.o \
$(SPS_HOME)src/add_remnants.o $(SPS_HOME)src/getindx.o \
$(SPS_HOME)src/velbroad.o $(SPS_HOME)src/mod_agb.o \
$(SPS_HOME)src/write_isochrone.o $(SPS_HOME)src/sfhstat.o

all: $(PROGS)

fspsq: fspsq.o fspsq.f90
	$(F90) -o fspsq fspsq.o $(COMMON)

%.o: %.f90
	$(F90) $(F90FLAGS) -o $@ -c $<


