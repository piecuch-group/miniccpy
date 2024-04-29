# Build the ccq_py python module
MODULES := ccsdt_p\
	   dipeom4_p

EXT_SUFFIX := $(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SOURCES := $(addsuffix .f90, $(MODULES))
TARGETS := $(addsuffix $(EXT_SUFFIX), $(MODULES))

# If OpenMP is not available, remove the last two flags, -fopenmp, -lgomp
# For GitHub CI pipeline, remove OpenMP
FFLAGS := -std=f2018 -O3 -ffree-line-length-512# -march=native -mtune=native -mavx -fopenmp -lgomp

.PHONY: all clean

all: $(TARGETS)

%$(EXT_SUFFIX): %.f90
	echo $(MKLFOLDER)
	f2py --f90flags="$(FFLAGS)" -c $< -m $*

clean:
	rm $(TARGETS)
