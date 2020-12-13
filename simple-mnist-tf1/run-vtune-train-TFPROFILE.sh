#! /bin/bash

# This script runs VTune to collect microarchitecture data on
# a TensorFlow model that is also recording TensorFlow profinling data

RUN_DIR="${PWD}"  # Set to directory where the run should occur
RUN_CMD='python mnist-deep-simplified-TFPROFILE.py'

# Put any vtune setup commands here, like OneAPI or module loading
# e.g. module load oneapi
source "${HOME}/intel/oneapi/setvars.sh"
echo -n '===== VTune Being Used =====: ' ; which vtune ; vtune --version

# Load conda and activate environment
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate py37-tf114-mkl  # Python 3.7, TF 1.14 built with MKL
echo '===== Python Being Used =====' ; which python ; python --version

# Environment variable settings - see "Maximize TensorFlow Performance on CPU"
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=4

VTUNE_OPTS='-finalization-mode=none'  # Finalize with vtune-gui, for compatibility

cd $RUN_DIR
vtune -collect uarch-exploration $VTUNE_OPTS -app-working-dir "${RUN_DIR}" -- ${RUN_CMD}
echo "Results can be found in ${PWD}"

# Example of collection with custom collector:
#    COLLECT_CMD="${HOME}/bin/vtune-custom-tf1-collector.sh ./tf-profile-train"
#    vtune -collect uarch-exploration -custom-collector="${COLLECT_CMD}" $VTUNE_OPTS -app-working-dir "${RUN_DIR}" -- ${RUN_CMD}


# Other collection types:
#   vtune -collect performance-snapshot
#   vtune -collect hotspots
#   vtune -collect hotspots -knob sampling-mode=hw -knob sampling-interval=0.5
#   vtune -collect hpc-performance
