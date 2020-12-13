#! /bin/bash

RUN_DIR="${PWD}"  # Set to directory where the run should occur
RUN_CMD='python cnn-cifar10-train-PTPROFILE.py'

# Put any vtune setup commands here, like OneAPI or module loading
# e.g. source /opt/intel/oneapi/setvars.sh
module load oneapi
echo -n '===== VTune Being Used =====: ' ; which vtune ; vtune --version

# Load conda and activate environment
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate py37-pytorch-mkl  # Python 3.7, TF 2.2 built with MKL
echo '===== Python Being Used =====' ; which python ; python --version

# Environment variable settings - see "Maximize TensorFlow Performance on CPU"
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=4

VTUNE_OPTS='-finalization-mode=none'  # Finalize with vtune-gui, for compatibility

cd $RUN_DIR
vtune -collect uarch-exploration $VTUNE_OPTS -app-working-dir "${RUN_DIR}" -- ${RUN_CMD}
vtune -collect hotspots $VTUNE_OPTS -app-working-dir "${RUN_DIR}" -- ${RUN_CMD}
echo "Results can be found in ${PWD}"

# Other collection types:
#   vtune -collect performance-snapshot
#   vtune -collect hotspots
#   vtune -collect hpc-performance
