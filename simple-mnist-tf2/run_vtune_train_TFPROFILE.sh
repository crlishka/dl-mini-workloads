#! /bin/bash

RUN_DIR="${PWD}"  # Change this if running in other directory than PWD
RUN_CMD='python simple_mnist_train_TFPROFILE.py'

# Put any vtune setup commands here, like OneAPI or module loading
source "${HOME}/intel/oneapi/setvars.sh"
echo -n '===== VTune Being Used =====: ' ; which vtune ; vtune --version

# Load conda and activate environment
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate py37-tf22-mkl  # Python 3.7, TF 2.2 built with MKL
echo '===== Python Being Used =====' ; which python ; python --version

# Environment variable settings - see "Maximize TensorFlow Performance on CPU"
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=4

VTUNE_OPTS='-finalization-mode=none'  # Finalize with vtune-gui, for compatibility

cd $RUN_DIR
vtune -collect uarch-exploration ${VTUNE_OPTS} -app-working-dir "${RUN_DIR}" -- ${RUN_CMD}
vtune -collect performance-snapshot ${VTUNE_OPTS} -app-working-dir "${RUN_DIR}" -- ${RUN_CMD}
echo "Results can be found in ${PWD}"
