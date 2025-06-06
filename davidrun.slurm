#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --partition=ampere
#SBATCH --nodelist=sw-amp02
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=hello_%j.out
#SBATCH --error=hello_%j.err

# Print diagnostic info
echo "Job started on $(hostname) at $(date)"
echo "Activating virtual environment..."

# Activate virtual environment
source /cephfs/store/gr-mc2473/jl2268/JAX_NestedSampling_Stream/David/bin/activate

echo "Using Python: $(which python)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Enable full JAX tracebacks for debugging
export JAX_TRACEBACK_FILTERING=off

python -c "import jax; print('JAX devices:', jax.devices())"

# Navigate to where the script is
cd /cephfs/store/gr-mc2473/jl2268/JAX_NestedSampling_Stream

# Run the inference script
echo "Running inference script..."
python main_v2.py

echo "Job finished at $(date)"
