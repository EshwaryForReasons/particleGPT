Job scripts for training and sampling data as most require more than 4 hours.

Template for the job script:

```
#!/bin/bash -l
module load pytorch

cd $HOME/hadron_collisions/particleGPT

cat << EOF > sjob_mass_train.sl
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=21:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --account=m3443
#SBATCH --qos=regular

srun -n 1 bash job_scripts/train_dataset_2_1_1.sh

EOF

sbatch sjob_mass_train.sl
```