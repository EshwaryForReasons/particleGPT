import configurator
import sys
import os
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))

out_dir = os.path.join(script_dir, "job_scripts", "temp")
os.makedirs(out_dir, exist_ok=True)

job_config_file_path = sys.argv[1]
job_config_file_name = os.path.splitext(os.path.basename(job_config_file_path))[0]

# Create temp script and definition files in job_scripts/temp

definition_script =f"""#!/bin/bash
#SBATCH --nodes={configurator.nodes}
#SBATCH --time={configurator.time_duration}
#SBATCH --constraint={configurator.constraint}
#SBATCH --gpus={configurator.gpus}
#SBATCH --cpus-per-task={configurator.cpus_per_task}
#SBATCH --ntasks-per-node={configurator.ntasks_per_node}
#SBATCH --account={configurator.account}
#SBATCH --qos={configurator.quality_of_service}

srun -n 1 bash temp_sjob_script_{job_config_file_name}.sh
"""

job_script = f"""
cd {script_dir}/job_scripts/temp
{f"shifter --image={configurator.shifter_image} /bin/bash" if configurator.use_shifter else ""}
{configurator.command}
"""

out_dir = os.path.join(script_dir, 'job_scripts', 'temp')
with (open(f"{out_dir}/temp_sjob_{job_config_file_name}.sh", 'w') as job_file,
    open(f"{out_dir}/temp_sjob_def_{job_config_file_name}.sl", 'w') as def_file):
    job_file.write(job_script)
    def_file.write(definition_script)
    
# Submit job
result = subprocess.run(f"sbatch {out_dir}/temp_sjob_def_{job_config_file_name}.sl", shell=True, capture_output=True, text=True)

if result.stderr:
    RED='\033[0;31m'
    RESET='\033[0m'
    print(RED)
    print("Job submission failed with error")
    print("-------------------")
    print(result.stderr, end='')
    print("-------------------")
    print(RESET)
else:
    job_id = result.stdout.split()[-1]
    print("Job submitted with id", job_id)
    
    with open('current_jobs.md', 'a') as current_jobs_file:
        current_jobs_file.write(f'{job_config_file_name} (queued {job_id})\n')