import os
import subprocess


base_dir = "/fsx/zhuokai/TabularFM/"
epoch_nums = [
    "1",
    "2",
    "3",
    "4",
    "5",
]

for cur_epoch_num in epoch_nums:
    job_name = f"data_processing_epoch_{cur_epoch_num}"
    script_path = f"./scripts/data_processing/{job_name}.slurm"
    # get the folder path and create it if it doesn't exist
    folder_path = os.path.dirname(script_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(script_path, "w") as f:
        lines_to_write = [
            "#!/bin/bash\n",
            "#\n",
            f"#SBATCH --chdir={base_dir}\n",
            f"#SBATCH --gres=gpu:0\n",
            "#SBATCH -c 64\n",
            f"#SBATCH --job-name={job_name}\n",
            "#SBATCH --mem 16G\n",
            f"#SBATCH --output=/fsx/zhuokai/TabularFM/slurm/data_processing/{job_name}.stdout\n",
            f"#SBATCH --error=/fsx/zhuokai/TabularFM/slurm/data_processing/{job_name}.stderr\n",
            "\n",
            f"python pre_data_tokenizer.py --save_path ./data/precomputed/epoch{cur_epoch_num}\n",
        ]
        for cur_line in lines_to_write:
            f.write(cur_line)
        f.close()

    subprocess.run(
        [
            "sbatch",
            f"{script_path}",
        ]
    )
    print(f"Submitted task for {job_name}\n")
