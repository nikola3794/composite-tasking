def get_important_dirs(which_cluster):
    if which_cluster == "eth":
        dirs = {
            "exp_root_dir": "/srv/beegfs02/scratch/switchtasks/data/NIKOLA/Experiments",
            "code_root_dir": "/scratch-second/nipopovic/workspace/cvpr2021/composit_tasking",
            #"data_root_dir": "/srv/beegfs02/scratch/switchtasks/data/NIKOLA/DATA/PASCAL_MT",
            "data_root_dir": "/home/nipopovic/scratch_second/workspace/cvpr2021/DATA/PASCAL_MT",
            }
    elif which_cluster == "aws":
        dirs = {
            "exp_root_dir": "/mnt/efs/fs1/logs/composit_tasking",
            "code_root_dir": "/mnt/efs/fs1/code/composit_tasking",
            "data_root_dir": "/mnt/efs/fs1/DATA/PASCAL_MT",
            }
    else:
        raise NotImplementedError

    return dirs