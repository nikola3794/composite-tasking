import os
import string
import random
import json
import time
from zipfile import ZipFile


class ExpDirManagement:
    
    @staticmethod
    def create_exp_main_dir(exp_root_dir, short_description):
        """
        Create/retrieve the main experiment directory.
        Comments in the code explain how the experiment main directory is generated. 
        """

        # Experiments are divided into different directories.
        # One directory for a different experiment starting day.
        # (i.e. different folder for 03.05, 04.05, 05.05, etc...)
        experiment_main_dir = os.path.join(exp_root_dir, date_to_str())
        if not os.path.isdir(experiment_main_dir):
            try:
                os.mkdir(experiment_main_dir)
            except FileExistsError:
                # If it has already been created due to a parallel faster process, just ignore
                # TODO Print some log message
                pass 

        # Inside the current experiment date directory, each experiment gets a 
        # directory who's name starts with an id num.
        # The id number is simply generated as the (biggest_id_number + 1). 
        tmp_exps = [e for e in os.listdir(experiment_main_dir) if os.path.isdir(os.path.join(experiment_main_dir, e))]
        tmp_exp_nums = [int(tmp.split("_")[0]) for tmp in tmp_exps if tmp.split("_")[0].isnumeric()]
        tmp_exp_nums.sort()
        exp_id = (tmp_exp_nums[-1] if len(tmp_exp_nums) > 0 else 0) + 1

        # Make a experiment directory name starting with the experiment id.
        exp_name = f"{exp_id}_{short_description}"
        experiment_main_dir = os.path.join(experiment_main_dir, exp_name)
        try:
            os.mkdir(experiment_main_dir)
        except FileExistsError:
            # If a folder has already been created from an parallel faswter process,
            #  add a few random characters to the folder name to make it unique
            experiment_main_dir += ''.join(random.choice(string.ascii_letters) for i in range(4))
            os.mkdir(experiment_main_dir)

        return experiment_main_dir
    
    @staticmethod
    def save_code_to_zip(code_root_dir, save_dir, ignore_hidden=True):
        """
        Save current code state in the experiment directory
        """
        if code_root_dir is None:
            return

        with ZipFile(os.path.join(save_dir, "code_screenshot.zip"), 'w') as zip_obj:
            # Iterate over all the files in directory
            for folder_name, subfolders, filenames in os.walk(code_root_dir):
                
                if ignore_hidden:
                    filenames = [f for f in filenames if not f[0] == '.']
                    subfolders[:] = [d for d in subfolders if not d[0] == '.']
                
                for filename in filenames:
                    #create complete filepath of file in directory
                    file_path = os.path.join(folder_name, filename)
                    # Add file to zip
                    zip_obj.write(file_path)

    @staticmethod
    def save_dict_to_json(dictionary, save_path, file_name):
        # Save the experiment config
        if file_name [-5:] != ".json":
            file_name += ".json"
        json_path = os.path.join(save_path, file_name)
        if not os.path.isfile(json_path):
            with open(json_path, "w") as fh:
                json.dump(dictionary, fh)
        else:
            assert FileExistsError

def date_to_str():
    """
    :return: String of current date in Y-M-D format
    """
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string