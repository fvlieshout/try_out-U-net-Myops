import os
from shutil import copy
import collections
import random
import shutil

folder_loc = os.getcwd()

def remove_extra(folder_loc):
    for f in os.listdir(folder_loc):
        numbering = f.split('.')[0][-1]
        if numbering == '2' or numbering == '3':
            os.remove(os.path.join(folder_loc, f))

def rename_files(folder_loc):
    for f in os.listdir(folder_loc):
        if 'mask' not in f:
            full_old_name = os.path.join(folder_loc, f)
            patid = f.split('_')[0]
            extension = f.split('.')[-1]
            if extension == 'py':
                continue
            mask_type = f.split('-')[1].split('.')[0]
            new_name = f"{patid}_LGE_{mask_type}_mask.{extension}"
            full_new_name = os.path.join(folder_loc, new_name)
            os.rename(full_old_name, full_new_name)

def get_patids(folder_loc):
    ids = []
    for f in os.listdir(folder_loc):
        patid = f.split('_')[0]
        ids.append(patid)
    return ids

def copy_niftis(pat_ids, LGE_folder_orig, LGE_folder_target, uitzonderingen):
    for f in os.listdir(LGE_folder_orig):
        patid = f.split('_')[0]
        if patid in pat_ids and 'PSIR' in f and patid not in uitzonderingen:
            full_old_location = os.path.join(LGE_folder_orig, f)
            full_new_location = os.path.join(LGE_folder_target, f)
            if not os.path.isfile(full_new_location):
                copy(full_old_location, full_new_location)

def remove_cycle_duplicates(folder_loc):
    pat_ids = []
    for f in os.listdir(folder_loc):
        pat_ids.append(f.split('_')[0])
    duplicates = [item for item, count in collections.Counter(pat_ids).items() if count > 1]
    for duplicate in duplicates:
        duplicate_list = []
        for f in os.listdir(folder_loc):
            pat_id = f.split('_')[0]
            if pat_id == duplicate:
                duplicate_list.append(f)
        dup1, dup2 = duplicate_list
        if 'seq0' in dup1 and 'seq0' not in dup2:
            os.remove(os.path.join(folder_loc, dup2))
        elif 'seq0' in dup2 and 'seq0' not in dup1:
            os.remove(os.path.join(folder_loc, dup1))
        else:
            print(f'Not clear which file to remove for {duplicate}')

def create_test_files_and_move(patids, num_test, current_folder):
    # choose test pats
    test_ids = random.sample(patids, num_test)
    orig_folders = {'LGE' :  os.path.join(current_folder, 'train', 'LGE_niftis'),
                    'myo' :  os.path.join(current_folder, 'train', 'myo'),
                    'aankleuring' :  os.path.join(current_folder, 'train', 'aankleuring'),}
    target_folders = {'LGE' :  os.path.join(current_folder, 'test', 'LGE_niftis'),
                    'myo' :  os.path.join(current_folder, 'test', 'myo'),
                    'aankleuring' :  os.path.join(current_folder, 'test', 'aankleuring'),}

    #copy LGE
    for f in os.listdir(orig_folders['LGE']):
        if f.split('_')[0] in test_ids:
            shutil.move(os.path.join(orig_folders['LGE'], f), target_folders['LGE'], f)

    #copy myo
    for f in os.listdir(orig_folders['myo']):
        if f.split('_')[0] in test_ids:
            shutil.move(os.path.join(orig_folders['myo'], f), target_folders['myo'], f)

    #copy aankleuring
    for f in os.listdir(orig_folders['aankleuring']):
        if f.split('_')[0] in test_ids:
            shutil.move(os.path.join(orig_folders['aankleuring'], f), target_folders['aankleuring'], f)


curr_folder = os.getcwd()
myo_folder = os.path.join(curr_folder, 'train\\myo')
aankleuring_folder = os.path.join(curr_folder, 'train\\aankleuring')
LGE_folder_orig = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis'
LGE_folder_target = os.path.join(curr_folder, 'train\\LGE_niftis')
# rename_files(aankleuring_folder)
patids = get_patids(myo_folder)
uitzonderingen = ['DRAUMC0008', 'DRAUMC0219', 'DRAUMC0235', 'DRAUMC0315', 'DRAUMC0365', 'DRAUMC0380', 'DRAUMC0588']
# copy_niftis(patids, LGE_folder_orig, LGE_folder_target, uitzonderingen)
# remove_cycle_duplicates(LGE_folder_target)
create_test_files_and_move(patids, 7, curr_folder)

