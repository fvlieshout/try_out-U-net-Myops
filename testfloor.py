import numpy as np
import torch
import torchvision.transforms as transforms

ar1 = np.array([[0,1,1], [0,0,0], [0,0,1]])
ar2 = np.array([0,1,2])
arr=np.array([[0,0,0],[1,1,0],[0,0,1],[0,0,0]])

# print(((ar1==0) | (ar1==1)).all())
# print(((ar2==0) | (ar2==1)).all())
def first_nonzero(arr, axis, invalid_val=int(1e3)):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def get_bounding_box_slice(arr):
    values = {}
    for row in range(arr.shape[0]):
        if 1 in arr[row]:
            values['upper'] = row
            break
    for row in sorted(range(arr.shape[0]), reverse=True):
        if 1 in arr[row]:
            values['lower'] = row
            break
    for column in range(arr.shape[1]):
        if 1 in arr[:, column]:
            values['left'] = column
            break
    for column in sorted(range(arr.shape[1]), reverse=True):
        if 1 in arr[:, column]:
            values['right'] = column
            break
    return values

# print(str('stringetje'))
# print(ar1)
# print(first_nonzero(ar1, axis=0))
# print(first_nonzero(ar1, axis=0).min(axis=0))
# print(get_bounding_box_slice(ar1))
# print(arr)
# print(first_nonzero(arr, axis=0))
# print(first_nonzero(arr, axis=0).min(axis=0))

tensora = torch.Tensor([[0,1],[2,3]])
print(tensora)
tensorb = torch.rot90(tensora, 1, [0,1])
tensorc = torch.rot90(tensora, 2, [0,1])
tensord = torch.rot90(tensora, 3, [0,1])
print(tensorb)
print(tensorc)
print(tensord)