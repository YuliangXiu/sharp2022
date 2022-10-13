import numpy as np

rotate = {
    'HumanAct12': [1., -1., -1.],
    'CMU_Mocap': [0.05, 0.05, 0.05],
    'UTD_MHAD': [-1., 1., -1.],
    'Human3.6M': [-0.001, -0.001, 0.001],
    'NTU': [-1., 1., -1.],
    'SHARP': [1., 1., 1.],
}
 

def transform(name, arr: np.ndarray):
    arr[:,:,1] -= arr[:,8,1]
    return arr
