# core/utils/space_utils.py
from gymnasium import spaces
import numpy as np

def are_spaces_equal(space1: spaces.Space, space2: spaces.Space) -> bool:
    """
    Compare two Gymnasium spaces for equality.
    Handles all standard space types including Box, Discrete, Dict, etc.
    """
    if type(space1) != type(space2):
        return False
    
    if isinstance(space1, spaces.Box):
        return (np.array_equal(space1.low, space2.low)) and \
               (np.array_equal(space1.high, space2.high)) and \
               (space1.shape == space2.shape) and \
               (space1.dtype == space2.dtype)
               
    elif isinstance(space1, spaces.Discrete):
        return space1.n == space2.n
    
    elif isinstance(space1, spaces.Dict):
        if space1.spaces.keys() != space2.spaces.keys():
            return False
        return all(are_spaces_equal(space1.spaces[key], space2.spaces[key]) 
                  for key in space1.spaces.keys())
                  
    elif isinstance(space1, spaces.Tuple):
        return len(space1.spaces) == len(space2.spaces) and \
               all(are_spaces_equal(s1, s2) 
                  for s1, s2 in zip(space1.spaces, space2.spaces))
    
    else:
        raise NotImplementedError(f"Space comparison not implemented for {type(space1)}")