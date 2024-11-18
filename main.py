import numpy as np
from scipy.spatial.transform import special_ortho_group

class Rotations:
    @staticmethod
    def generate_random_rotation(dim=3):
        return special_ortho_group.rvs(dim)
    
    @classmethod
    def random_rotations(cls, num_matrices, dim=3):
        if num_matrices < 1:
            raise ValueError("Number of matrices must be at least 1.")
        return [cls.generate_random_rotation(dim) for _ in range(num_matrices)]