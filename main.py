import os
import cv2
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
    
    @staticmethod
    def optimal_rotations():
        predefined_matrices = [
            np.eye(3),
            np.array([[0.333333, 0.666667, 0.666667], 
                      [0.666667, 0.333333, -0.666667], 
                      [-0.666667, 0.666667, -0.333333]]),
            np.array([[0.577350, 0.211297, 0.788682], 
                      [-0.577350, 0.788668, 0.211352], 
                      [0.577350, 0.577370, -0.577330]]),
            np.array([[0.577350, 0.408273, 0.707092], 
                      [-0.577350, -0.408224, 0.707121], 
                      [0.577350, -0.816497, 0.000029]]),
            np.array([[0.332572, 0.910758, 0.244778], 
                      [-0.910887, 0.242977, 0.333536], 
                      [-0.244295, 0.333890, -0.910405]]),
            np.array([[0.243799, 0.910726, 0.333376], 
                      [0.910699, -0.333174, 0.244177], 
                      [-0.333450, -0.244075, 0.910625]])
        ]
        return predefined_matrices
    
class ColorTransfer:
    def __init__(self, eps=1e-6, matrix_count=6, color_channels=3):
        self.eps = eps
        self.rotation_matrices = []

        if color_channels == 3:
            self.rotation_matrices = Rotations.optimal_rotations()
        else:
            self.rotation_matrices = Rotations.random_rotations(matrix_count, color_channels)

    def pdf_transfer(self, img_input, img_reference):
        h, w, c = img_input.shape
        input_pixels = img_input.reshape(-1, c).T / 255.0
        reference_pixels = img_reference.reshape(-1, c).T / 255.0
        output_pixels = self._transfer_pixels(input_pixels, reference_pixels)
        
        output_pixels = np.clip(output_pixels * 255.0, 0, 255).astype("uint8")
        output_img = output_pixels.T.reshape(h, w, c)

        return output_img
    
    def _transfer_pixels(self, arr_input, arr_reference):
        output = arr_input.copy()
        for rotation in self.rotation_matrices:
            rotated_input = rotation @ arr_input
            rotated_reference = rotation @ arr_reference

            output = self._match_distribution(rotated_input, rotated_reference, rotation)
        return output
    
    def _match_distribution(self, arr_input, arr_reference, rotation):
        mean_in, std_in = arr_input.mean(axis=1), arr_input.std(axis=1) + self.eps
        mean_ref, std_ref = arr_reference.mean(axis=1), arr_reference.std(axis=1) + self.eps

        adjusted = ((arr_input - mean_in[:, None]) * (std_ref / std_in)[:, None]) + mean_ref[:, None]
        return np.linalg.inv(rotation) @ adjusted


def demo():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    input_folder = os.path.join(cur_dir, "input")
    reference_folder = os.path.join(cur_dir, "reference")
    output_folder = os.path.join(cur_dir, "output")
    os.makedirs(output_folder, exist_ok=True)

    input_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    reference_files = sorted(f for f in os.listdir(reference_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    for input_file in input_files:
        input_path = os.path.join(input_folder, input_file)
        input_img = cv2.imread(input_path)

        if input_img is None:
            print(f"Error: Could not read input image {input_file}. Skipping...")
            continue

        for reference_file in reference_files:
            reference_path = os.path.join(reference_folder, reference_file)
            reference_img = cv2.imread(reference_path)

            if reference_img is None:
                print(f"Error: Could not read reference image {reference_file}. Skipping...")
                continue

            output_img = ColorTransfer.pdf_transfer(input_img, reference_img)

            output_file_name = f"{os.path.splitext(input_file)[0]}_{os.path.splitext(reference_file)[0]}.png"
            output_path = os.path.join(output_folder, output_file_name)

            cv2.imwrite(output_path, output_img)
            print(f"Output saved to {output_path}")

if __name__ == "__main__":
    demo()