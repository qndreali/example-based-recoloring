import os
import cv2
import numpy as np
from scipy.stats import special_ortho_group

class Rotations:
    @staticmethod
    def random_rotations(num_matrices, dim=3):
        if num_matrices < 1:
            raise ValueError("Number of matrices must be at least 1.")
        return [special_ortho_group.rvs(dim) for _ in range(num_matrices)]
    
    @staticmethod
    def optimal_rotations():
        return [
            np.eye(3),
            [[0.333, 0.667, 0.667], [0.667, 0.333, -0.667], [-0.667, 0.667, -0.333]],
            [[0.577, 0.211, 0.789], [-0.577, 0.789, 0.211], [0.577, 0.577, -0.577]],
            [[0.577, 0.408, 0.707], [-0.577, -0.408, 0.707], [0.577, -0.816, 0.000]],
            [[0.333, 0.911, 0.245], [-0.911, 0.243, 0.334], [-0.244, 0.334, -0.910]],
            [[0.244, 0.911, 0.333], [0.911, -0.333, 0.244], [-0.333, -0.244, 0.911]],
        ]
    
class ColorTransfer:
    def __init__(self, eps=1e-6, matrix_count=6, color_channels=3):
        self.eps = eps
        self.rotations = (
            Rotations.optimal_rotations() if color_channels == 3 
            else Rotations.random_rotations(matrix_count, color_channels)
        )

    def pdf_transfer(self, img_input, img_reference):
        h, w, c = img_input.shape
        input_pixels = img_input.reshape(-1, c).T / 255.0
        reference_pixels = img_reference.reshape(-1, c).T / 255.0

        for rotation in self.rotations:
            input_pixels = self._match_distribution(
                rotation @ input_pixels, rotation @ reference_pixels, rotation
            )
        return np.clip((input_pixels.T.reshape(h, w, c) * 255), 0, 255).astype("uint8")
    
    def _match_distribution(self, arr_input, arr_reference, rotation):
        mean_in, std_in = arr_input.mean(1), arr_input.std(1) + self.eps
        mean_ref, std_ref = arr_reference.mean(1), arr_reference.std(1) + self.eps
        adjusted = ((arr_input - mean_in[:, None]) * (std_ref / std_in)[:, None]) + mean_ref[:, None]
        return np.linalg.inv(rotation) @ adjusted
    
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at {path}.")
    return img

def create_comparison_image(input_img, ref_img, output_img):
    height = max(input_img.shape[0], ref_img.shape[0], output_img.shape[0])
    input_img = cv2.resize(input_img, (int(input_img.shape[1] * height / input_img.shape[0]), height))
    ref_img = cv2.resize(ref_img, (int(ref_img.shape[1] * height / ref_img.shape[0]), height))
    output_img = cv2.resize(output_img, (int(output_img.shape[1] * height / output_img.shape[0]), height))
    return np.hstack((input_img, ref_img, output_img))

def process_images(input_folder, reference_folder, output_folder, comparison_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(comparison_folder, exist_ok=True)
    
    os.makedirs(output_folder, exist_ok=True)
    color_transfer = ColorTransfer()

    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    reference_files = [f for f in os.listdir(reference_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for input_file in input_files:
        input_img = load_image(os.path.join(input_folder, input_file))
        for reference_file in reference_files:

            if os.path.splitext(input_file)[0] == os.path.splitext(reference_file)[0]:
                print(f"Skipping: {input_file} and {reference_file} (same file name)")
                continue
             
            ref_img = load_image(os.path.join(reference_folder, reference_file))
            output_img = color_transfer.pdf_transfer(input_img, ref_img)
            output_path = os.path.join(output_folder, f"{os.path.splitext(input_file)[0]}_{os.path.splitext(reference_file)[0]}.png")
            cv2.imwrite(output_path, output_img)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    process_images(
        os.path.join(base_dir, "input"),
        os.path.join(base_dir, "reference"),
        os.path.join(base_dir, "output"),
    )