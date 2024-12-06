import os
import cv2
import numpy as np
from scipy.stats import special_ortho_group

class Rotations:
    """
    Handles generation of rotation matrices for color transfer.

    Methods:
        - random_rotations(num_matrices, dim=3): Generates random rotation matrices.
        - optimal_rotations(): Returns predefined optimal rotation matrices.
    """
    
    @staticmethod
    def random_rotations(num_matrices, dim=3):
        """
        Generates a list of random rotation matrices.
        
        Args:
            num_matrices (int): Number of matrices to generate.
            dim (int): Dimension of the rotation matrices (default is 3).
        
        Returns:
            list: A list of `dim x dim` numpy arrays representing rotation matrices.

        Raises:
            ValueError: If `num_matrices` is less than 1.
        """
        if num_matrices < 1:
            raise ValueError("Number of matrices must be at least 1.")
        return [special_ortho_group.rvs(dim) for _ in range(num_matrices)]
    
    @staticmethod
    def optimal_rotations():
        """
        Provides a list of optimal rotation matrices used in color distribution transfer,
        inspired by methods described in the paper:
    
        F. Pitié, A. Kokaram, and R. Dahyot, "Automated color grading using color distribution transfer,"
        Computer Vision and Image Understanding, vol. 107, no. 1–2, pp. 123–137, 2007.
        DOI: 10.1016/j.cviu.2006.11.011
    
        Returns:
            list: A list of 3x3 numpy arrays representing rotation matrices.
        """
        return [
            np.eye(3),
            [[0.333, 0.667, 0.667], [0.667, 0.333, -0.667], [-0.667, 0.667, -0.333]],
            [[0.577, 0.211, 0.789], [-0.577, 0.789, 0.211], [0.577, 0.577, -0.577]],
            [[0.577, 0.408, 0.707], [-0.577, -0.408, 0.707], [0.577, -0.816, 0.000]],
            [[0.333, 0.911, 0.245], [-0.911, 0.243, 0.334], [-0.244, 0.334, -0.910]],
            [[0.244, 0.911, 0.333], [0.911, -0.333, 0.244], [-0.333, -0.244, 0.911]],
        ]
    
class ColorTransfer:
    """
    Handles color distribution transfer between input and reference images.

    Attributes:
        eps (float): Small epsilon value to avoid division by zero.
        rotations (list): Rotation matrices used in the transformation.

    Methods:
        - pdf_transfer(img_input, img_reference): Matches the color distribution of the input image 
          to that of the reference image.
    """
    
    def __init__(self, eps=1e-6, matrix_count=6, color_channels=3):
        """
        Initializes the ColorTransfer class.
        
        Args:
            eps (float): Small value to avoid division by zero in calculations.
            matrix_count (int): Number of random rotation matrices (if applicable).
            color_channels (int): Number of color channels in the images (default is 3).
        """
        self.eps = eps
        self.rotations = (
            Rotations.optimal_rotations() if color_channels == 3 
            else Rotations.random_rotations(matrix_count, color_channels)
        )

    def pdf_transfer(self, img_input, img_reference):
        """
        Matches the color distribution of the input image to that of the reference image.

        Args:
            img_input (numpy.ndarray): Input image array.
            img_reference (numpy.ndarray): Reference image array.

        Returns:
            numpy.ndarray: Output image with color distribution matched to the reference image.
        """
        h, w, c = img_input.shape
        input_pixels = img_input.reshape(-1, c).T / 255.0
        reference_pixels = img_reference.reshape(-1, c).T / 255.0

        for rotation in self.rotations:
            input_pixels = self._match_distribution(
                rotation @ input_pixels, rotation @ reference_pixels, rotation
            )
        return np.clip((input_pixels.T.reshape(h, w, c) * 255), 0, 255).astype("uint8")
    
    def _match_distribution(self, arr_input, arr_reference, rotation):
        """
        Matches the distribution of input pixels to reference pixels under a rotation matrix.

        Args:
            arr_input (numpy.ndarray): Transformed input pixels.
            arr_reference (numpy.ndarray): Transformed reference pixels.
            rotation (numpy.ndarray): Rotation matrix.

        Returns:
            numpy.ndarray: Adjusted input pixels with matched distribution.
        """
        mean_in, std_in = arr_input.mean(1), arr_input.std(1) + self.eps
        mean_ref, std_ref = arr_reference.mean(1), arr_reference.std(1) + self.eps
        adjusted = ((arr_input - mean_in[:, None]) * (std_ref / std_in)[:, None]) + mean_ref[:, None]
        return np.linalg.inv(rotation) @ adjusted
    
def load_image(path):
    """
    Loads an image from the specified file path.
    
    Args:
        path (str): Path to the image file.
    
    Returns:
        numpy.ndarray: Loaded image array.
    
    Raises:
        ValueError: If the image cannot be read from the file.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at {path}.")
    return img

def create_comparison_image(input_img, ref_img, output_img):
    """
    Creates a side-by-side comparison image.
    
    Args:
        input_img (numpy.ndarray): Input image.
        ref_img (numpy.ndarray): Reference image.
        output_img (numpy.ndarray): Output image.
    
    Returns:
        numpy.ndarray: Comparison image with input, reference, and output images side by side.
    """
    height = max(input_img.shape[0], ref_img.shape[0], output_img.shape[0])
    input_img = cv2.resize(input_img, (int(input_img.shape[1] * height / input_img.shape[0]), height))
    ref_img = cv2.resize(ref_img, (int(ref_img.shape[1] * height / ref_img.shape[0]), height))
    output_img = cv2.resize(output_img, (int(output_img.shape[1] * height / output_img.shape[0]), height))
    return np.hstack((input_img, ref_img, output_img))

def process_images(input_folder, reference_folder, output_folder, comparison_folder):
    """
    Processes images for color distribution transfer.
    
    Args:
        input_folder (str): Path to the input images folder.
        reference_folder (str): Path to the reference images folder.
        output_folder (str): Path to save output images.
        comparison_folder (str): Path to save comparison images.
    """
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

            comparison_img = create_comparison_image(input_img, ref_img, output_img)
            comparison_path = os.path.join(comparison_folder, f"{os.path.splitext(input_file)[0]}_{os.path.splitext(reference_file)[0]}.png")
            cv2.imwrite(comparison_path, comparison_img)
            
            print(f"Saved output: {output_path}")
            print(f"Saved comparison: {comparison_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    process_images(
        os.path.join(base_dir, "input"),
        os.path.join(base_dir, "reference"),
        os.path.join(base_dir, "output"),
        os.path.join(base_dir, "comparison"),
    )
