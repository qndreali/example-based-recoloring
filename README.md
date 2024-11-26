<a name="readme-top"></a>
<h1 align="center">Example-Based Recoloring</h3>
<h4 align="center"> [DIGIMAP] Digital Image Processing — Final Project Practical Manifestation</h4>
<div align="center"> <img src = "/comparison/yellow-sunset_purple-field.png"  width = "1000"></div>

## About The Project

A Simple Python Example-Based Recoloring Program that perform recoloring of images based on color distribution matching. It uses Probability Density Function (PDF) transfer techniques to transfer the color characteristics of a reference image to an input image. The program generates both recolored images and comparison visualizations.

By leveraging matrix rotations and statistical color distribution matching, this program is particularly suited for tasks requiring accurate color harmonization, such as artistic image editing, data augmentation for computer vision tasks, or aesthetic enhancement of photographs.

</div>

## Key Features
- **PDF Transfer**: Uses probability density function matching to adjust the color distribution of the input image based on a reference.
- **Rotations for Precision**: Applies a series of precomputed or randomly generated rotation matrices to optimize the color transformation process.
- **Batch Processing**: Efficiently handles multiple input and reference images, automatically saving results.
- **Comparison Visualizations**: Creates side-by-side output for easy comparison of input, reference, and recolored images.
- **Customizable Workflow**: Supports flexible configurations, including custom numbers of color channels or rotation strategies.
- **Error Handling**: Automatically skips processing of files with identical names or unsupported formats.

## Installation
1. **Clone the Repository**:
   Begin by cloning this repository to your local system:
   ```bash
   git clone https://github.com/qndreali/example-based-recoloring.git
   cd example-based-recoloring
   ```
2. **Install Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install numpy opencv-python scipy
   ```

## Directory Structure
The program is organized to streamline the image recoloring process. Here is the following directory layout:
```graphql
project/
├── input/          
├── reference/      
├── output/         
├── comparison/     
└── main.py         
```
- Input Folder: Contains the images that need recoloring.
- Reference Folder: Contains the reference images with the desired color characteristics.
- Output Folder: Recolored images will be saved here.
- Comparison Folder: Side-by-side comparison images will be saved here for visual inspection.


## Example Outputs
<img src = "/comparison/main-input_main-reference.png" width = "700">
<img src = "/comparison/pink-sunset_green-land.png" width = "700">
<img src = "/comparison/green-forest_day-mountains.png" width = "700">

## Authors
[@qndreali](https://github.com/qndreali)
 
<p align="right">(<a href="#readme-top">back to top</a>)</p>
