import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io
import os

def calculate_psnr_ssim(image1_path, image2_path):
    # Load the two images
    image1 = io.imread(image1_path, as_gray=True)
    image2 = io.imread(image2_path, as_gray=True)

    # Calculate PSNR
    psnr_value = psnr(image1, image2)

    # Calculate SSIM
    ssim_value = ssim(image1, image2,  data_range=image1.max() - image1.min())

    return psnr_value, ssim_value

def calculate_average_psnr_ssim(num_images, folder_path):
    total_psnr, total_ssim = 0, 0
    for i in range(1, num_images + 1):
        image1_path = f'{folder_path}/test_batch_{i}.jpg'
        image2_path = f'{folder_path}/test_batch_{i}_truth.jpg'
        psnr_value, ssim_value = calculate_psnr_ssim(image1_path, image2_path)
        total_psnr += psnr_value
        total_ssim += ssim_value

    average_psnr = total_psnr / num_images
    average_ssim = total_ssim / num_images
    return average_psnr, average_ssim

def rename_files(folder_path, prefix):
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix):
            new_name = filename[len(prefix):]  # Remove the prefix
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
            
if __name__ == "__main__":
    folder_path = './Test_image/LOLSyndeeperResdenoisereducelayerscheduleradd485reducedecomSE'
    num_images = 100
    average_psnr, average_ssim = calculate_average_psnr_ssim(num_images, folder_path)
    print("Average PSNR:", average_psnr)
    print("Average SSIM:", average_ssim)
    # low_folder = "Train_data/VE-LOL-L-Cap-Full/test/low"
    # high_folder = "Train_data/VE-LOL-L-Cap-Full/test/high"

    # rename_files(low_folder, "low")
    # rename_files(high_folder, "normal")
        
