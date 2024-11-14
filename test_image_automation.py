import os
import pytest
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import logging

# Set up logging with colors to improve the visibility of log messages
class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter that adds color to log messages based on the severity level.
    """
    COLOR_CODES = {
        logging.DEBUG: "\033[37m",   # white
        logging.INFO: "\033[32m",    # green
        logging.WARNING: "\033[33m", # yellow
        logging.ERROR: "\033[31m",   # red
        logging.CRITICAL: "\033[41m" # white on red background
    }

    def format(self, record):
        color_code = self.COLOR_CODES.get(record.levelno, "\033[0m")  # Default to no color
        reset_code = "\033[0m"
        message = super().format(record)
        return f"{color_code}{message}{reset_code}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter())
logger.addHandler(console_handler)

# Helper functions

def ensure_dir(directory):
    """
    Ensures that a directory exists. If not, it creates the directory.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    except OSError as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        raise

def load_image(image_path):
    """
    Loads an image from the specified path and returns it.
    Raises FileNotFoundError if the image cannot be loaded.
    """
    try:
        logger.info(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        logger.info(f"Image loaded successfully: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise

def save_image(image, save_path):
    """
    Saves an image to the specified path. If the directory doesn't exist, it will be created.
    Raises IOError if the save operation fails.
    """
    try:
        ensure_dir(os.path.dirname(save_path))  # Ensure the target directory exists
        logger.info(f"Saving image to {save_path}")
        result = cv2.imwrite(save_path, image)
        if not result:
            raise IOError(f"Failed to save image: {save_path}")
        logger.info(f"Image saved successfully to {save_path}")
    except Exception as e:
        logger.error(f"Error saving image {save_path}: {e}")
        raise

def compare_images(image1_path, image2_path, output_diff_path):
    """
    Compares two images using SSIM (Structural Similarity Index) and saves the difference image.
    Returns a tuple (identical, diff_path), where identical is a boolean indicating if the images are the same.
    """
    try:
        logger.info(f"Comparing images: {image1_path} and {image2_path}")
        image1 = load_image(image1_path)
        image2 = load_image(image2_path)

        if image1.shape != image2.shape:
            raise ValueError(f"Images have different dimensions: {image1.shape} vs {image2.shape}")

        min_dim = min(image1.shape[0], image1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 != 0 else min_dim - 1)  # Ensure odd window size

        # Set the correct axis for SSIM calculation based on whether the image is grayscale or RGB
        channel_axis = -1 if image1.ndim == 3 else None

        # Compute SSIM and obtain the difference image
        ssim_index, diff = ssim(image1, image2, full=True, multichannel=True, win_size=win_size, channel_axis=channel_axis)

        # If the images are identical, no need to compute and save the difference
        if np.array_equal(image1, image2):
            logger.info("Images are identical.")
            return True, None

        # Convert the difference image to an 8-bit format for visualization
        diff = (diff * 255).astype("uint8")
        save_image(diff, output_diff_path)
        logger.info(f"Difference image saved to {output_diff_path}")
        return False, output_diff_path
    except Exception as e:
        logger.error(f"Error comparing images {image1_path} and {image2_path}: {e}")
        raise

def plot_image_comparison(image1_path, image2_path, diff_path, output_plot_path):
    """
    Plots the comparison of two images and their difference. Saves the resulting plot as a PNG image.
    """
    try:
        logger.info(f"Plotting image comparison: {image1_path}, {image2_path}, {diff_path}")
        image1 = cv2.cvtColor(load_image(image1_path), cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(load_image(image2_path), cv2.COLOR_BGR2RGB)
        diff = cv2.cvtColor(load_image(diff_path), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        axes[0].imshow(image1)
        axes[0].set_title('Image 1')
        axes[0].axis('off')

        axes[1].imshow(image2)
        axes[1].set_title('Image 2')
        axes[1].axis('off')

        axes[2].imshow(diff)
        axes[2].set_title('Difference')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(output_plot_path)
        plt.close()
        logger.info(f"Comparison plot saved to {output_plot_path}")
    except Exception as e:
        logger.error(f"Error plotting image comparison: {e}")
        raise

# Define test class
@pytest.mark.usefixtures("record_tests")
class TestImageEditor:

    @pytest.fixture
    def record_tests(self, request):
        pass

    def test_image1_vs_ref_image(self, request):
        """
        Test that compares IMAGE_1 with the reference image (REF_Image).
        Verifies that the images are identical and saves the difference if any.
        """
        try:
            base_dir = os.path.dirname(__file__)
            image_dir = os.path.join(base_dir, "images")
            report_dir = os.path.join(base_dir, "reports")

            image1_path = os.path.join(image_dir, 'IMAGE_1.png')
            ref_image_path = os.path.join(image_dir, 'REF_Image.png')
            output_diff_path = os.path.join(report_dir, 'DIFF_Image_1_REF.png')

            logger.info("Starting test: test_image1_vs_ref_image")
            image1 = load_image(image1_path)
            save_image(image1, ref_image_path)

            # Check if the reference image was saved successfully
            if not os.path.exists(ref_image_path):
                raise IOError(f"Failed to save reference image: {ref_image_path}")

            identical, diff_path = compare_images(image1_path, ref_image_path, output_diff_path)

            request.node.funcargs['image1_path'] = image1_path
            request.node.funcargs['ref_image_path'] = ref_image_path
            request.node.funcargs['output_diff_path'] = diff_path

            assert identical, "IMAGE_1 and REF_Image are not identical"

        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise

    def test_image2_vs_export_image(self, request):
        """
        Test that compares IMAGE_2 with the export image (EXPORT_Image).
        Verifies that the images are identical and saves the difference if any.
        """
        try:
            base_dir = os.path.dirname(__file__)
            image_dir = os.path.join(base_dir, "images")
            report_dir = os.path.join(base_dir, "reports")

            image2_path = os.path.join(image_dir, 'IMAGE_2.png')
            export_image_path = os.path.join(image_dir, 'EXPORT_Image.jpg')
            output_diff_path = os.path.join(report_dir, 'DIFF_Image_2_EXPORT.png')

            logger.info("Starting test: test_image2_vs_export_image")
            image2 = load_image(image2_path)
            save_image(image2, export_image_path)

            # Check if the export image was saved successfully
            if not os.path.exists(export_image_path):
                raise IOError(f"Failed to save export image: {export_image_path}")

            identical, diff_path = compare_images(image2_path, export_image_path, output_diff_path)

            request.node.funcargs['image2_path'] = image2_path
            request.node.funcargs['export_image_path'] = export_image_path
            request.node.funcargs['output_diff_path'] = diff_path

            assert identical, "IMAGE_2 and EXPORT_Image are not identical"

        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    report_dir = os.path.join(base_dir, "reports")
    pytest.main(["-v", f"--html={os.path.join(report_dir, 'report.html')}", "--self-contained-html", "--capture=sys"])
