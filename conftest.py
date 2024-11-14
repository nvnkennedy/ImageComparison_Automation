import pytest
import os
import base64
import logging
from io import StringIO
import re

# Convert an image to a base64-encoded string (for embedding in self-contained reports)
def image_to_base64(image_path):
    """
    Converts an image file to a base64-encoded string for embedding into HTML reports.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path} to base64: {e}")
        raise

# Configure logging to use color in terminal output
class ColorFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log messages based on their severity.
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

# Add a colored logging handler for pytest output
def setup_logging():
    """
    Sets up logging with colored output and captures logs in memory.
    """
    try:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Set the base logging level to INFO
        log_stream = StringIO()
        console_handler = logging.StreamHandler(log_stream)
        console_handler.setFormatter(ColorFormatter())
        logger.addHandler(console_handler)
        return log_stream  # Return the log stream so we can access the logs later
    except Exception as e:
        logger.error(f"Error setting up logging: {e}")
        raise

# Function to remove color codes from the log output
def remove_color_codes(text):
    """
    Removes color escape sequences from log text.
    """
    try:
        color_code_pattern = re.compile(r'\033\[[0-9;]*m')
        return re.sub(color_code_pattern, '', text)
    except Exception as e:
        logger.error(f"Error removing color codes: {e}")
        raise

# Hook to add extra data (image) to the HTML report
@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    """
    Hook to add additional information (such as images and logs) to the HTML report
    after each test runs.
    """
    outcome = yield
    report = outcome.get_result()

    # Set up logging for colored output and capturing logs
    log_stream = setup_logging()

    try:
        # Only add extra content for the 'call' phase (after test execution)
        if report.when == 'call':
            pytest_html = item.config.pluginmanager.getplugin('html')
            extra = getattr(report, 'extra', [])

            # Get the request context and pass the image paths to the test
            feature_request = item.funcargs.get('request')  # This will fetch 'request' fixture
            if feature_request:
                base_dir = os.path.dirname(__file__)
                image_dir = os.path.join(base_dir, "images")

                # Fetch image paths from test function arguments
                image1_path = feature_request.node.funcargs.get('image1_path', os.path.join(image_dir, 'IMAGE_1.png'))
                ref_image_path = feature_request.node.funcargs.get('ref_image_path', os.path.join(image_dir, 'REF_Image.png'))
                export_image_path = feature_request.node.funcargs.get('export_image_path', os.path.join(image_dir, 'EXPORT_Image.jpg'))
                diff_path = feature_request.node.funcargs.get('output_diff_path', None)

                def add_image_to_report(image_path, label):
                    """
                    Adds an image to the HTML report using base64 encoding.
                    """
                    try:
                        if os.path.exists(image_path):
                            base64_image = image_to_base64(image_path)
                            extra.append(pytest_html.extras.image(f"data:image/png;base64,{base64_image}", label))
                        else:
                            logger.error(f"Image file not found: {image_path}")
                    except Exception as e:
                        logger.error(f"Error adding image {image_path} to report: {e}")
                        raise

                # Add the correct images to the report based on the test function
                if 'test_image1_vs_ref_image' in item.name:
                    add_image_to_report(image1_path, 'Image 1')
                    add_image_to_report(ref_image_path, 'Reference Image')
                elif 'test_image2_vs_export_image' in item.name:
                    add_image_to_report(image1_path, 'Image 2')
                    add_image_to_report(export_image_path, 'Export Image')

                # If diff image exists, add it to the report
                if diff_path and os.path.exists(diff_path):
                    add_image_to_report(diff_path, 'Difference Image')

                # Capture only the last 10 lines of logs if the test fails
                if report.outcome == 'failed':
                    log_content = log_stream.getvalue().splitlines()[-10:]  # Get the last 10 log lines
                    # Remove color codes from the log before adding it to the report
                    log_capture_clean = remove_color_codes("\n".join(log_content))
                    extra.append(pytest_html.extras.text(f"Log Output:\n{log_capture_clean}"))

            # Attach the extra information (images and logs) to the report
            report.extra = extra

    except Exception as e:
        logger.error(f"Error in pytest_runtest_makereport hook: {e}")
        raise
