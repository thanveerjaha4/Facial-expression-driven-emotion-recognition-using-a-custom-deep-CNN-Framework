# Facial-expression-driven-emotion-recognition-using-a-custom-deep-CNN-Framework
import os

# Check if kaggle is installed
try:
    import kaggle
except ImportError:
    print("Kaggle API not installed. Install using:")
    print("pip install kaggle")
    exit()

# Command to download the dataset
print("Downloading FER2013 dataset from Kaggle...")
os.system("kaggle datasets download -d msambare/fer2013")

print("Download completed!")
print("Check the ZIP file in the current directory.")
