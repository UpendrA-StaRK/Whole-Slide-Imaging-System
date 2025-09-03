# PACKAGES
import os
import dask.array as da
import tifffile
import numpy as np
import json
from PIL import Image
import shutil
from zarr.errors import ArrayNotFoundError

class wsi_da():
    def _init_(self):
        self.num_processed_frames = 0
        self.processed_da = da.zeros((0, 0, 0), chunks=(0, 0, 0), dtype='uint8')

    def rechucking(self):
        # Rechunk the array to optimize storage structure before saving
        self.processed_da = self.processed_da.rechunk({0: -1, 1: 'auto'}, block_size_limit=1e8)

    def save_as_zarr(self, path):
        # Remove existing directory if it exists to avoid overwriting issues
        if os.path.isdir(path):
            shutil.rmtree(path)

        # Save the Dask array as a Zarr file
        self.processed_da.to_zarr(path, compressor=tifffile.Zlib(level=9))  # Lossless compression using zlib

    @staticmethod
    def zarr_to_ome_tiff(path, output_tiff_file, metadata_file, label_image_file):
        """
        Converts a Zarr folder to OME-TIFF format for large RGB images and includes metadata and a label image.
        
        Args:
            path (str): Path to the Zarr folder.
            output_tiff_file (str): Path to the output OME-TIFF file.
            metadata_file (str): Path to the JSON metadata file.
            label_image_file (str): Path to the label image file (JPEG/PNG).
        """
        # Ensure the path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"The Zarr folder does not exist at the given path: {path}")

        # Ensure the metadata JSON file exists
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata JSON file not found at the given path: {metadata_file}")

        # Ensure the label image file exists
        if not os.path.exists(label_image_file):
            raise FileNotFoundError(f"Label image file not found at the given path: {label_image_file}")

        # Load metadata from the JSON file
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        try:
            # Load the Dask array from Zarr
            image_data = da.from_zarr(path)
        except ArrayNotFoundError as e:
            raise ArrayNotFoundError(f"Array not found in the Zarr folder: {path}. Ensure it contains valid Zarr data.") from e
        except KeyError as e:
            raise KeyError(f"Zarr metadata is missing or incorrect in the folder: {path}. Error: {e}")

        # Ensure the data type is uint8
        if image_data.dtype != np.uint8:
            image_data = image_data.astype(np.uint8)

        # Compute the entire image to get a single NumPy array
        full_image = image_data.compute()

        # Load the label image (ensure it's loaded as a numpy array)
        label_img = Image.open(label_image_file)
        label_array = np.array(label_img)

        # Write the full image and the label image to OME-TIFF with lossless compression
        with tifffile.TiffWriter(output_tiff_file, ome=True, bigtiff=True) as tif:
            # Write the Zarr data as the main image
            tif.write(full_image, photometric='rgb', compression='deflate', 
                      metadata={'axes': 'YXC', 'Description': json.dumps(metadata)})
            
            # Write the label image as a second page in the TIFF file
            tif.write(label_array, metadata={'Description': 'Label Image'})

        print(f"OME-TIFF file with metadata and label image created successfully at: {output_tiff_file}")
