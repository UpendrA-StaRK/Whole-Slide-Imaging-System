#PACKAGES
import cv2
import numpy as np

import time
import json
import os

import shutil

#CLASSES
from collections import deque
from .wsi_dask import wsi_da
import dask.array as da


#Collect accquired frames from the queue and stitch and write into zarr file


class ProcessCollection():
    def _init_(self, queue):
        super()._init_()
        self.wsi_dask = wsi_da()
        self.queue = queue

        try:
            file_path = os.path.join(os.path.dirname(os.path.realpath(_file_)),
                                     '..', 'microscope_parameters', 'spinnaker_parameters.json')
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Set frame parameters from JSON
            self.frame_y = int(data['camera']['width'])
            self.frame_x = int(data['camera']['height'])
            self.frame_channels = 3 if data['camera']['pixel_format'] == 'RGB8Packed' else 1

            # Motion parameters
            self.total_cols = int(float(data['motion']['range']['x']) / float(data['motion']['increment']['x'])) + 1
            self.total_rows = int(float(data['motion']['range']['y']) / float(data['motion']['increment']['y'])) + 1

            self.step_x = int(float(data['motion']['increment']['x']) * (10**6) / 117)
            self.step_y = int(float(data['motion']['increment']['y']) * (10**6) / 117)
            print("print pixels x, y", self.step_x, self.step_y)
            print("print tiles, cols, rows", self.total_cols, self.total_rows)

            self.tile_size = (self.step_x, self.step_y, 3)

        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
        except FileNotFoundError:
            print("File not found:", file_path)
        except KeyError as e:
            print("KeyError: The key", e, "does not exist in the JSON data.")

        self.frame_size = (self.frame_x, self.frame_y, self.frame_channels)
        self.total_frames = self.total_cols * self.total_rows
        self.Min_Required_frames = self.total_cols + 1

    def remove_item(self):
        if self.queue:
            return self.queue.popleft()
        else:
            raise IndexError("The collection is empty.")

    def peek(self, index):
        if self.queue:
            return self.queue.get_item(index)
        else:
            raise IndexError("The collection is empty.")

    def trim(self, lefttrim, topttrim, width, height, image):
        """Trim the input image based on the specified bounds."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 2D RGB image")

        image_height, image_width, _ = image.shape
        righttrim = lefttrim + width
        bottomtrim = topttrim + height

        lefttrim = int(max(0, lefttrim))
        righttrim = int(min(image_width, righttrim))
        toptrim = int(max(0, topttrim))
        bottomtrim = int(min(image_height, bottomtrim))

        trimmed_image = image[toptrim:bottomtrim, lefttrim:righttrim, :]
        return trimmed_image

    def Find_Offsets(self, target_image, ref_image):
        """Find the offsets between the reference and target images using phase correlation."""
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)
        result = cv2.phaseCorrelate(np.float32(ref_gray), np.float32(target_gray))
        offset_x, offset_y = int(abs(result[0][1] / 2)), int(abs(result[0][0] / 2))
        return offset_x, offset_y

    def run(self):
        self.process_frames_thread()
        print('entered run')

    def process_frames_thread(self):
        """Process frames by registering them and handling rows."""
        num_processed_frames = int(self.wsi_dask.num_processed_frames)

        while num_processed_frames < self.total_frames:
            length = self.queue.size()
            if length >= self.total_cols * 2:
                self.register_raster_group()
                for _ in range(self.total_cols):
                    self.remove_item()
            else:
                time.sleep(0.5)
            num_processed_frames = int(self.wsi_dask.num_processed_frames)

        print('out of loop process frames thread')
        return True

    def process_first_col(self):
        """Process the first column of frames."""
        row_count = int(self.wsi_dask.num_processed_frames) // self.total_cols
        if row_count == 0:
            offset_x, offset_y, start = 0, 0, 0
        else:
            offset_x, offset_y, start = 0, 0, self.total_cols

        trimmed_image = self.trim(0, 0, self.tile_size[0], self.tile_size[1], self.peek(start))
        da_row = da.from_array(trimmed_image, chunks=self.tile_size)
        return offset_x, offset_y, da_row

    def process_row(self):
        """Process a full row of frames and stitch them together."""
        row_count = int(self.wsi_dask.num_processed_frames) // self.total_cols

        group_horizantal_shift, group_vertical_shift, da_row = self.process_first_col()
        start = 0 if row_count == 0 else self.total_cols

        for i in range(start + 1, start + self.total_cols):
            off_x, off_y = 0, 0
            offset_x = int(off_x) + group_horizantal_shift
            offset_y = int(off_y) + group_vertical_shift
            trimmed_image = self.trim(0, 0, self.tile_size[0], self.tile_size[1], self.peek(i))
            da_row = da.hstack([da_row, da.from_array(trimmed_image, chunks=self.tile_size)])

        self.wsi_dask.num_processed_frames += self.total_cols
        if int(self.wsi_dask.num_processed_frames) <= self.total_cols:
            self.wsi_dask.processed_da = da_row
        else:
            self.wsi_dask.processed_da = da.vstack((self.wsi_dask.processed_da, da_row))

        print('row processed')

    def register_raster_group(self):
        """Perform raster group registration."""
        if int(self.wsi_dask.num_processed_frames) == 0:
            self.process_row()
        self.process_row()
        return True

    def save_processed_images(self, zarr_path, ome_tiff_path):
        """Save processed images to Zarr and OME-TIFF formats."""
        if os.path.isdir(zarr_path):
            shutil.rmtree(zarr_path)

        self.wsi_dask.rechucking()
        self.wsi_dask.processed_da.to_zarr(zarr_path)
        self.wsi_dask.zarr_to_ome_tiff(zarr_path, ome_tiff_path)


# Static method for multiprocessing
@staticmethod
def zarr_process(queue, zarr_path, ome_tiff_path):
    process_collector = ProcessCollection(queue)
    process_collector.run()
    process_collector.save_processed_images(zarr_path, ome_tiff_path)
