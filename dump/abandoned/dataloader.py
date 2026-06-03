import openslide
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from openslide.deepzoom import DeepZoomGenerator
from tqdm import tqdm

def letterbox(img, new_shape=(1024, 1024), auto=False, scaleup=True):
    """Resize and pad image to a fixed size of (1024, 1024)"""
    shape = img.shape[:2]  # Current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2

    # Resize image
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    # Ensure padding is exact to avoid shape mismatches
    top, bottom = dh, new_shape[0] - (new_unpad[1] + dh)
    left, right = dw, new_shape[1] - (new_unpad[0] + dw)
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    
    return img


class WSITileDataset(Dataset):
    def __init__(self, path, img_size, tile_size, overlap, batch_size=16):
        """
        Custom dataset to load and process WSI tiles.
        Args:
            path (str): Path to the .ndpi WSI file.
            img_size (int): Final image size after letterbox processing.
            tile_size (int): Size of each tile.
            overlap (int): Overlap between tiles.
            batch_size (int): Number of tiles per batch.
        """
        self.img_size = img_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.path = path

        # Load slide and DeepZoomGenerator
        self.slide = openslide.open_slide(path)
        self.tile_source = DeepZoomGenerator(self.slide, tile_size=tile_size, overlap=overlap, limit_bounds=False)
        self.level = self.tile_source.level_count - 1  # Highest level
        self.cols, self.rows = self.tile_source.level_tiles[self.level]

        # Generate (col, row) tile indices
        self.img_files = [(col, row) for row in range(self.rows) for col in range(self.cols)]
        self.n = len(self.img_files)

        print(f"Loaded {self.n} tiles from {path}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        Load a single tile from the WSI.
        Args:
            index (int): Tile index.

        Returns:
            torch.Tensor: Processed tile image.
            tuple: (original height, original width).
            tuple: (tile X coordinate, tile Y coordinate).
        """
        tile_id = self.img_files[index]

        # Extract tile
        img = np.array(self.tile_source.get_tile(self.level, tile_id))
        h0, w0 = img.shape[:2]

        # Apply letterbox resizing
        img = letterbox(img, self.img_size)

        # Convert to tensor
        img = img.transpose(2, 0, 1)  # Convert to (C, H, W)
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), (h0, w0), self.tile_source.get_tile_coordinates(self.level, tile_id)

    @staticmethod
    def collate_fn(batch):
        img, shapes, coords = zip(*batch)
        return torch.stack(img, 0), shapes, coords

def WSI_dataloader(path, img_size, tile_size, overlap, batch_size):
    """
    Create a PyTorch dataloader for WSI processing.
    """
    dataset = WSITileDataset(path, img_size, tile_size, overlap, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=WSITileDataset.collate_fn)
    return dataloader

# Example Usage:
if __name__ == "__main__":
    wsi_path = r"D:\Shreyan\Development\Test\check\C25 - 1307 2 - 3028850 - 75Y.ndpi"
    batch_size = 16
    img_size = 1024
    tile_size = 1024
    overlap = 128

    dataloader = WSI_dataloader(wsi_path, img_size, tile_size, overlap, batch_size)

    for batch_i, (img, shapes, coords) in enumerate(tqdm(dataloader)):
        print(f"Batch {batch_i}: {img.shape}, {len(shapes)}, {len(coords)}")
