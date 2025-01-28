import os
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from constanst import *


def create_binary_image(sar_image_path, shapefile_path, output_image_path):
    with rasterio.open(sar_image_path) as src:
        sar_data = src.read(1)
        transform = src.transform
        profile = src.profile

    gdf = gpd.read_file(shapefile_path)
    mask = geometry_mask(
        gdf.geometry, out_shape=sar_data.shape, transform=transform, invert=True
    )
    binary_image = mask.astype("uint8") * 255
    profile.update(count=1, dtype="uint8")
    with rasterio.open(output_image_path, "w", **profile) as dst:
        dst.write(binary_image, 1)

if __name__ == "__main__":
    for i in range(len(os.listdir(DATASET_IMAGE_DIR))):
        sar_image_path = os.path.join(DATASET_IMAGE_DIR, f"{i+1}.tiff")
        shapefile_path = os.path.join(OUTPUT_INTERMEDIATE_DIR, rf"{i+1}.shp")
        output_image_path = os.path.join(OUTPUT_DIR, rf"{i+1}_mask.tiff")
        create_binary_image(sar_image_path, shapefile_path, output_image_path)
        print(f"processed image {i}")
