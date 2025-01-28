import os
import rasterio
import geopandas as gpd
from shapely.geometry import box
from fiona.crs import from_epsg
from constanst import *


def process_image(i, input_data):
    sar_image_path = os.path.join(DATASET_IMAGE_DIR, f"{i}.tiff")
    output_file_path = OUTPUT_INTERMEDIATE_DIR
    with rasterio.open(sar_image_path) as src:
        left, bottom, right, top = src.bounds
    clip_extent = gpd.GeoDataFrame(
        geometry=[box(left, bottom, right, top)], crs=from_epsg(4326)
    )
    clipped_data = gpd.overlay(input_data, clip_extent, how="intersection")
    clipped_data.to_file(output_file_path, driver="ESRI Shapefile")
    print(f"Processed image {i}")


def main():
    input_file_path = LAND_POLYGONS_SHP
    input_data = gpd.read_file(input_file_path)
    for i in range(len(os.listdir(DATASET_IMAGE_DIR))):
        process_image(i + 1, input_data)


if __name__ == "__main__":
    main()
