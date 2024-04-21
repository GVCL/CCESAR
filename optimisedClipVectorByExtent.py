import rasterio
import geopandas as gpd
from shapely.geometry import box
from fiona.crs import from_epsg
from concurrent.futures import ProcessPoolExecutor

def process_image(i, input_data):
    sar_image_path = f'ur/{i}.tiff'
    output_file_path = f'ur/shpfiles1/{i}.shp'

    with rasterio.open(sar_image_path) as src:
        left, bottom, right, top = src.bounds

    clip_extent = gpd.GeoDataFrame(geometry=[box(left, bottom, right, top)], crs=from_epsg(4326))
    clipped_data = gpd.overlay(input_data, clip_extent, how='intersection')
    clipped_data.to_file(output_file_path, driver='ESRI Shapefile')
    print(f'Processed image {i}')

def main():
    input_file_path = 'land-polygons-split-4326/land-polygons-split-4326/land_polygons.shp'
    input_data = gpd.read_file(input_file_path)

    #with ProcessPoolExecutor() as executor:
    #    executor.map(lambda i: process_image(i, input_data), range(1, 168))
    for i in range(1,168):
        process_image(i, input_data)

if __name__ == "__main__":
    main()
