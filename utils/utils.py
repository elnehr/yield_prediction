import pandas as pd
import xarray as xr
import regionmask
from typing import Optional
import geopandas as gpd
import rasterio.features
import numpy as np
import rasterio as rio

def convert_raster_to_vectors(raster_data: rio.DatasetReader, area_mask: Optional[np.ndarray] = None) -> gpd.GeoDataFrame:
    """
    Convert raster data from a GeoTiff file into vector shapes in a GeoDataFrame,
    excluding areas marked as nodata.

    Parameters:
        raster_data: A raster data source to convert.
        area_mask: An optional mask to define nodata areas.

    Returns:
        GeoDataFrame with geometry and attributes derived from raster data.
    """
    data_array = raster_data.read(1)
    affine_transform = raster_data.transform
    coordinate_system = raster_data.crs

    # Assemble the geometry and value into a GeoDataFrame
    vector_features = (
        {"properties": {"value": value}, "geometry": geometry}
        for geometry, value in rasterio.features.shapes(data_array, mask=area_mask, transform=affine_transform)
    )

    vector_list = list(vector_features)
    vector_gdf = gpd.GeoDataFrame.from_features(vector_list, crs=coordinate_system)

    # Exclude no data value areas
    no_value = raster_data.nodata
    if no_value is not None:
        vector_gdf = vector_gdf[vector_gdf.value != no_value]

    # Log a warning if there are invalid geometries
    invalid_geometries = len(vector_gdf) - vector_gdf.geometry.is_valid.sum()
    if invalid_geometries > 0:
        print(f"Warning: {invalid_geometries} invalid geometries found in raster data.")
    return vector_gdf

