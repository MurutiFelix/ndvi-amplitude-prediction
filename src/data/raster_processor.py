# src/data/raster_processor.py
import geopandas as gpd
import rioxarray
import xarray as xr
from rasterio.features import rasterize

def process_soil_from_zip(zip_path, template_raster_path):
    """Reads zipped DSMW shapefile, reprojects to MODIS Sinusoidal, and rasterizes."""
    # Open template to copy its spatial grid matrix
    template = rioxarray.open_rasterio(template_raster_path).squeeze()
    
    # Read zipped shapefile directly
    soil_vector = gpd.read_file(f"zip://{zip_path}")
    
    # Reproject vector to match MODIS Sinusoidal
    soil_vector_proj = soil_vector.to_crs(template.rio.crs)
    
    # Clip vector to template bounding box to save processing memory
    xmin, ymin, xmax, ymax = template.rio.bounds()
    clipped_vector = soil_vector_proj.cx[xmin:xmax, ymin:ymax]
    
    # Burn 'SNUM' attributes into the 159x181 grid array
    shapes = ((geom, val) for geom, val in zip(clipped_vector.geometry, clipped_vector['SNUM']))
    soil_array = rasterize(
        shapes=shapes,
        out_shape=(template.rio.height, template.rio.width),
        transform=template.rio.transform(),
        fill=0,
        dtype='int32'
    )
    
    # Wrap back into an xarray DataArray with matching coordinates
    soil_da = xr.DataArray(soil_array, coords=template.coords, dims=template.dims)
    return soil_da

def process_precipitation_nc(nc_path, template_raster_path):
    """Opens NetCDF precipitation data, reprojects and resamples it to 1km."""
    template = rioxarray.open_rasterio(template_raster_path).squeeze()
    
    # Open NetCDF dataset
    nc_data = xr.open_dataset(nc_path)
    
    # Extract precipitation variable (adjust key name e.g., 'tp' or 'precip' based on your file)
    precip_var = nc_data['tp'] 
    
    # Write initial CRS if it's not detected (usually EPSG:4326 for ERA5)
    if precip_var.rio.crs is None:
        precip_var.rio.write_crs("EPSG:4326", inplace=True)
        
    # Reproject and match the resolution/bounds of the MODIS template via bilinear interpolation
    precip_resampled = precip_var.rio.reproject_match(template)
    
    return precip_resampled