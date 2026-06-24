# src/data/raster_processor.py
import rioxarray

def align_raster(raster_path, template_raster_path):
    """
    Opens a driver raster, squeezes any singleton dimensions, 
    and ensures it perfectly matches the spatial grid shape and CRS of the template.
    """
    raster = rioxarray.open_rasterio(raster_path).squeeze()
    template = rioxarray.open_rasterio(template_raster_path).squeeze()
    
    # Structural safety check to protect array manipulation down the pipeline
    if raster.shape != template.shape or raster.rio.crs != template.rio.crs:
        raster = raster.rio.reproject_match(template)
        
    return raster