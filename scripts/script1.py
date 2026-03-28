import rasterio

path = r"C:\Users\24bcscs005\Downloads\PB_training_dataSet_shp_file\TIMMOWAL_37695_ORI.tif"

with rasterio.open(path) as src:
    print("Shape:", src.shape)
    print("CRS:", src.crs)
    print("Transform:", src.transform)