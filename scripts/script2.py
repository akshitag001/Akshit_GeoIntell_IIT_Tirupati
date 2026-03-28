import geopandas as gpd

gdf = gpd.read_file(r"C:\Users\24bcscs005\Downloads\PB_training_dataSet_shp_file\shp-file\Built_Up_Area_typ.shp")
print(gdf.head())
print(gdf.crs)