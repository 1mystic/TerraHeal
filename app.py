
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from shapely import wkt
from datetime import datetime, timedelta

import ee
import geemap

app = Flask(__name__, template_folder='templates', static_folder='static')

#GEE Initialization
try:
    if not ee.data._credentials: # auth check
        ee.Authenticate()
    # GEE Project ID 
    ee.Initialize(project='terraheal-461612', opt_url='https://earthengine-highvolume.googleapis.com')
    print("Google Earth Engine initialized successfully with project terraheal-461612.")
except Exception as e:
    print(f"Error initializing Google Earth Engine: {e}")
    print("Please ensure you have authenticated and set up a GEE project.")
    

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CSV_FILE_PATH = os.path.join(DATA_DIR, 'fires_dataframe.csv')


FIRE_MAPPING = {
    "California Camp Fire (2018)": {"Incid_Name": "CAMP FIRE"},
    "Australia Black Summer (2019-20)": {"Incid_Name": "AU_BLACK_SUMMER_EXAMPLE"},
    "Amazon Basin (2020)": {"Incid_Name": "AMAZON_BASIN_EXAMPLE"},       
    "Oregon Cascades (2023)": {"Incid_Name": "OR_CASCADES_EXAMPLE"}     
}

#GEE Helper Functions 
def mask_clouds_s2(img):
    """Masks clouds in a Sentinel-2 SR image using QA60 band."""
    qa = img.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_mask = (1 << 10) | (1 << 11)
    mask = qa.bitwiseAnd(cloud_mask).eq(0)
   
    return img.updateMask(mask).copyProperties(img, ["system:time_start"])

def add_ndvi_s2(img):
    """Computes NDVI for a Sentinel-2 image and adds it as a band."""
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return img.addBands(ndvi).copyProperties(img, ["system:time_start"])

def get_fire_data(fire_name_from_dropdown):
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"Data file not found: {CSV_FILE_PATH}")
    
    df = pd.read_csv(CSV_FILE_PATH)
    
    fire_params = FIRE_MAPPING.get(fire_name_from_dropdown)
    if not fire_params:
        raise ValueError(f"Configuration for '{fire_name_from_dropdown}' not found in FIRE_MAPPING.")
        
    incid_name_csv = fire_params["Incid_Name"]
    
    fire_row = df[df['Incid_Name'] == incid_name_csv]
    if fire_row.empty:
        raise ValueError(f"Fire '{incid_name_csv}' (mapped from '{fire_name_from_dropdown}') not found in {os.path.basename(CSV_FILE_PATH)}.")
    
    geometry_wkt = fire_row.iloc[0]['geometry']
    ignition_date_str = fire_row.iloc[0]['Ig_Date'] # 'YYYY-MM-DD'

    # WKT to ee.Geometry
    shapely_geom = wkt.loads(geometry_wkt)
    geojson_geom = shapely_geom.__geo_interface__
    aoi = ee.Geometry(geojson_geom).simplify(maxError=100) # for GEE performance

    return aoi, ignition_date_str

def generate_gee_analysis(aoi, ignition_date_str, fire_display_name):
    """
    Generates GEE map and statistics for the given fire AOI and ignition date.
    Returns a tuple: (map_html, stats_dict)
    """
    try:
        ignition_date = datetime.strptime(ignition_date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid Ig_Date format: {ignition_date_str}. Expected YYYY-MM-DD.")

    # Date Ranges
    pre_fire_end_date = ignition_date - timedelta(days=30) # End 1 month before fire for cleaner baseline
    pre_fire_start_date = pre_fire_end_date - timedelta(days=365) # 1 year period

    post_fire_intervals = [
      
        (ignition_date + timedelta(days=11*30),ignition_date + timedelta(days=14*30), "ndvi_t11_14m"), 
        (ignition_date + timedelta(days=23*30),ignition_date + timedelta(days=26*30), "ndvi_t23_26m"), 
    ]

    s2_collection_id = "COPERNICUS/S2_SR_HARMONIZED"

    # Baseline NDVI
    pre_fire_s2 = (
        ee.ImageCollection(s2_collection_id)
        .filterBounds(aoi)
        .filterDate(pre_fire_start_date.strftime('%Y-%m-%d'), pre_fire_end_date.strftime('%Y-%m-%d'))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) # Cloud tolerance
        .map(mask_clouds_s2)
        .map(add_ndvi_s2)
        .select("NDVI")
    )
    baseline_ndvi_img = pre_fire_s2.median().rename('baseline_ndvi').clip(aoi)
    print(f"Pre-fire images found for {fire_display_name}: {pre_fire_s2.size().getInfo()}")
    if pre_fire_s2.size().getInfo() == 0:
        print(f"Warning: No pre-fire images for {fire_display_name}. Baseline NDVI might be inaccurate.")
        baseline_ndvi_img = ee.Image(0).rename('baseline_ndvi').clip(aoi)


    # Post-fire NDVI Stack 
    ndvi_images_list = [baseline_ndvi_img] # Start with baseline
    for start_dt, end_dt, name in post_fire_intervals:
        s2_img_col = (
            ee.ImageCollection(s2_collection_id)
            .filterBounds(aoi)
            .filterDate(start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) # Slightly higher tolerance post-fire
            .map(mask_clouds_s2)
            .map(add_ndvi_s2)
            .select("NDVI")
        )
        img = s2_img_col.median().unmask(0).rename(name).clip(aoi) 
        ndvi_images_list.append(img)
        print(f"Images for {name} ({fire_display_name}): {s2_img_col.size().getInfo()}")


    ndvi_stack = ee.Image.cat(ndvi_images_list)
    band_names = ndvi_stack.bandNames().getInfo()

    # Heuristic Cold Spot Labeling 
    target_t12_band = "ndvi_t11_14m"
    cold_spots_heuristic_img = ee.Image().rename('cold_spots_heuristic').clip(aoi) # Default empty

    if target_t12_band in band_names and 'baseline_ndvi' in band_names:
        ndvi_t12_img = ndvi_stack.select(target_t12_band)
        recovery_threshold_val = baseline_ndvi_img.multiply(0.6) # Recovered to <60% of baseline
        is_vegetated_pre_fire = baseline_ndvi_img.gt(0.2)

        cold_spots_heuristic_img = ndvi_t12_img.lt(recovery_threshold_val) \
                                    .And(is_vegetated_pre_fire) \
                                    .rename('cold_spots_heuristic').selfMask()
    else:
        print(f"Warning: Missing bands for heuristic cold spot calculation for {fire_display_name}.")


    # --- Persistent Cold Spots ---
    target_t24_band = "ndvi_t23_26m" 
    persistent_cold_spots_img = ee.Image().rename('persistent_cold_spots').clip(aoi) # Default empty

    heuristic_band_present = 'cold_spots_heuristic' in cold_spots_heuristic_img.bandNames().getInfo()

    if target_t24_band in band_names and 'baseline_ndvi' in band_names and heuristic_band_present:
        ndvi_t24_img = ndvi_stack.select(target_t24_band)
        recovery_threshold_val = baseline_ndvi_img.multiply(0.6) # Same threshold
        
        persistent_cold_spots_img = ndvi_t24_img.lt(recovery_threshold_val) \
                                    .And(cold_spots_heuristic_img.eq(1)) \
                                    .rename('persistent_cold_spots').selfMask()
    else:
        print(f"Warning: Missing bands/data for persistent cold spot calculation for {fire_display_name}.")

    # --- Calculate Stats ---
    stats = {'name': fire_display_name, 'ignition_date': ignition_date_str}
    pixel_area_ha = ee.Image.pixelArea().divide(10000) # Area in hectares per pixel

    try:
        stats['total_area_burned_ha'] = round(aoi.area(maxError=100).getInfo() / 10000, 2)
    except Exception: stats['total_area_burned_ha'] = 'N/A'

    try:
        cs_area_result = cold_spots_heuristic_img.multiply(pixel_area_ha).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e10, bestEffort=True
        ).get('cold_spots_heuristic')
        stats['cold_spot_area_ha'] = round(cs_area_result.getInfo(), 2) if cs_area_result.getInfo() is not None else 0.0
    except Exception: stats['cold_spot_area_ha'] = 'N/A'

    try:
        pcs_area_result = persistent_cold_spots_img.multiply(pixel_area_ha).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e10, bestEffort=True
        ).get('persistent_cold_spots')
        stats['persistent_cold_spot_area_ha'] = round(pcs_area_result.getInfo(), 2) if pcs_area_result.getInfo() is not None else 0.0
    except Exception: stats['persistent_cold_spot_area_ha'] = 'N/A'

    # --- Create Map ---
    Map = geemap.Map(plugin_Draw=False, ZoomControl=True, ScaleControl=True, layer_control=True)
    Map.centerObject(aoi, 10)
    Map.addLayer(aoi, {"color": "grey", "fillColor": "grey", "opacity": 0.5}, "Fire Boundary (AOI)", True)

    if 'baseline_ndvi' in band_names:
      Map.addLayer(ndvi_stack.select('baseline_ndvi'), {"min": 0, "max": 1, "palette": ['brown', 'yellow', 'green']}, "Baseline NDVI (Pre-Fire)", False)
    if target_t12_band in band_names:
      Map.addLayer(ndvi_stack.select(target_t12_band), {"min": 0, "max": 1, "palette": ['red', 'yellow', 'green']}, "NDVI ~1 Year Post-Fire", False)
    if target_t24_band in band_names:
      Map.addLayer(ndvi_stack.select(target_t24_band), {"min": 0, "max": 1, "palette": ['red', 'yellow', 'green']}, "NDVI ~2 Years Post-Fire", False)

    Map.addLayer(cold_spots_heuristic_img, {"palette": "FF0000"}, "Recovery Cold Spots (~1yr)", True) # Red
    Map.addLayer(persistent_cold_spots_img, {"palette": "800080"}, "Persistent Cold Spots (~2yr)", False) # Purple

    html_output = Map.to_html(filename=None, add_layer_control=False) 
    import re
    html_output = re.sub(r'width:\s*100\.0%;\s*height:\s*100\.0%;', 'width:100.0%;height:420px;', html_output)


    return html_output, stats

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('datavista_page.html')

@app.route('/generate_map_and_stats', methods=['POST']) # Renamed route for clarity
def generate_map_and_stats_route():
    try:
        data = request.get_json()
        fire_name_from_dropdown = data.get('fire_name') # This is the key from FIRE_MAPPING
        if not fire_name_from_dropdown:
            return jsonify({"error": "Missing fire_name parameter."}), 400

        aoi, ignition_date_str = get_fire_data(fire_name_from_dropdown)
        map_html, stats_dict = generate_gee_analysis(aoi, ignition_date_str, fire_name_from_dropdown)
        
        return jsonify({"map_html": map_html, "stats": stats_dict})

    except FileNotFoundError as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500
    except ValueError as e: # Handle issues from get_fire_data or date parsing
        print(f"User/Data Error: {e}")
        return jsonify({"error": str(e)}), 400
    except ee.EEException as e: # GEE specific errors
        print(f"GEE Error: {e}")
        return jsonify({"error": f"Google Earth Engine error: {e}. This might be due to request limits, unavailable data, or GEE server issues. Check server logs."}), 500
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"An unexpected error occurred: {e}\nDetails: {error_details}")
        return jsonify({"error": f"An unexpected server error occurred. Check logs for details."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
