# TerraHeal: Recovery Cold Spots

## Unmasking Lingering Wildfire Damage with Satellite Intelligence

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green?style=flat-square&logo=google-cloud)](https://earthengine.google.com/)


---

## 💡 The Problem: The Silent Scars of Wildfires

Wildfires leave behind devastating immediate impacts, but the long-term ecological consequences often go unnoticed. While most burned areas are expected to recover over time, some patches of land exhibit **anomalously slow or stalled vegetation regrowth** for months or even years post-fire. These "Recovery Cold Spots" are critical indicators of severe, persistent damage, including:

* **Severe Soil Degradation:** Loss of topsoil, altered hydrology, nutrient depletion.
* **Loss of Native Seed Bank:** Preventing natural regeneration of indigenous plant species.
* **Ecosystem Shift:** Vulnerability to invasive species, leading to reduced biodiversity and altered ecological functions.

Current wildfire analysis primarily focuses on immediate burn severity. Our project addresses the crucial gap of tracking the *rate and nature of recovery over time*, identifying these hidden ecological traps that generic assessments might miss.

## ✨ The Solution: Precision Satellite Intelligence

**TerraHeal** is an innovative Earth Observation project designed to automatically identify, map, and analyze these "Recovery Cold Spots" using advanced satellite imagery and machine learning. We provide actionable intelligence to guide targeted ecological restoration efforts, fostering long-term resilience in fire-affected landscapes.

### Key Features:

* **Wildfire Event Selection:** Easily select historical wildfire events for detailed analysis.
* **Multi-Stage Mapping:** Visualize recovery progression through:
    * **Pre-Fire Vegetation Maps:** Establishing a baseline of ecological health.
    * **Immediate Post-Fire Burn Maps:** Quantifying the initial impact and extent.
    * **Recovery Cold Spots Maps:** Precisely highlighting areas of stalled vegetation regrowth.
* **Time-Series Analysis:** Track vegetation health (e.g., using NVTI, NDVI) at critical post-fire intervals (1, 3, 6 months, and annually) to capture dynamic changes.
* **Proprietary Anomaly Detection Model:** Our core innovation, leveraging statistical and machine learning algorithms to identify significant deviations from expected recovery trajectories.
* **Contextual Analysis:** Overlay cold spots with ancillary data (soil types, topography/shaded relief) to infer underlying causes of delayed recovery.
* **Predictive Features (Future):** Forecast potential future risks like erosion hotspots, invasive species vulnerability, and long-term ecosystem shifts.
* **Intuitive UI (Planned):** A modern, SaaS-like dashboard for easy access and visualization of complex data.

## 🚀 Technical Details

### Technology Stack:

* **Primary Language:** Python
* **Cloud-Based Geospatial Processing:** Google Earth Engine (GEE)
* **Raster Data Handling:** Rasterio, GDAL, Xarray, rioxarray
* **Vector Data Handling:** GeoPandas, Shapely
* **Data Manipulation:** NumPy, pandas
* **Machine Learning & Statistics:** scikit-learn, scipy.stats
* **Development Environment:** Google Colab
* **Visualization (Current/Planned):** Matplotlib, Folium, Streamlit (for dashboard deployment)

### Data Sources:

* **Sentinel-1 (Radar):** For all-weather surface change detection.
* **Sentinel-2 (Optical):** High-resolution (10m) multi-spectral imagery for vegetation indices (e.g., NVTI, NDVI).
* **Landsat Time Series:** Decades of historical optical data for robust pre-fire baselines.
* **MODIS:** Broad-scale, daily context data.
* **Burn Scar Data:** Copernicus data products, or derived dNBR (differenced Normalized Burn Ratio) from Sentinel/Landsat.
* **Ancillary Data:** Soil type maps (e.g., SoilGrids), Digital Elevation Models (DEMs) for slope and shaded relief analysis.

## 🔬 Methodology: Our Scientific Approach

1.  **Data Acquisition & Preprocessing:** Automated fetching of Sentinel/Landsat imagery from GEE, applying cloud masks, and atmospheric corrections.
2.  **Vegetation Index Time-Series Generation:** Calculation of spectral vegetation indices (like NVTI, NDVI) at specific post-fire intervals (1, 3, 6 months, and annually) to track detailed recovery progress.
3.  **Baseline Recovery Modeling:** Stratification of the burn scar by initial burn severity and pre-fire vegetation type; establishing statistically sound "expected" recovery curves for each stratum.
4.  **Proprietary Anomaly Detection Model:** Statistical and machine learning algorithms rigorously compare each pixel's recovery trajectory against its predicted baseline, identifying significant negative deviations as "cold spots." We also analyze and group **similar shapes of cold spots** to categorize recovery failures.
5.  **Contextual Analysis:** Overlaying identified cold spots with soil, slope, and other environmental data to infer potential causative factors for delayed recovery.

## 📊 Impact: Why TerraHeal Matters

* **Targeted Restoration:** Directs limited resources to the *exact* areas where intervention is most critical, maximizing ecological recovery efforts.
* **Reduced Environmental Risk:** Proactively mitigates erosion, water quality degradation, and biodiversity loss post-wildfire.
* **Enhanced Disaster Management:** Provides a missing piece for comprehensive, long-term post-fire assessment for **all wildfires globally**.
* **Advanced Scientific Understanding:** Offers novel insights into complex ecological recovery processes.
* **Community & Economic Resilience:** Supports the long-term health, safety, and economic vitality of fire-affected regions.

## 🎬 Prototype & Demo

See TerraHeal in action! Our prototype demonstrates how users can easily select existing wildfires, visualize pre-fire, post-fire, and critical cold spot maps, and observe dynamic changes across various post-fire intervals.

**Watch the Demo Video:** [Google Drive Link to your Demo Video]
*(Ensure the link is publicly viewable)*

## 🗺️ Project Roadmap & Future Enhancements

* **Phase 1 (Current):** Core Cold Spot Detection (including NVTI/spectral coverage, 1-3-6 month intervals), Interactive Mapping (pre-fire, post-fire, cold spots), Wildfire Selection.
* **Phase 2 (Near-Term):** Deeper Integration of Predictive Features (e.g., highly refined erosion modeling, invasive species vulnerability maps), User Collaboration Tools, API Integration.
* **Phase 3 (Long-Term):** Global Coverage Expansion & Automation for all wildfires, AI-driven Restoration Recommendations, Integration with Climate Change Scenarios, Real-time Alerting, Mobile Accessibility.


---
