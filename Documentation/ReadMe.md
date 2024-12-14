# Multi-Band Satellite Image Processing Workflow

## Background

This repository presents an end-to-end pipeline for processing **multi-band satellite imagery** using modern cloud-native geospatial tools. The workflow is tailored to work with **STAC-compliant APIs** (SpatioTemporal Asset Catalog) and cloud-hosted geospatial data like those provided by **Microsoft's Planetary Computer (MPC)**. 

Instead of focusing only on traditional RGB bands (Red, Green, Blue), the pipeline dynamically processes **all available spectral bands**, including NIR, SWIR, and thermal bands. This flexibility makes it suitable for diverse applications such as vegetation monitoring, urban mapping, and climate studies.

## Environment Setup

This project uses a **Conda environment** to manage dependencies for the multi-band satellite image processing pipeline. Below is the `environment.yml` file that you can use to set up the environment. The environment ensures compatibility with all required libraries, including geospatial, machine learning, and visualization tools.

### Conda Environment File: `environment.yml`

```yaml
name: geospatial-clustering
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9  # You can change the Python version as needed
  - planetary-computer
  - pystac-client
  - xarray
  - rioxarray
  - numpy
  - dask
  - dask-distributed
  - pyproj
  - rasterio
  - shapely
  - requests
  - tqdm
  - matplotlib
  - scikit-learn
  - pip
  - pip:
      - xarray  # For compatibility with rioxarray dependencies
```

### What is STAC?

The **STAC Specification** provides a unified way to describe and interact with geospatial data assets. A STAC API enables:
- Efficient **search** and **retrieval** of satellite imagery based on spatial, temporal, and metadata filters.
- Access to diverse geospatial datasets like **Landsat**, **Sentinel**, and derived products such as NDVI.

**Microsoft Planetary Computer** is a robust platform providing access to large-scale STAC-compliant datasets for public use. These datasets include **Cloud Optimized GeoTIFFs (COGs)**, enabling efficient access and processing without requiring local downloads.

### Objectives of This Workflow

The main goal of this workflow is to **streamline multi-band satellite image processing** while minimizing local resource constraints. Key objectives include:
1. **Dynamic Band Handling**: Process all spectral bands available in the dataset, not limited to RGB.
2. **Scalability**: Utilize **Dask** for distributed computing, enabling the processing of large datasets.
3. **Cloud-Native Integration**: Leverage cloud-hosted COGs and perform computations in a scalable, memory-efficient manner.

---

## Key Features

### Dynamic Band Support
Unlike conventional workflows that focus on specific bands, this pipeline:
- Automatically identifies **all available bands** from STAC item assets (e.g., Red, Green, Blue, NIR, SWIR).
- Processes each band independently and merges them into a single multi-band mosaic.

### Cloud-Native Processing
By utilizing **rioxarray** and **Dask**, the pipeline operates directly on cloud-hosted datasets:
- No need for large local storage.
- Lazy loading ensures that only necessary parts of the data are loaded into memory.

### Distributed Computing
The pipeline incorporates **Dask** to distribute tasks across multiple workers:
- Parallel reading and merging of raster data.
- Efficient processing of large AOIs.

### Seamless Output
The processed data is exported as a **multi-band GeoTIFF** that retains all spectral information:
- Ideal for use in downstream analysis like machine learning, land cover classification, and temporal studies.

---

## Technologies Used

### Python Libraries
- **pystac-client**: For querying and retrieving STAC items from the MPC API.
- **planetary-computer**: To authenticate and access Microsoft Planetary Computer assets.
- **rioxarray/xarray**: For reading, processing, and exporting raster data.
- **Dask**: For distributed and parallel processing.
- **shapely**: To define and manipulate geometric shapes (e.g., AOIs).
- **matplotlib**: For visualization of results.
- **scikit-learn**: Optional use for clustering and data analysis.

### Data Sources
- **Microsoft Planetary Computer** provides access to:
  - **Landsat Collection 2 Level 2** imagery.
  - **Sentinel-2 L2A** data.
  - Derived and ancillary datasets for advanced analysis.

---

## Workflow Overview

### 1. Querying STAC Items
The workflow queries the MPC STAC API with the following parameters:
- **Bounding Box (AOI)**: Define the area of interest (e.g., [-79.5, 37.9, -78.5, 38.9] for Charlottesville, VA).
- **Time Range**: Specify a date range for the data (e.g., "2020-01-01/2020-12-31").
- **Cloud Cover Filter**: Retrieve imagery with low cloud cover (e.g., < 2%).

### 2. Extracting Band URLs
All spectral bands available in the retrieved items are dynamically identified. URLs for each band are collected for further processing.

### 3. Lazy Loading with Dask
Raster data is opened lazily using **rioxarray**:
- Each band is loaded into a Dask-backed **xarray.Dataset**, ensuring efficient memory usage.

### 4. Mosaicking
Overlapping images within each band are merged into seamless mosaics:
- Uses **rioxarray.merge** to handle alignment and overlaps.

### 5. Clipping to AOI
The mosaicked bands are clipped to the exact bounding box of the AOI using **rio.clip**.

### 6. Multi-Band GeoTIFF Export
All processed bands are stacked into a single **multi-band GeoTIFF**:
- The exported file can be directly used for analysis in GIS tools or machine learning pipelines.

---

## Getting Started

### Prerequisites
1. Install Python (>= 3.8).
2. Install dependencies using `pip`:
   ```bash
   pip install pystac-client planetary-computer rioxarray dask shapely matplotlib scikit-learn
   ```

# Detailed Landsat Image Clustering and Spectral Signature Analysis

This repository demonstrates a comprehensive approach to working with Landsat 8 imagery, focusing on clustering the spectral data using unsupervised machine learning techniques, computing the average spectral signatures for each cluster, and visualizing the results. The script facilitates access to the Microsoft Planetary Computer (MPC), which provides satellite data through the SpatioTemporal Asset Catalog (STAC) API. By using this approach, users can search, access, and process large volumes of Landsat data for analysis without needing to download the images manually.

In particular, this example focuses on:
- Searching and retrieving Landsat 8 images based on a defined area of interest (AOI) and time range.
- Extracting and processing relevant spectral bands (specifically, the red, green, and blue bands) for clustering.
- Applying an unsupervised machine learning algorithm (MiniBatch KMeans) to segment the image into clusters based on spectral similarity.
- Visualizing the clustered results and comparing the average spectral signatures for each cluster.

The resulting analysis can be used for various applications, including land cover classification, vegetation analysis, environmental monitoring, urban development tracking, and change detection. The methods demonstrated here provide a flexible framework for working with satellite imagery, enabling users to easily adapt the script for different regions, times, or types of analysis.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [How to Run the Script](#how-to-run-the-script)
4. [Detailed Explanation of the Code](#detailed-explanation-of-the-code)
    1. [Step 1: Searching and Loading a Landsat Image from MPC](#step-1-searching-and-loading-a-landsat-image-from-mpc)
    2. [Step 2: Opening and Stacking Relevant Bands](#step-2-opening-and-stacking-relevant-bands)
    3. [Step 3: Masking Invalid Data](#step-3-masking-invalid-data)
    4. [Step 4: Applying MiniBatch KMeans Clustering](#step-4-applying-minibatch-kmeans-clustering)
    5. [Step 5: Visualizing the Results](#step-5-visualizing-the-results)
5. [Additional Considerations](#additional-considerations)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction

The goal of this script is to demonstrate how to efficiently retrieve, process, and analyze Landsat 8 satellite imagery using the Microsoft Planetary Computer (MPC) platform. Landsat 8 provides high-resolution multispectral imagery that is widely used in remote sensing applications, from environmental monitoring to land use/land cover classification.

In this workflow, we first access Landsat 8 data using the MPC’s STAC API, which allows us to search for and retrieve satellite images based on specific metadata such as geographic area, date range, and satellite collection. Once the imagery is retrieved, we use the `rioxarray` library to load the specific spectral bands (such as red, green, and blue) that are typically used in RGB visualization. These bands are stacked together into a multi-dimensional array to form the basis for further analysis.

The unsupervised clustering algorithm, MiniBatch KMeans, is then applied to the pixel data in these bands. This algorithm groups similar pixels together based on their spectral characteristics, effectively segmenting the image into regions that share similar reflectance patterns. Each resulting cluster represents an area of the image with common spectral features, which could correspond to specific land cover types such as vegetation, water, or urban areas.

Finally, the clustered image is visualized, allowing for an intuitive understanding of the spatial distribution of the different clusters. Additionally, the average spectral signature for each cluster is computed and plotted, showing the reflectance values for each of the RGB bands. These average spectral signatures provide insight into the typical characteristics of each cluster, such as the differences in reflectance between vegetation and urban areas.

The entire process is highly adaptable, meaning users can modify parameters such as the number of clusters, geographic area, or time period to suit different use cases. By using this approach, users can quickly process large volumes of satellite data and extract meaningful information about the Earth’s surface.

## Prerequisites

Before running this script, ensure that you have the following libraries installed in your Python environment. These libraries are essential for querying, processing, and analyzing Landsat 8 imagery, as well as for performing machine learning and visualization tasks.

### Required Libraries:

- **`pystac_client`**: A Python library for interacting with the SpatioTemporal Asset Catalog (STAC) API, which is used to query and retrieve metadata and assets (e.g., satellite images) from the MPC.
- **`planetary_computer`**: A library that simplifies the process of signing URLs for accessing data hosted on the Microsoft Planetary Computer, ensuring that the data retrieval is secure and authorized.
- **`rioxarray`**: A library built on `xarray` that allows for the reading and processing of raster data in cloud-optimized GeoTIFF format (COG). It supports efficient, on-the-fly processing of large geospatial datasets without the need to download the entire image.
- **`numpy`**: A library for numerical computing, widely used for manipulating arrays and performing mathematical operations on large datasets.
- **`matplotlib`**: A versatile library for creating visualizations, used in this script to generate plots of the clustered image and average spectral signatures.
- **`scikit-learn`**: A machine learning library that provides the `MiniBatchKMeans` algorithm used in this script to perform unsupervised clustering on the image data.

### Installation:

To install the required libraries, run the following command in your terminal or command prompt:

```bash
pip install pystac_client planetary_computer rioxarray numpy matplotlib scikit-learn
```

---

## How to Run the Script

Once the environment is set up and dependencies are installed, you can run the script as follows:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/landsat-clustering.git
   cd landsat-clustering
   ```

2. Activate the Conda environment:
   ```bash
   conda activate geospatial-clustering
   ```

3. Modify the parameters in the script to suit your needs (e.g., AOI, time range, number of clusters).

4. Run the script:
   ```bash
   python landsat_clustering.py
   ```

The script will then fetch Landsat data from the Microsoft Planetary Computer, apply clustering, and generate output visualizations and spectral signature plots.

---

## Detailed Explanation of the Code

### Step 1: Searching and Loading a Landsat Image from MPC
The first step is to query the MPC STAC API for Landsat 8 imagery. The script defines parameters like the **area of interest (AOI)**, **date range**, and **cloud cover** to filter images. Once the query is executed, the relevant STAC items are returned, and the URLs for the required bands (e.g., red, green, blue) are extracted.

### Step 2: Opening and Stacking Relevant Bands
Using the `rioxarray` library, the script opens the individual spectral bands (e.g., red, green, blue) as **cloud-optimized GeoTIFFs** (COGs). These bands are then stacked into a multi-dimensional `xarray.Dataset` object. This structure allows for efficient data handling and processing without downloading the entire image.

### Step 3: Masking Invalid Data
Some areas in the image might contain **no data** (e.g., clouds, invalid pixels). These are masked out during preprocessing. The script uses the **masking** feature of `rioxarray` to filter out these invalid regions, ensuring that only meaningful data is included in the analysis.

### Step 4: Applying MiniBatch KMeans Clustering
The **MiniBatch KMeans** algorithm from **scikit-learn** is applied to the stacked bands. The algorithm clusters pixels based on their **spectral similarity** across the three bands (RGB). The script allows you to specify the **number of clusters**, which will influence the granularity of the segmentation. The result is a labeled image where each pixel belongs to one of the identified clusters.

### Step 5: Visualizing the Results
Finally, the clustered image is visualized using **matplotlib**. The resulting image is displayed, where each cluster is represented by a unique color. Additionally, the script computes the **average spectral signature** for each cluster and plots these values to show the typical reflectance characteristics of each cluster.

---

## Additional Considerations

### Cloud Masking
The accuracy of the clustering algorithm can be affected by clouds, which may appear as outliers in the spectral data. Consider applying cloud masking techniques or using data with low cloud cover for better results.

### Temporal Analysis
For time-series analysis, you can modify the script to process multiple images over a specific time range and analyze the temporal changes in each cluster.

### Number of Clusters
The choice of the number of clusters in **MiniBatch KMeans** can significantly impact the results. Experiment with different numbers of clusters to identify the optimal segmentation for your use case.

---


## Example Applications

1. **Vegetation Monitoring**
   - Process NIR and Red bands to calculate NDVI for vegetation health analysis.
   - Multi-temporal studies to track changes over time.

2. **Urban Mapping**
   - Use SWIR bands to differentiate between urban areas and natural features.
   - Analyze impervious surface cover.

3. **Climate and Water Studies**
   - Leverage thermal bands to map surface temperature variations.
   - Identify water bodies and their dynamics using SWIR and NIR bands.

4. **Land Cover Classification**
   - Combine all spectral bands for supervised or unsupervised classification.
   - Use clustering algorithms like **KMeans** for segmentation.


---

## Conclusion

This workflow provides a robust pipeline for processing and analyzing Landsat 8 imagery. By leveraging cloud-based resources and advanced machine learning techniques, it enables efficient analysis of large-scale geospatial datasets. The ability to perform unsupervised clustering on multi-spectral data offers powerful insights for applications such as land cover classification, vegetation analysis, and environmental monitoring.

With this framework, you can easily modify the parameters to process different regions, time periods, or datasets, making it a flexible tool for diverse geospatial analysis tasks.

---

## References

- [Planetary Computer Documentation](https://planetarycomputer.microsoft.com/docs)
- [STAC API Specification](https://github.com/stac-utils/stac-api-spec)
- [Landsat 8 Documentation](https://www.usgs.gov/centers/eros/science/landsat-8)
- [Scikit-learn - MiniBatch KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
- [Dask Documentation](https://docs.dask.org/en/stable/)

