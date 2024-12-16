# Multi-Band Satellite Image Processing and Landsat Image Clustering Workflow

## Introduction

This repository serves as a comprehensive resource for performing **multi-band satellite imagery processing** and **unsupervised clustering analysis** using data from Landsat 8. Designed to handle the complexities of modern geospatial workflows, it combines advanced cloud-based geospatial tools and cutting-edge machine learning methods to process and analyze large-scale geospatial datasets efficiently. The workflow focuses on **dynamic spectral band handling** and clustering spectral signatures, making it ideal for diverse applications such as land cover classification, vegetation monitoring, and environmental studies.

The pipeline is designed to integrate seamlessly with **STAC-compliant APIs (SpatioTemporal Asset Catalogs)**, ensuring easy access to structured metadata and imagery from various sources. It is optimized for cloud-hosted geospatial data platforms, such as the **Microsoft Planetary Computer (MPC)**, reducing the need for extensive local storage or compute power. By utilizing **cloud-native processing technologies**, this workflow minimizes reliance on local resources and takes advantage of scalable, distributed computing frameworks to handle the immense volume and complexity of satellite data. Tools like **rioxarray**, **xarray**, and **Dask** ensure efficient raster data processing, while machine learning libraries like **scikit-learn** enable unsupervised clustering and analysis.

### Key Highlights of the Workflow
1. **Cloud-Native Geospatial Processing**: The pipeline leverages cloud-hosted geospatial assets, eliminating the need for manual downloading and preprocessing of satellite imagery.
2. **Dynamic and Scalable**: Built with tools that support distributed computation, allowing for scalability as data volume grows.
3. **End-to-End Workflow**: Covers every stage of satellite data analysis, from querying data sources to deriving insights through clustering.

Whether you are a geospatial data scientist, environmental researcher, or student, this repository provides a robust, modular foundation for tackling large-scale geospatial challenges in a modern, cloud-enabled environment.

---

## Technologies Used

The repository brings together a powerful combination of Python libraries, APIs, and geospatial data sources. Below is a detailed overview of the technologies and tools used:

### Python Libraries
This workflow relies heavily on Python's ecosystem of geospatial and machine learning libraries. Each tool serves a specific purpose, ensuring the pipeline is both efficient and easy to use:

- **pystac-client**:  
   Enables seamless querying and retrieval of STAC items. This library simplifies interaction with STAC-compliant APIs, allowing you to search for and filter geospatial data based on specific criteria such as time range, location, and cloud cover percentage.

- **planetary-computer**:  
   Provides the tools necessary to authenticate and access geospatial data hosted on the **Microsoft Planetary Computer (MPC)**. It ensures smooth integration with MPC datasets, enabling you to work with imagery like Landsat and Sentinel data directly in your Python environment.

- **rioxarray/xarray**:  
   Facilitates advanced raster data processing. These libraries handle the ingestion, manipulation, and analysis of multi-dimensional data arrays, making it possible to extract and analyze spectral bands from satellite imagery with ease.

- **Dask**:  
   Adds the capability for distributed and parallel processing. Dask is used to manage large datasets and perform computations that would otherwise be constrained by memory limitations.

- **shapely**:  
   Used for manipulating and analyzing geometric shapes. Shapely plays a critical role in defining and working with spatial boundaries, such as Areas of Interest (AOIs), which are essential for clipping imagery to a specific region.

- **matplotlib**:  
   Powers the visualization aspect of the workflow. Matplotlib is used to create informative plots, such as scatter plots, histograms, and clustering result visualizations.

- **scikit-learn**:  
   Provides tools for machine learning and statistical analysis. Scikit-learn is primarily used for unsupervised clustering in this workflow, enabling analysis of spectral signatures and the identification of patterns within the data.

---

### Data Sources

The workflow is designed to work with data hosted on **cloud-native platforms**, ensuring rapid access to high-quality satellite imagery. Below are the primary data sources utilized:

- **Microsoft Planetary Computer (MPC)**:  
   This pipeline integrates seamlessly with MPC, leveraging its extensive catalog of geospatial datasets. Key datasets used in this workflow include:  
   - **Landsat Collection 2 Level 2 Imagery**: Provides high-quality, preprocessed imagery with surface reflectance and other derived products.  
   - **Sentinel-2 L2A Imagery**: Includes high-resolution multispectral imagery suitable for vegetation monitoring, land use analysis, and more.  
   - **Derived Datasets**: Calculated indices such as the **Normalized Difference Vegetation Index (NDVI)** and thermal indices, which are critical for environmental monitoring and resource management.

By combining these tools and datasets, this repository provides a scalable, flexible, and efficient framework for geospatial data analysis. Whether working with large-scale datasets or performing localized analysis, this pipeline offers the capabilities needed to derive actionable insights from satellite imagery.

---

## Workflow Overview

This section outlines the entire geospatial processing and analysis pipeline, including critical steps for querying, processing, and analyzing satellite imagery. Each step is explained in detail, including potential challenges or errors that may arise and how to address them effectively.

---

### 1. Querying STAC Items

The first step involves querying a **SpatioTemporal Asset Catalog (STAC)** to retrieve relevant imagery metadata for the selected **Area of Interest (AOI)** and **time range**.  
- **Define AOI and Time Range**: The AOI is defined using a polygon geometry (typically a GeoJSON format) that specifies the geographic region to focus on. This region should be well-defined to ensure accurate results. The time range allows you to filter imagery from a specific time period, ensuring that you are working with relevant data for your analysis.  
- **Apply Filters**: Filters are applied to the query to refine the search results. Filters such as **maximum cloud cover percentage** ensure that the retrieved images are of sufficient quality for analysis, excluding images with excessive cloud coverage. Other filters might include filtering by product type or sensing instrument.  
- **Use MPC's STAC API**: The **pystac-client** library makes it easy to query the **Microsoft Planetary Computer (MPC)**, which provides access to a vast repository of satellite imagery. This library allows users to interact with STAC catalogs, retrieve metadata, and access links to the actual satellite imagery.

#### Potential Errors:
- **Incorrect AOI Geometry**: If the AOI geometry is malformed or misdefined (e.g., invalid GeoJSON format), the API request will fail. It is crucial to validate the geometry before querying the STAC API.  
- **Empty Search Results**: This error occurs if the time range or filters are too restrictive, resulting in no matching images. You can adjust your search criteria by widening the time window or relaxing the cloud cover thresholds to ensure you get valid results.  
- **Authentication Issues**: Errors with MPC API authentication can occur if the authentication tokens are expired, invalid, or incorrectly configured. Ensure that authentication is properly set up using the **planetary-computer** library, and check for any issues related to token renewal.

---

### 2. Processing Spectral Bands

Once the imagery has been retrieved, the next step is to load and organize the imagery data into various spectral bands, such as **RGB**, **NIR**, **SWIR**, and **thermal** bands. These bands contain essential information for analyzing the Earth's surface and are the foundation for further analysis.  
- **Identify and Extract URLs**: Use the metadata retrieved from the STAC query to locate the specific URLs of the spectral bands (such as **RGB**, **NIR**, **SWIR**, and **thermal** bands). These URLs point to **Cloud-Optimized GeoTIFFs (COGs)**, which are hosted on the cloud and optimized for efficient access and processing.  
- **Load Raster Data**: The **rioxarray** library is used to load raster imagery as lazy arrays, meaning that the data is loaded in chunks, reducing memory usage. This is especially important when dealing with large-scale datasets, as it allows for efficient processing without consuming excessive system memory.

#### Potential Errors:
- **Missing or Incorrect URLs**: Sometimes, the URLs for the spectral bands may be missing, incorrect, or broken. This can result in failed attempts to load the raster data. Double-check the URLs in the metadata and ensure they are accessible.  
- **File Format Issues**: This workflow is designed to work with **Cloud-Optimized GeoTIFFs (COGs)**. If the imagery is not in COG format, errors may arise when trying to load the data. Be sure to check the file format before attempting to load it.  
- **Memory Limitations**: Despite lazy loading, large datasets may still consume considerable memory. To mitigate this, consider using **Dask** for parallel processing, which can efficiently handle large datasets across multiple CPU cores or even multiple machines.

---

### 3. Mosaicking and Clipping

Once the spectral bands are loaded, the next step is to combine overlapping rasters and clip the final output to the **AOI**. This ensures that the data is tailored to the area of interest and that the spatial resolution is consistent across the entire image.  
- **Merge Overlapping Rasters**: Multiple raster images that cover the same area but with slight overlaps need to be merged into a single image. The **rioxarray.merge.merge_arrays** function handles this task efficiently, creating a seamless mosaic from the overlapping rasters.  
- **Clip to AOI**: After the rasters are merged, they need to be clipped to the exact boundaries of the AOI. This ensures that the final output only contains the region of interest, removing any irrelevant areas outside the AOI.

#### Potential Errors:
- **Merging Conflicts**: If the rasters to be merged have different spatial resolutions or coordinate reference systems (CRS), merging them can lead to errors. It is important to ensure that all rasters share the same CRS and resolution before attempting to merge. If the CRS is different, you may need to reproject one or more rasters to match the others.  
- **Clipping Issues**: If the AOI does not overlap with the raster data or if the AOI is misdefined, the clipping process will result in an empty output. Ensure that the AOI is correctly defined and overlaps with the raster data.

---

### 4. Exporting Multi-Band GeoTIFF

Once the image has been mosaicked and clipped to the AOI, the next step is to export the processed bands into a **multi-band GeoTIFF**. This format allows you to retain all the spectral information for downstream analysis, such as clustering or classification.  
- **Combine Bands**: The various spectral bands are stacked into a multi-dimensional array, where each band corresponds to a different spectral feature (e.g., RGB, NIR, SWIR). This stacking ensures that all the spectral information is preserved in a single file.  
- **Export as GeoTIFF**: Using **rioxarray.to_raster()**, the stacked array is written to disk as a multi-band GeoTIFF. This file can be used for further analysis or exported to other tools for visualization.

#### Potential Errors:
- **Alignment Issues**: It is important that all bands are aligned correctly in terms of their spatial dimensions. If the bands do not align, the export process will fail. Misaligned bands may arise if the rasters had different resolutions or extents prior to merging.  
- **File Writing Limitations**: Ensure that there is sufficient disk space and that the file path is correct. If disk space is insufficient, the export will fail. Similarly, ensure that the destination directory has appropriate write permissions.

---

### 5. Clustering Spectral Signatures

After preparing the imagery, the next step is to apply **unsupervised clustering** to group pixels based on their spectral similarity. This allows for the identification of patterns within the data, such as different types of land cover.  
- **Stack Bands**: The selected spectral bands are organized into a multi-dimensional array, where each pixel is a point in feature space, with each feature corresponding to a spectral band. This array serves as input for the clustering algorithm.  
- **Apply MiniBatch KMeans**: **MiniBatch KMeans** from **scikit-learn** is used to cluster the pixels into groups based on their spectral characteristics. This algorithm is particularly useful for large datasets, as it performs faster by using small random batches of data at each iteration.  
- **Analyze Clusters**: After clustering, you can compute the average spectral signature for each cluster, which provides insight into the typical characteristics of each group.

#### Potential Errors:
- **High Dimensionality**: If too many spectral bands are used, the clustering algorithm may struggle with high-dimensional data. Dimensionality reduction techniques like **Principal Component Analysis (PCA)** can be used to reduce the number of features.  
- **Poor Cluster Separation**: If the clusters are not well separated, you may need to adjust the number of clusters or experiment with different distance metrics. In some cases, poor clustering can also indicate that the spectral bands used are not sufficiently distinct.

---

### 6. Visualization and Analysis

Finally, visualizing the results of the clustering is key to interpreting the data and understanding the relationships between different spectral signatures.  
- **Clustered Images**: After clustering, you can visualize the results by assigning distinct colors to each cluster, making it easy to see spatial patterns in the data. This is particularly useful for understanding the spatial distribution of different land cover types.  
- **Spectral Signatures**: You can also plot the average spectral signatures for each cluster, providing insights into the typical spectral characteristics of each group. This can help to identify specific land cover types, such as water, vegetation, or urban areas.

#### Potential Errors:
- **Plot Overcrowding**: If the number of clusters is too high, the visualizations may become crowded and difficult to interpret. It is helpful to limit the number of clusters or focus on specific regions for better clarity.  
- **Color Assignments**: Poorly chosen color maps can make it hard to distinguish between clusters. It's essential to use clear and distinct color schemes, especially if the results will be shared with a wide audience.

---

## Conda Environment Setup

### Environment File: `environment.yml`

The provided **environment.yml** file simplifies the setup of a reproducible Python environment. This ensures compatibility with the libraries and tools used in the workflow. The environment contains all necessary dependencies for geospatial data processing and clustering analysis.

```yaml
name: geospatial-clustering
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
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
      - xarray
---

## LINES 178-200 NEED UPDATING/EXPANDING

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-band-workflow.git
   cd multi-band-workflow
   ```
2. Set up the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate geospatial-clustering
   ```

---

## How to Run the Workflow

1. **Modify Parameters**: Update AOI, time range, and other parameters in the script.
2. **Run the Script**:
   ```bash
   python multi_band_workflow.py
   ```

---

## Visualization and Analysis

Clustering and spectral data analysis can yield valuable insights, but understanding the results often requires more than numerical metrics or raw data. Visualizations are a crucial part of the analysis workflow because they provide an intuitive way to explore the data, assess the quality of clustering, and identify patterns that might otherwise go unnoticed. This section introduces a series of plots that have been carefully designed to help you interpret the clustering results and evaluate the overall performance of the process. 

By incorporating these visualizations into your analysis, you can make informed decisions about data preprocessing, clustering parameter selection, and result interpretation. Each visualization serves a specific purpose, allowing you to examine the data from multiple perspectives.

### Why Are Visualizations Important?

In clustering workflows, visualizations are not just a helpful addition—they are often essential. Since clustering is an unsupervised learning technique, the ground truth labels for data points are typically unavailable, making it challenging to quantify the success of a clustering algorithm. Visualizations provide the ability to:
- **Assess Cluster Quality:** By showing how well-separated the clusters are, how cohesive individual clusters appear, and whether there are significant overlaps between clusters.
- **Identify Optimal Parameters:** Certain visualizations, such as the elbow plot, can guide the selection of the number of clusters (`k`) or other hyperparameters.
- **Detect Issues:** Problems such as imbalanced cluster sizes, outliers, or poorly defined clusters can be spotted easily in well-designed plots.
- **Communicate Findings:** Clear visual representations of data and results are invaluable when explaining clustering outputs to collaborators, stakeholders, or clients.

By using the visualizations provided in this workflow, you will gain a deeper understanding of the results and their implications, ensuring that your clustering process is both rigorous and effective.

### Overview of Visualizations

The following visualizations are included in this workflow, with detailed explanations and accompanying code snippets to help you create and interpret each plot:

1. **Silhouette Plot**  
   A silhouette plot is a powerful tool for evaluating clustering quality. For each data point, it calculates a "silhouette coefficient" that reflects how similar the point is to its own cluster compared to other clusters. These coefficients range from -1 to 1:
   - **Positive values (closer to 1):** Indicate that the point is well-clustered and closer to other points in its assigned cluster.
   - **Negative values:** Suggest that the point might belong to a different cluster.  

   The plot groups points by cluster, arranging them in order of their silhouette values. Additionally, a dashed red line represents the average silhouette score across all clusters, giving you an overall measure of clustering performance. Ideally, most points will have high silhouette values, and the clusters will be of roughly equal size.

2. **Elbow Method Plot**  
   Selecting the optimal number of clusters is a fundamental challenge in clustering analysis. The elbow method addresses this by plotting the within-cluster sum of squares (inertia) against the number of clusters (`k`). As `k` increases, inertia decreases because more clusters mean less distance between points and their cluster centroids. However, beyond a certain point, adding more clusters yields diminishing returns, resulting in a noticeable "elbow" in the plot.  
   - **Sharp drop (left of the elbow):** Indicates significant improvement in clustering quality as more clusters are added.
   - **Flattening curve (right of the elbow):** Suggests that additional clusters are not providing substantial benefits.  

   The "elbow point" marks the ideal balance between cluster quality and model complexity, guiding you to the appropriate value for `k`.

3. **Cluster Size Distribution Plot**  
   This plot visualizes the size of each cluster by showing the number of points assigned to it. The goal is to ensure that clusters are reasonably balanced:
   - **Well-balanced clusters:** Indicate effective clustering and a meaningful separation of data into groups.
   - **Imbalanced clusters:** Where one or more clusters are disproportionately large or small, may suggest issues such as overfitting, poor initialization, or noisy data.  

   The bar chart provides a quick overview of how the data points are distributed across clusters, helping you assess the clustering solution's reliability.

4. **2D Scatter Plot of Clusters**  
   The 2D scatter plot is one of the most intuitive ways to visualize clustering results. It maps data points onto a two-dimensional space using two selected features (e.g., spectral bands or principal components). Each point is colored according to its assigned cluster, allowing you to:
   - Visually assess the separation and cohesion of clusters.
   - Identify overlaps or ambiguities between clusters.
   - Gain insights into the structure of the data.  

   Cluster centroids (if plotted) provide additional context, showing the central position of each cluster in the feature space. This plot is particularly helpful for exploring the relationships between features or validating the clustering results.

### How to Use These Visualizations

Each visualization includes a detailed Python code snippet that you can execute to generate the corresponding plot. These snippets are designed to work seamlessly with your clustering workflow, leveraging popular Python libraries such as:
- `matplotlib` for creating plots and visualizations.
- `scikit-learn` for clustering algorithms and metrics.
- `numpy` for efficient data manipulation and computation.  

The descriptions provided for each plot will guide you through their interpretation, ensuring that you understand the insights they reveal. By integrating these visualizations into your analysis, you can not only evaluate the quality of your clustering but also refine your process and make data-driven decisions.

### Final Thoughts

Visualizations are a bridge between raw data and actionable insights. By leveraging the power of graphical analysis, you can uncover patterns, validate results, and communicate findings more effectively. The plots described in this section are an integral part of any robust clustering workflow, empowering you to make informed decisions and extract maximum value from your data.


### Silhouette Plot
#### Description
A silhouette plot shows a bar for each data point, with the length of the bar representing how well the point fits within its assigned cluster.

- **High silhouette value (close to 1)**: Well-clustered points.
- **Negative silhouette value**: Points that might belong to a different cluster.

#### Code
```python
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

valid_data_reshaped = valid_data.reshape(-1, 1)
cluster_labels = kmeans.predict(valid_data_reshaped)
silhouette_vals = silhouette_samples(valid_data_reshaped, cluster_labels)
silhouette_avg = silhouette_score(valid_data_reshaped, cluster_labels)
print(f"Average Silhouette Score: {silhouette_avg:.3f}")

plt.figure(figsize=(10, 7))
y_lower = 10

for i in range(n_clusters):
    ith_cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    ith_cluster_silhouette_vals.sort()
    size_cluster_i = ith_cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals, alpha=0.7, label=f'Cluster {i}')
    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg, color='red', linestyle='--', label='Average Silhouette Score')
plt.title("Silhouette Plot for KMeans Clustering", fontsize=14)
plt.xlabel("Silhouette Coefficient Values", fontsize=12)
plt.ylabel("Cluster Labels", fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```

---

### Elbow Method Plot
#### Description
The elbow plot helps determine the optimal number of clusters by identifying where the inertia curve flattens.

#### Code
```python
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

valid_data_reshaped = valid_data.reshape(-1, 3)
cluster_range = range(1, 15)
inertia_values = []

for n_clusters in cluster_range:
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    kmeans.fit(valid_data_reshaped)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(cluster_range, inertia_values, marker='o', linestyle='--', color='b')
plt.title("Elbow Method for Optimal Clusters", fontsize=14)
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
plt.xticks(cluster_range)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

---

### Cluster Size Distribution
#### Description
Bar chart displaying the number of points assigned to each cluster to assess balance.

#### Code
```python
import matplotlib.pyplot as plt
import numpy as np

unique_labels, counts = np.unique(cluster_labels, return_counts=True)

plt.figure(figsize=(10, 6))
plt.bar(unique_labels, counts, color='skyblue', edgecolor='black')
plt.title("Cluster Size Distribution", fontsize=14)
plt.xlabel("Cluster Label", fontsize=12)
plt.ylabel("Number of Pixels", fontsize=12)
plt.xticks(unique_labels)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

---

### 2D Scatter Plot of Clusters
#### Description
Scatter plot of data points in a two-dimensional space, colored by cluster assignment.

#### Code
```python
import matplotlib.pyplot as plt

red_band = 'red_band_name'
nir_band = 'nir_band_name'

red_values = final_mosaic[red_band].values.flatten()
nir_values = final_mosaic[nir_band].values.flatten()
valid_mask = (~np.isnan(red_values)) & (~np.isnan(nir_values)) & (red_values > 0) & (nir_values > 0)
red_values = red_values[valid_mask]
nir_values = nir_values[valid_mask]

plt.figure(figsize=(10, 8))
plt.scatter(red_values, nir_values, s=1, c='blue', alpha=0.5)
plt.title("2D Scatter Plot of Red vs NIR Values", fontsize=14)
plt.xlabel("Red Band Reflectance", fontsize=12)
plt.ylabel("NIR Band Reflectance", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```
