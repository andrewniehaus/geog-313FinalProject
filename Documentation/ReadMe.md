# Multi-Band Satellite Image Processing and Landsat Image Clustering Workflow

## Introduction

This repository serves as a comprehensive and versatile resource for performing **multi-band satellite imagery processing** and **unsupervised clustering analysis**, primarily using data from **Landsat 8**. It is designed to address and solve the complex challenges associated with modern geospatial workflows by integrating advanced cloud-based tools and cutting-edge machine learning methodologies. The workflow is specifically built to process and analyze large-scale geospatial datasets efficiently, while ensuring that users can manage data-intensive tasks with minimal effort. By combining cloud-based processing with machine learning techniques, this repository offers an easy-to-use and highly efficient solution for working with large satellite datasets, enabling users to extract meaningful insights from complex geospatial information.

The core focus of this repository is on **dynamic spectral band handling** and clustering **spectral signatures** from satellite imagery. This capability makes the workflow ideal for a variety of real-world applications, including but not limited to **land cover classification**, **vegetation monitoring**, and **environmental studies**. The flexibility of the tools and methodologies in this repository allows them to be used in a wide array of domains that require the analysis of satellite imagery, from urban planning and agriculture to climate change research and biodiversity conservation. The combination of spectral analysis and machine learning clustering makes it an invaluable resource for both researchers and professionals looking to gain deeper insights from satellite imagery and geospatial data.

The pipeline is designed with a high level of interoperability, allowing it to integrate seamlessly with **STAC-compliant APIs** (SpatioTemporal Asset Catalogs). These APIs are a standard way of storing and serving geospatial metadata, making it easier to search for and access satellite imagery and other geospatial datasets. By leveraging STAC-compliant APIs, the repository ensures that users can easily query structured metadata and retrieve imagery from a variety of geospatial data sources. This integration enhances the ability to work with multiple data providers, enabling access to a broad range of geospatial datasets that are essential for analysis. Whether users are working with publicly available datasets like Landsat and Sentinel-2 or proprietary datasets hosted by various platforms, the repository offers a streamlined method for obtaining and managing data.

This repository is optimized for cloud-hosted geospatial data platforms, such as the **Microsoft Planetary Computer (MPC)**, which provides access to a rich selection of satellite imagery and environmental data. The integration with the MPC reduces the need for extensive local storage or compute power, allowing users to process and analyze large datasets directly in the cloud. The workflow is designed to minimize reliance on local resources by taking full advantage of **cloud-native processing technologies**. These technologies include distributed computing frameworks that scale according to the computational needs of the data being processed. The cloud-native approach not only reduces the environmental footprint of data processing but also enables efficient collaboration, as users can easily share data and results across teams and organizations.

The use of cloud-native technologies in this workflow provides significant advantages over traditional, on-premise processing pipelines. By leveraging scalable, distributed computing frameworks, this workflow can handle the immense volume and complexity of satellite data without encountering the performance bottlenecks that often arise with local processing. **Dask**, **rioxarray**, and **xarray** are among the key libraries utilized in this pipeline to ensure efficient raster data processing. These libraries allow for the manipulation, analysis, and visualization of multi-dimensional geospatial data with ease. The combination of Dask's parallel processing capabilities with the raster data handling of `xarray` and `rioxarray` makes it possible to process large datasets quickly and efficiently, even when dealing with gigabytes or terabytes of satellite imagery.

Machine learning is a critical component of this repository, as it enables **unsupervised clustering** and data analysis, which are fundamental for extracting insights from satellite imagery. **Scikit-learn**, one of the most widely-used libraries in the machine learning ecosystem, is used extensively in this workflow to perform clustering and other forms of analysis. Unsupervised clustering allows users to group similar data points based on their spectral characteristics, which is useful for identifying patterns and trends within satellite imagery. This clustering can reveal valuable insights, such as areas of interest, vegetation types, or land cover classifications, without the need for manually labeled training data. The ability to perform such analysis on large satellite datasets opens up a wide range of possibilities for automated monitoring, environmental assessment, and change detection.

### Key Highlights of the Workflow
1. **Cloud-Native Geospatial Processing**: The pipeline is designed to leverage cloud-hosted geospatial assets, removing the need for manual downloading and preprocessing of satellite imagery. By accessing geospatial data directly from cloud platforms, the repository simplifies data access and processing workflows. This cloud-native approach ensures that data storage, computation, and access are all streamlined and efficient, reducing overhead and enabling faster analysis.
   
2. **Dynamic and Scalable**: Built with a strong emphasis on distributed computation, this workflow is inherently scalable. It can handle increasingly large datasets as data volume grows, ensuring that it can adapt to the demands of large-scale geospatial analysis. Tools like Dask and xarray enable parallelized processing, which allows the workflow to handle a variety of data sizes—from small datasets to massive satellite image archives—with ease and efficiency.

3. **End-to-End Workflow**: This repository covers the entire satellite data analysis process, providing a comprehensive end-to-end solution. From **querying data sources** to **retrieving and processing imagery**, to **deriving insights through clustering**, the workflow supports every stage of the process. It offers a complete toolkit for users looking to analyze satellite data, ensuring that each step of the pipeline is efficient, reproducible, and easy to use.

Whether you are a geospatial data scientist, an environmental researcher, a student, or someone working on any other field that requires large-scale geospatial analysis, this repository provides a solid foundation for tackling complex geospatial challenges. It is modular, flexible, and easily adaptable to various use cases, making it an ideal tool for anyone working with satellite imagery and geospatial data. By harnessing the power of modern cloud computing, distributed processing, and machine learning, this repository enables users to extract valuable insights from satellite data with minimal effort and maximum efficiency. The cloud-enabled architecture makes this repository a powerful tool for handling large-scale data analysis in a cost-effective and environmentally sustainable way.

---

## Technologies Used

This repository brings together a powerful combination of Python libraries, APIs, and geospatial data sources to facilitate geospatial data analysis, processing, and visualization. The integration of these technologies makes it possible to handle, analyze, and visualize large-scale satellite imagery data in a streamlined and efficient manner. Below is a detailed overview of the technologies and tools employed in this workflow, describing each technology's purpose and how it contributes to the overall pipeline:

### Python Libraries
The workflow is primarily built around Python's ecosystem of geospatial, scientific computing, and machine learning libraries. These libraries provide a wide range of functionality, from efficient data processing and manipulation to advanced machine learning algorithms and high-quality visualizations. Each library plays a specific role, ensuring the pipeline is both flexible and powerful. The following is a description of the key Python libraries used in the project:

- **pystac-client**:  
   The `pystac-client` library is a critical component of this workflow, enabling seamless querying and retrieval of STAC (SpatioTemporal Asset Catalog) items. By interacting with STAC-compliant APIs, the `pystac-client` library simplifies the process of searching for and filtering geospatial data based on various criteria, such as time range, location, cloud cover percentage, and other metadata attributes. This library allows users to efficiently access and retrieve satellite imagery and related datasets from different sources, ensuring that only relevant data is included in the analysis. The library provides a clean, Pythonic interface for accessing vast geospatial datasets and interacting with them in an intuitive manner.

- **planetary-computer**:  
   The `planetary-computer` library plays a crucial role in enabling access to geospatial data hosted on the **Microsoft Planetary Computer (MPC)** platform. This platform provides an extensive collection of satellite imagery and environmental data, including Landsat and Sentinel-2 datasets, which are key resources for geospatial analysis. The `planetary-computer` library is designed to authenticate users and provide direct access to MPC-hosted data, ensuring smooth integration of MPC datasets into the Python workflow. By utilizing this library, users can authenticate, retrieve, and interact with remote geospatial data without the need for manual downloading or data management. This integration makes it easy to work with satellite imagery and environmental datasets directly in Python, streamlining the data access process.

- **rioxarray/xarray**:  
   The `rioxarray` and `xarray` libraries provide essential functionality for processing and analyzing multi-dimensional geospatial data. These libraries are designed to handle raster data, which are commonly used to represent spatial information in satellite imagery. `xarray` provides data structures and tools for working with multi-dimensional arrays, which is especially useful when dealing with multi-band imagery. `rioxarray`, built on top of `xarray`, extends its capabilities by adding geospatial metadata, making it easier to work with geospatial data formats such as GeoTIFF. Together, these libraries facilitate the ingestion, manipulation, and analysis of satellite imagery, allowing users to extract, analyze, and visualize spectral bands from the data with ease. These libraries make the entire process of working with large raster datasets more efficient, enabling users to perform complex operations without running into performance bottlenecks.

- **Dask**:  
   The `Dask` library is essential for handling large datasets that exceed the memory limits of a single machine. Dask allows users to parallelize computations, enabling distributed processing of data across multiple cores or even multiple machines. This is particularly important when working with large satellite imagery datasets, which can be computationally expensive and memory-intensive. Dask is used to manage the computational workload in this workflow, ensuring that large-scale data processing tasks can be executed efficiently. By breaking down complex operations into smaller, parallel tasks, Dask enables users to perform computations that would otherwise be constrained by memory limitations. This distributed processing capability is vital for handling high-resolution satellite imagery and large environmental datasets, ensuring that the workflow can scale to meet the demands of real-world geospatial analysis.

- **shapely**:  
   The `shapely` library is a key tool for manipulating and analyzing geometric shapes and spatial data. In this workflow, `shapely` is primarily used for working with Areas of Interest (AOIs), which are specific regions of the earth's surface that are relevant to the analysis. AOIs are essential for clipping satellite imagery to a particular region, which helps focus the analysis on the areas of interest. `shapely` provides an intuitive interface for creating, manipulating, and analyzing geometric shapes, such as points, lines, and polygons, making it an invaluable tool for working with spatial boundaries. The ability to perform spatial operations such as buffering, intersection, and union makes `shapely` ideal for defining AOIs, performing spatial analysis, and ensuring that the relevant portion of the satellite imagery is processed.

- **matplotlib**:  
   The `matplotlib` library is a widely-used tool for creating high-quality visualizations in Python. In this workflow, `matplotlib` is used to generate a variety of plots, such as scatter plots, histograms, and clustering result visualizations, to help interpret the data and present the findings. Visualizations play a crucial role in geospatial analysis, as they help users better understand complex patterns, relationships, and trends in the data. Whether visualizing spectral bands, clustering results, or statistical distributions, `matplotlib` provides the flexibility and power needed to create informative and aesthetically pleasing plots. The library's wide range of customization options allows users to fine-tune the appearance of their plots, ensuring that the visualizations are clear, informative, and easy to interpret.

- **scikit-learn**:  
   The `scikit-learn` library is a cornerstone of machine learning and statistical analysis in Python. In this workflow, `scikit-learn` is used to implement unsupervised clustering algorithms, which are essential for identifying patterns and structures in the satellite imagery data. Clustering is an important technique in geospatial analysis, as it allows for the grouping of similar data points based on their spectral properties, helping to identify areas with similar land cover or environmental characteristics. `scikit-learn` provides a wide array of tools for clustering, classification, regression, and dimensionality reduction, making it an invaluable resource for data analysis. In this workflow, it is primarily used to perform clustering on satellite data, enabling the identification of meaningful patterns and relationships within the dataset. The library's implementation of popular clustering algorithms, such as KMeans, ensures that users can apply well-established techniques with ease and confidence.

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
      
Installation and Setup

This section guides you through setting up the environment, running the workflow, and understanding the Docker-based execution for processing multi-band satellite imagery efficiently.

---

1. Clone the Repository

The first step is to download this repository to your local machine. This contains all the necessary code, scripts, configuration files, and the `Dockerfile` to streamline the workflow. Follow these steps to get started:

1. Open your terminal or command prompt.
2. Run the following `git` commands:

```bash
git clone https://github.com/yourusername/multi-band-workflow.git
cd multi-band-workflow


Once inside the repository, you will find:
- **multi_band_workflow.py**: The main script to process and analyze multi-band satellite imagery.
- **Dockerfile**: Configuration file to containerize the workflow using Docker.
- **environment.yml**: Conda environment file for setting up the necessary Python dependencies.
- **outputs/**: A directory where the generated GeoTIFF files and visualizations will be saved.

---

2. Running the Workflow with Docker

To ensure a consistent and hassle-free execution environment, this workflow is containerized using **Docker**. Containerization allows you to avoid manual installation of dependencies, mismatched Python versions, or operating system issues. Follow the detailed steps below to build and execute the workflow.

#### Step 2.1: Build the Docker Image

The `Dockerfile` contains all the instructions to set up the environment, install required Python libraries, and configure the dependencies. Building the image prepares everything needed to run the workflow. Here’s how you do it:

1. Make sure Docker is installed on your system and that the Docker service is running. You can check by typing:

   ```bash
   docker --version
   ```

   If Docker is not installed, download and install it from [Docker’s official website](https://www.docker.com/get-started).

2. Build the Docker image using the following command from the root of the project (where the `Dockerfile` is located):

   ```bash
   docker build -t geospatial-clustering:latest .
   ```

   - `docker build`: Tells Docker to build an image.
   - `-t geospatial-clustering:latest`: Assigns a human-readable tag to the image, where `geospatial-clustering` is the image name, and `latest` is the tag version.
   - `.`: Specifies the current directory as the build context.

   This step might take a few minutes, as Docker downloads the necessary base image (a lightweight Python environment) and installs all required Python libraries, including:
   - `pystac-client`
   - `rioxarray`
   - `Dask`
   - `scikit-learn`
   - `matplotlib`
   - And more…

   You will see messages showing the installation progress, including confirmation that the image has been successfully built.

3. Verify that the Docker image has been created by running:

   ```bash
   docker images
   ```

   You should see `geospatial-clustering` listed in the output with the tag `latest` and its size.

#### Step 2.2: Run the Docker Container

Once the Docker image is successfully built, you can start a container to execute the workflow. The container acts as a lightweight virtual machine that runs the script in a fully isolated environment.

Use the following command to start the container:

```bash
docker run --rm -v $(pwd):/app geospatial-clustering:latest
```

Here’s a detailed breakdown of the command:
- **`docker run`**: Launches a new container from the specified image.
- **`--rm`**: Automatically removes the container after it finishes running. This avoids cluttering your system with unused containers.
- **`-v $(pwd):/app`**: Mounts your local directory (the project folder) to the `/app` directory inside the container. This ensures:
  - Input files are accessible to the container.
  - All outputs (e.g., processed GeoTIFF files and plots) generated inside the container are saved back to your local directory.
- **`geospatial-clustering:latest`**: Specifies the Docker image to run.

Once executed, the workflow script (`multi_band_workflow.py`) will automatically run inside the container. You will see logs showing the progress of each step, such as querying STAC items, processing bands, mosaicking imagery, clustering pixels, and generating outputs.

---

### 3. Modify Workflow Parameters

Before running the workflow, it’s important to customize the parameters to suit your specific use case. The workflow accepts inputs such as the **Area of Interest (AOI)**, **time range**, and other filters (e.g., cloud cover percentage).

1. Open the `multi_band_workflow.py` script in your preferred text editor (e.g., VS Code, PyCharm, Nano).
2. Locate the section where key parameters are defined, for example:

   ```python
   AOI = {"type": "Polygon", "coordinates": [[ ... ]]}  # Define the geographic extent
   TIME_RANGE = ["2023-01-01", "2023-03-31"]  # Time range for imagery
   CLOUD_COVER = 10  # Maximum cloud cover percentage
   ```

3. Update the following parameters as needed:
   - **AOI**: Replace with your area of interest in GeoJSON format.
   - **TIME_RANGE**: Set the start and end dates to filter imagery.
   - **CLOUD_COVER**: Adjust the threshold to control cloud cover in the selected imagery.

Save the changes and rerun the workflow to process data with your updated parameters.

---

### 4. Execute the Workflow

You can execute the workflow in two ways:

#### Option 1: Run Directly with Conda

If you prefer running the workflow outside of Docker, ensure the Conda environment is set up:

1. Activate the environment:

   ```bash
   conda activate geospatial-clustering
   ```

2. Run the main script:

   ```bash
   python multi_band_workflow.py
   ```

This method assumes all dependencies are already installed via the `environment.yml` file.

#### Option 2: Run with Docker (Recommended)

To execute the workflow using Docker, simply repeat the container run command:

```bash
docker run --rm -v $(pwd):/app geospatial-clustering:latest
```

The container will process the workflow and save the results locally.

---

### 5. Output and Results

After the workflow completes successfully, you will find the following outputs in the `outputs/` directory:
- **Processed Multi-Band GeoTIFFs**: These files contain mosaicked and clipped imagery for your AOI.
- **Clustered Images**: Visualization results showing clusters of pixels based on spectral similarity.
- **Spectral Signatures**: Plots or data summarizing the average spectral signatures for each cluster.

If you mounted your current directory using `-v $(pwd):/app`, the outputs will appear directly in your local `outputs/` folder.

---

### 6. Verify Execution

To confirm everything ran successfully:
1. Check for log messages in the terminal output. Each step (querying data, mosaicking, clustering, and saving results) is logged.
2. Inspect the `outputs/` directory to ensure all expected files are present.
3. Review any generated plots or images for accuracy.

---

### Notes and Troubleshooting

- If you encounter **Docker permission errors** on Linux, prefix commands with `sudo`, or add your user to the Docker group.
- For **memory issues**, adjust Docker resource limits (CPU/RAM) in Docker Desktop or the command-line settings.
- If outputs are missing, verify your AOI, time range, and cloud cover settings in the script.
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
A silhouette plot is a powerful tool used in cluster analysis to assess the quality of clustering results. It provides a graphical representation of how well each data point fits within its assigned cluster relative to other clusters. Each bar in the silhouette plot corresponds to a data point, and the length of the bar represents how well that point fits within its assigned cluster. A silhouette value close to +1 indicates that the data point is well-clustered, while a silhouette value close to -1 suggests that the data point may have been assigned to the wrong cluster. This method is particularly useful for identifying clusters with high internal cohesion and low separation between them. By examining the silhouette score for each point and cluster, you can make informed decisions about the optimal number of clusters and the overall quality of the clustering.

- **High silhouette value (close to 1)**: Indicates that a data point is well-matched to its own cluster and poorly matched to neighboring clusters. This suggests a good clustering assignment and well-separated clusters.
- **Negative silhouette value**: Suggests that the data point is likely misclassified and would fit better in a neighboring cluster. This could indicate a need for revisiting the clustering parameters or the data's inherent structure.

#### Code
```python
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

# Reshape valid data for clustering analysis
valid_data_reshaped = valid_data.reshape(-1, 1)

# Predict cluster labels based on reshaped data
cluster_labels = kmeans.predict(valid_data_reshaped)

# Calculate the silhouette values for each data point in the dataset
silhouette_vals = silhouette_samples(valid_data_reshaped, cluster_labels)

# Calculate the average silhouette score for the entire clustering
silhouette_avg = silhouette_score(valid_data_reshaped, cluster_labels)

# Output the average silhouette score for review
print(f"Average Silhouette Score: {silhouette_avg:.3f}")

# Set up the figure for plotting the silhouette chart
plt.figure(figsize=(10, 7))

# Initialize the lower limit for the Y-axis
y_lower = 10

# Loop through each cluster to plot the silhouette values
for i in range(n_clusters):
    # Select silhouette values for the points in the current cluster
    ith_cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    
    # Sort the silhouette values to plot them in an ordered fashion
    ith_cluster_silhouette_vals.sort()

    # Determine the number of data points in the current cluster
    size_cluster_i = ith_cluster_silhouette_vals.shape[0]

    # Set the upper limit for the current cluster
    y_upper = y_lower + size_cluster_i

    # Fill the area between y_lower and y_upper with the silhouette values for this cluster
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals, alpha=0.7, label=f'Cluster {i}')
    
    # Update the lower limit for the next cluster
    y_lower = y_upper + 10

# Plot the average silhouette score as a vertical dashed line
plt.axvline(x=silhouette_avg, color='red', linestyle='--', label='Average Silhouette Score')

# Add title, axis labels, and legend to the plot
plt.title("Silhouette Plot for KMeans Clustering", fontsize=14)
plt.xlabel("Silhouette Coefficient Values", fontsize=12)
plt.ylabel("Cluster Labels", fontsize=12)
plt.legend(loc='upper right')

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

```

---

### Elbow Method Plot
#### Description
The elbow method is a common technique used to determine the optimal number of clusters (denoted as k) in a clustering analysis. The basic idea behind the elbow method is to plot the "inertia" (within-cluster sum of squares) for different values of k and look for a point on the plot where the inertia begins to decrease at a slower rate. This point is often referred to as the "elbow" and is considered the optimal number of clusters. The inertia measures how well the data points fit within their clusters. As the number of clusters increases, inertia tends to decrease, but after a certain number of clusters, the decrease becomes less significant. The elbow represents the point where adding more clusters doesn't result in a significant improvement in inertia.

#### Code
```python
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

# Reshape valid data for clustering analysis
valid_data_reshaped = valid_data.reshape(-1, 3)

# Define the range of clusters to test
cluster_range = range(1, 15)

# Initialize a list to store inertia values for different numbers of clusters
inertia_values = []

# Loop through different values of k (number of clusters)
for n_clusters in cluster_range:
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    kmeans.fit(valid_data_reshaped)
    inertia_values.append(kmeans.inertia_)

# Plot the inertia values against the number of clusters
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, inertia_values, marker='o', linestyle='--', color='b')

# Add title, labels, and grid to the plot
plt.title("Elbow Method for Optimal Clusters", fontsize=14)
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
plt.xticks(cluster_range)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

```

---

### Cluster Size Distribution
#### Description
This plot is a bar chart that displays the distribution of data points across the different clusters. It helps assess the balance of the clustering, as some clusters may have significantly more data points than others, which could indicate imbalanced clustering. A uniform distribution of data points across clusters typically suggests a well-performing clustering algorithm. However, clusters with significantly different sizes may suggest the need for parameter adjustments.

#### Code
```python
import matplotlib.pyplot as plt
import numpy as np

# Get the unique labels (clusters) and count the number of data points in each cluster
unique_labels, counts = np.unique(cluster_labels, return_counts=True)

# Plot the distribution of cluster sizes as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(unique_labels, counts, color='skyblue', edgecolor='black')

# Add title and axis labels
plt.title("Cluster Size Distribution", fontsize=14)
plt.xlabel("Cluster Label", fontsize=12)
plt.ylabel("Number of Pixels", fontsize=12)

# Customize the ticks and grid
plt.xticks(unique_labels)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()

```

---

### 2D Scatter Plot of Clusters
#### Description
A 2D scatter plot is a graphical representation that displays data points in a two-dimensional space, where each point's position is determined by two feature values. In the case of clustering, the points are colored by their respective cluster label to visualize the spatial separation between clusters. This plot helps to see how well the data points are separated according to their cluster assignments.

#### Code
```python
import matplotlib.pyplot as plt

# Define the red and NIR band names (assuming these are already extracted)
red_band = 'red_band_name'
nir_band = 'nir_band_name'

# Flatten the data values for both the red and NIR bands
red_values = final_mosaic[red_band].values.flatten()
nir_values = final_mosaic[nir_band].values.flatten()

# Apply a mask to remove invalid values (e.g., NaN values or non-positive values)
valid_mask = (~np.isnan(red_values)) & (~np.isnan(nir_values)) & (red_values > 0) & (nir_values > 0)
red_values = red_values[valid_mask]
nir_values = nir_values[valid_mask]

# Create a scatter plot of the valid red and NIR values
plt.figure(figsize=(10, 8))
plt.scatter(red_values, nir_values, s=1, c='blue', alpha=0.5)

# Add title, axis labels, and grid
plt.title("2D Scatter Plot of Red vs NIR Values", fontsize=14)
plt.xlabel("Red Band Reflectance", fontsize=12)
plt.ylabel("NIR Band Reflectance", fontsize=12)
plt.grid(alpha=0.3)

# Adjust the layout to fit the plot
plt.tight_layout()

# Display the plot
plt.show()

```


## References and Credits

This project was made possible with the use of various open-source tools and technologies. Special thanks to the following contributors:

- **ChatGPT**: Provided guidance and assistance throughout the development of this workflow, including the generation of code snippets, explanations, and structuring of the README documentation.
  
- **Microsoft Copilot**: Assisted with code completion, suggestions, and efficient writing of scripts during the development process, helping to streamline the implementation of geospatial data processing tasks.

### Libraries and Tools
- **Python**: A general-purpose programming language used throughout this project.
- **rioxarray**: For processing and handling raster data.
- **Dask**: Used for parallel and distributed computing.
- **scikit-learn**: Used for performing unsupervised clustering.
- **Docker**: Containerized the workflow to ensure a consistent environment across different platforms.

### Data Sources
- **Microsoft Planetary Computer**: Provided access to satellite imagery for processing.

