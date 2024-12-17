# Dockerfile for Multi-Band Satellite Imagery Workflow

# 1. Use an official Conda base image with Python 3.9
FROM continuumio/miniconda3:4.12.0

# 2. Set the working directory
WORKDIR /app

# 3. Copy the environment.yml file to the container
COPY environment.yml /app/environment.yml

# 4. Install Conda environment and dependencies
RUN conda env create -f environment.yml && conda clean -a

# 5. Add Conda environment to PATH
ENV PATH /opt/conda/envs/geospatial-clustering/bin:$PATH

# 6. Copy all project files into the container
COPY . /app

# 7. Expose port for external communication (optional, if needed)
EXPOSE 8888

# 8. Define the default command to run the script
CMD ["python", "multi_band_workflow.py"]
