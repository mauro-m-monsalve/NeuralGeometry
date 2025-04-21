FROM jupyter/scipy-notebook:latest

# Install required packages with conda (reflecting environment.yml)
RUN conda install -y -c conda-forge \
    python=3.10 \
    numpy pandas scipy matplotlib plotly \
    scikit-learn statsmodels tqdm \
    ipykernel jupyter jupyterlab notebook \
    nodejs ipywidgets jupyterlab_widgets jupyter_contrib_nbextensions \
    xarray seaborn pot \
    && conda clean -afy

# Set environment variables
ENV MEM_LIMIT=8G
ENV PYTHONPATH=/home/jovyan

# Copy the full repo into the container (Binder does this by default,
# this line is a reminder if using Docker outside Binder)
# COPY . /home/jovyan/

# Set working directory
WORKDIR /home/jovyan

# Optional: pre-cache notebook outputs
RUN mkdir -p figures