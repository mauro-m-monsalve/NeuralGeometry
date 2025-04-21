FROM jupyter/scipy-notebook:latest

# Install required packages via conda
RUN conda install -y -c conda-forge \
    python=3.10 \
    numpy pandas scipy matplotlib seaborn xarray \
    plotly scikit-learn statsmodels tqdm \
    ipykernel jupyterlab ipywidgets jupyterlab_widgets \
    pot \
    && conda clean -afy

# Optional: memory environment variable for local use (Binder ignores this)
ENV MEM_LIMIT=8G
ENV PYTHONPATH=/home/jovyan

# Set working directory
WORKDIR /home/jovyan

# Create a directory for figures
RUN mkdir -p figures