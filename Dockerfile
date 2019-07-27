FROM nvcr.io/nvidia/pytorch:19.06-py3

RUN apt-get update -y && ACCEPT_EULA=Y apt-get install -y \
    apt-transport-https \
    python3-dev \
    build-essential \
    python-numpy \
    python-scipy \
    python-nose \
    python-h5py \
    python-skimage \
    python-matplotlib \
    python-sympy \
    && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/

# pip installs
RUN pip --no-cache-dir install --upgrade pip && \
    pip --no-cache-dir install --upgrade ipython && \
    pip --no-cache-dir install \
        simplejson \
        jsonschema \
        flask \
        flask-restplus==0.10.1 \
        urllib3==1.24.0 \
        pandas \
        pandasql \
        sklearn \
        setuptools \
        xgboost \
        pyodbc>=4.0.22 \
        matplotlib \
        requests \
        unidecode \
        asyncio \
        numpy \
        scipy \
        torch \
        keras \
        jupyter \
        torchbiggraph \
        fastai

# APP to run on port 8811
EXPOSE 8811
