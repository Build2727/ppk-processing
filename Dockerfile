# ---- Base image
FROM python:3.11-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# System deps for building RTKLIB and running Streamlit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Build RTKLIB rnx2rtkp inside the container
RUN git clone https://github.com/tomojitakasu/RTKLIB.git /tmp/RTKLIB && \
    make -C /tmp/RTKLIB/app/rnx2rtkp/gcc -j && \
    mkdir -p /app/bin && \
    cp /tmp/RTKLIB/app/rnx2rtkp/gcc/rnx2rtkp /app/bin/ && \
    chmod +x /app/bin/rnx2rtkp && \
    rm -rf /tmp/RTKLIB

# Copy the rest of your app
COPY . /app

# Streamlit config for Cloud
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Launch
CMD ["streamlit", "run", "app.py"]
