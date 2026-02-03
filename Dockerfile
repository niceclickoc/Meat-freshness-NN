FROM python:3.10-slim

# Install system dependencies
# libgl1, libglib2.0-0 for OpenCV
# qt dependencies for PyQt5
# libasound2 for some audio/qt dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libqt5gui5 \
    libqt5widgets5 \
    libxcb-xinerama0 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies using the Pi-optimized list
RUN pip install --no-cache-dir -r requirements_pi.txt

# Set the display environment variable (can be overridden at runtime)
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Run the application
CMD ["python", "src/main.py"]
