from ultralytics import YOLO
import cv2 as cv
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

# Load the model
model = YOLO('yolov10_v1.0/best (3).pt')

# Read the image
img = cv.imread('base_file/valid/images/81_jpg.rf.bc6161b584b180ffbbb3829a0599e23f.jpg')

# Advanced pre-processing
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(gray)
blurred = cv.GaussianBlur(equalized, (5, 5), 0)
edges = cv.Canny(blurred, 50, 150)
binary = cv.adaptiveThreshold(equalized, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

# Morphological operations
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
morph = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

# Run inference
results = model(img, stream=True)

# Metadata for selenographic conversion
scale = 0.1  # 1 pixel = 100 meters
center_lat, center_lon = 0.0, 0.0

# Function to convert pixel coordinates to selenographic coordinates
def pixel_to_selenographic(x, y, center_lat, center_lon, scale):
    lon = center_lon + (x - img.shape[1] / 2) * scale / 1000.0
    lat = center_lat - (y - img.shape[0] / 2) * scale / 1000.0
    return lat, lon
def calculate_diameter(width, height, scale):
    diameter = (width**2 + height**2) / 2
    diameter = (diameter**0.5) * scale
    return diameter

# List to hold all polygons and their properties
crater_data = []

# Initialize crater ID
crater_id = 1

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Create a mask for the current bounding box
        mask = np.zeros_like(morph)
        mask[y1:y2, x1:x2] = morph[y1:y2, x1:x2]

        # Find contours in the mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Simplify the contour
            epsilon = 0.01 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)

            # Ensure the contour has at least 4 points
            if len(approx) >= 4:
                # Convert contour to polygon and selenographic coordinates
                selenographic_coords = [pixel_to_selenographic(pt[0][0], pt[0][1], center_lat, center_lon, scale) for pt in approx]
                polygon = Polygon(selenographic_coords)
                crater_data.append({'geometry': polygon, 'crater_id': crater_id})

                # Calculate the diameter (using the bounding box of the contour)
                rect = cv.minAreaRect(approx)
                width, height = rect[1]
                # diameter = max(width, height) * scale  /1000 # Convert to km
                diameter = calculate_diameter(width,height,scale) / 1000
                crater_data[-1]['diameter_km'] = diameter

                # Calculate and print the center
                M = cv.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center_lat_lon = pixel_to_selenographic(cx, cy, center_lat, center_lon, scale)
                    print(f"Crater ID: {crater_id}, Center: Latitude: {center_lat_lon[0]:.6f}, Longitude: {center_lat_lon[1]:.6f}")

                # Print diameter
                print(f"Estimated Diameter (km): {diameter:.9f}")

                # Increment crater ID
                crater_id += 1

# Save detections
cv.imwrite('detections.jpg', img)

# Create a GeoDataFrame
# gdf = gpd.GeoDataFrame(crater_data)
#
# # Save to shapefile
# gdf.to_file("craters.shp")
