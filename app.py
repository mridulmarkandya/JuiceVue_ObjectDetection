import streamlit as st
from ultralytics import YOLO
from collections import Counter
from PIL import Image

class YOLO:
    """
    This class loads and performs inference with a YOLOv8 model.
    """
    def __init__(self, weights_path):
        self.model = torch.hub.load('ultralytics/yolov8', 'yolov8n', weights='/best.pt')

    def __call__(self, images):
        # Assuming images is a list of PIL Images
        results = self.model(images)
        return results

def count_objects(results):
  total_objects = 0
  for result in results:
    total_objects += len(result.boxes)  # Assuming boxes attribute holds detections
  return total_objects


def main():
  st.title("Object Detection and Counting App")
  st.write("Upload an image and see the detected objects!")
  uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load YOLO model
    model = YOLO('/best.pt')  # Replace with actual path

    images = [image]  # Convert to list for model inference

    # Run inference with YOLO
    results = model(images)

    # Count objects
    total_objects = count_objects(results)

    st.success(f'Total objects detected: {total_objects}')


if __name__ == "__main__":
  main()
