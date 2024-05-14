import streamlit as st
import cv2 
from ultralytics import YOLO
from collections import Counter
from PIL import Image
import torch

class YOLO:
    def __init__(self, weights_path):
        self.model = torch.load(weights_path)

    def __call__(self, images):
        results = self.model(images)
        return results

def count_objects(results):
  total_objects = 0
  for result in results:
    total_objects += len(result.boxes)
  return total_objects


def main():
  st.title("Object Detection and Counting App")
  st.write("Upload an image and see the detected objects!")
  uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    
    model = YOLO('./best.pt')

    images = [image]

    results = model(images)

    total_objects = count_objects(results)

    st.success(f'Total objects detected: {total_objects}')


if __name__ == "__main__":
  main()
