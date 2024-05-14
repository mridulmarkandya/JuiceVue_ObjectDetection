import streamlit as st
import PIL
from PIL import Image, ImageDraw
from ultralytics import YOLO
from collections import Counter
import torch
import cv2
import numpy as np


def load_image(image_file):
    """Loads an image from the uploaded file."""
    if image_file is not None:
        try:
            image = Image.open(image_file).convert('RGB')
            return image
        except Exception as e:
            st.error("Error loading image:", e)
            return None
    return None


def add_text_to_image(image_array, text, position=(10, 10), font_size=30, color='red'):
    """Adds text to an image array."""
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, fill=color)  # No font argument needed for basic text
    return np.array(image)  # Convert back to NumPy array


def predict_image(image):
    """Predicts objects in an image and displays results."""
    model_path = 'best.pt'  # Assuming 'best.pt' is in the same directory

    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        return

    try:
        res = model.predict(image, line_width=1, show_labels=True, show_conf=False)
        boxes = res[0].boxes.cpu()  # Move boxes to CPU for NumPy conversion
        class_names = res[0].names
        class_ids_list = res[0].boxes[:, -1].int().tolist()  # Get class IDs
        class_id_counts = Counter(class_ids_list)
        class_name_counts = {class_names.get(id, f"Class_{id}"): count for id, count in class_id_counts.items()}

        count_list = ""
        for class_name, count in class_name_counts.items():
            # Efficiently build count list
            count_list += f"{class_name:<17}: {count} \n"

        res_plotted = res[0].plot(labels=True, line_width=1)[:, :, ::-1]
        res_plotted = add_text_to_image(
            res_plotted.copy(),
            f"Total detections : {len(boxes)}\n{count_list}",
            position=(10, 10)
        )

        st.image(res_plotted, caption='Detected Image')

        st.write(f"Total detections : {len(boxes)}")
        for class_name, count in class_name_counts.items():
            st.write(f"{class_name} : {count}")

    except Exception as e:
        st.error("Error in prediction:", e)


def main():
    st.title("Object Detection and Counting App")
    st.write("Upload an image and see the detected objects!")

    uploaded_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = load_image(uploaded_image)
        if image:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            predict_image(image)
        else:
            st.warning("Failed to load image. Please try again.")


if __name__ == "__main__":
    main()
