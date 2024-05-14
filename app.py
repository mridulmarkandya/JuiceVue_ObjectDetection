import streamlit as st
import PIL
from PIL import Image, ImageDraw
from ultralytics import YOLO
from collections import Counter
import torch
import cv2
import numpy as np



def load_image(image_file):
    if image_file is not None:
        image = Image.open(image_file).convert('RGB')
        return image
    return None

def add_text_to_image(image_array, text, position=(10, 10), font_size=50, color='red'):
    
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, fill=color)  #, font=font if you loaded a font
    return image


def predict_image(image):
    
    #home = r'/Users/mridulmarkandya/Downloads'
    model_path = rf'best.pt'

    st.title("Object Detection")
    st.caption('Updload a photo by selecting :blue[Browse files]')
    st.caption('Then click the :blue[Detect Objects] button and check the result.')    
    
    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

        if st.button('Detect Objects'):
            res = model.predict(image, line_width=1, show_labels=True, show_conf=False)
            boxes = res[0].boxes
            class_names = res[0].names
            class_ids_list = res[0].boxes.cls.int().tolist()
            class_id_counts = Counter(class_ids_list)
            class_name_counts = {class_names.get(id, f"Class_{id}"): count for id, count in class_id_counts.items()}
            count_list = ''
            for class_name, count in class_name_counts.items():
                # Properly use append to add formatted strings to the list
                count_list=count_list+(f"{class_name:<17}: {count} \n")
            
            res_plotted = res[0].plot(labels=True, line_width=1)[:, :, ::-1]
    
            res_plotted = add_text_to_image(res_plotted.copy(), f"Total detections : {len(boxes)}\n{count_list}", position=(10, 10), font_size=30, color='red')
        
            st.image(res_plotted,
                     caption='Detected Image'                
                     )
            try:
                st.write(f"Total detections : {len(boxes)}")
                for class_name, count in class_name_counts.items():
                    # Properly use append to add formatted strings to the list
                    st.write(f"{class_name} : {count}")
    
    
            except Exception:
                st.write("No image is uploaded yet!")
    


def main():
    st.title("Object Detection and Counting App")
    st.write("Upload an image and see the detected objects!")
    uploaded_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        # Load and display the original image
        image = load_image(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if image:
            predict_image(image)
                

if __name__ == "__main__":
    main()
