import streamlit as st
import cv2
import os
import zipfile
import base64
import shutil
import torch

car_plate_model = torch.hub.load('ultralytics/yolov5', 'custom', path='car.pt')
face_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path='face.pt')

def process_image(image_path, use_case):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Perform anonymization based on the selected use case
    if use_case == 'License Plate Anonymization':
        processed_image = anonymize_license_plate(image)
    elif use_case == 'Facial Data Anonymization':
        processed_image = anonymize_facial_data(image)

    return processed_image

def anonymize_license_plate(image):
    # Add your license plate anonymization logic here
    # Use your pre-trained model
    results = car_plate_model(image)
    df = results.pandas().xyxy[0]
    if df.empty:
        return image
    x = int(results.pandas().xyxy[0]._get_value(0, 'xmin', takeable=False))
    y = int(results.pandas().xyxy[0]._get_value(0, 'ymin', takeable=False))
    x2 = int(results.pandas().xyxy[0]._get_value(0, 'xmax', takeable=False))
    y2 = int(results.pandas().xyxy[0]._get_value(0, 'ymax', takeable=False))
    w = x2 - x
    h = y2 - y
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
    roi = image[y:y+h, x:x+w]    # applying a gaussian blur over this new rectangle area
    roi = cv2.GaussianBlur(roi, (47, 47), 0)    # impose this blurred image on original image to get final image
    image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    return image

def anonymize_facial_data(image):
    results = face_detect_model(image)
    df = results.pandas().xyxy[0]
    if df.empty:
        return image
    x = int(results.pandas().xyxy[0]._get_value(0, 'xmin', takeable=False))
    y = int(results.pandas().xyxy[0]._get_value(0, 'ymin', takeable=False))
    x2 = int(results.pandas().xyxy[0]._get_value(0, 'xmax', takeable=False))
    y2 = int(results.pandas().xyxy[0]._get_value(0, 'ymax', takeable=False))
    w = x2 - x
    h = y2 - y
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
    roi = image[y:y+h, x:x+w]    # applying a gaussian blur over this new rectangle area
    roi = cv2.GaussianBlur(roi, (47, 47), 0)    # impose this blurred image on original image to get final image
    image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    return image
    return image

def save_processed_files(files, use_case):
    output_dir = "processed_images"
    os.makedirs(output_dir, exist_ok=True)

    for file_path in files:
        image = process_image(file_path, use_case)
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        cv2.imwrite(output_path, image)

    return output_dir

def main():
    st.title('Image Anonymization Platform')

    st.markdown("## Upload Dataset (Single or bulk image)")
    uploaded_file = st.file_uploader("Choose a zip file or an individual image...", type=["zip", "jpg", "png"])

    use_case = st.selectbox('Choose Use Case:', ['License Plate Anonymization', 'Facial Data Anonymization'])
    st.write(f"You have selected: {use_case}")

    if uploaded_file is not None:
        temp_dir = "temp_folder"
        os.makedirs(temp_dir, exist_ok=True)
        files_to_process = []

        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                for root, _, files in os.walk(temp_dir):
                    for file_name in files:
                        if file_name.endswith((".jpg", ".png")):
                            files_to_process.append(os.path.join(root, file_name))
        else:
            temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
            with open(temp_image_path, "wb") as file:
                file.write(uploaded_file.read())
            files_to_process.append(temp_image_path)

        if st.button("Load Data"):
            processed_folder = save_processed_files(files_to_process, use_case)
            shutil.make_archive('processed_images', 'zip', processed_folder)

            # Provide a download link
            with open('processed_images.zip', 'rb') as file:
                href = f'<a download="processed_images.zip" href="data:file/zip;base64,{base64.b64encode(file.read()).decode()}">Click here to download the processed images</a>'
                st.markdown(href, unsafe_allow_html=True)

            # Clean up temporary directories
            shutil.rmtree(temp_dir)
            shutil.rmtree(processed_folder)

if __name__ == "__main__":
    main()
