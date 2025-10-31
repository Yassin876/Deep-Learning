# Fake and Real Image Detection

This project uses a deep learning model to distinguish between fake and real images via a web interface built with Streamlit.

## Project Structure
```
fake_and_real_image/
├── app.py
├── fake_img_model.h5
├── fake_img.ipynb
├── real_and_fake_image_data/
│   ├── train/
│   │   ├── FAKE/
│   │   └── REAL/
│   └── test/
│       ├── FAKE/
│       └── REAL/
```

## Description of Files and Folders
- **app.py**: The main Streamlit application. Handles the user interface, loads the model, and performs image analysis.
- **fake_img_model.h5**: The trained Keras model for classifying images as FAKE or REAL.
- **fake_img.ipynb**: Jupyter notebook containing the model training code, experiments, and steps of the training process.
- **real_and_fake_image_data/**: Main folder containing the image dataset used for training and testing.
  - **train/**: Training data.
    - **FAKE/**: Fake images used for model training (typically grayscale 150x150px images).
    - **REAL/**: Real images used for training.
  - **test/**: Test data for evaluating model accuracy.
    - **FAKE/**: Fake images for validating the model after training.
    - **REAL/**: Real images for validating the model after training.

## Features
- Upload an image (JPG or PNG)
- Predicts whether the image is FAKE or REAL using a pre-trained Keras model
- Shows confidence of prediction

## Requirements
- Python 3.7+
- streamlit
- tensorflow
- numpy
- pillow

You can install the required dependencies using:
```bash
pip install streamlit tensorflow numpy pillow
```

## Files and Directories
- `app.py`: Main Streamlit application
- `fake_img_model.h5`: Pre-trained Keras model for image classification
- `real_and_fake_image_data/`: Dataset with FAKE and REAL images (used for training/testing)
- `fake_img.ipynb`: Training notebook

## How to Run
1. Ensure `fake_img_model.h5` is in the project directory.
2. Install requirements (see above).
3. In the terminal, run:
   ```bash
   streamlit run app.py
   ```
4. Open the shown URL in your browser. 
5. Upload an image and click "Predict" to see the result.

## Notes
- The model expects grayscale images sized 150x150 pixels, but the app will automatically preprocess uploads.
- Training data and model training are not covered in this README; see `fake_img.ipynb` for details.

## Author
- Yassin Ahmed