# ASL YOLO - American Sign Language Detection

A deep learning project for detecting and recognizing American Sign Language (ASL) using YOLOv8 object detection.

## Project Overview

This project uses YOLOv8 for real-time detection of American Sign Language (ASL) gestures and signs. The detected text is then converted to speech, enabling seamless communication from sign language to spoken words. The model is trained to identify various ASL signs with high accuracy.

## Project Structure

```
.
├── app.py              # Main application file
├── best.pt            # Pre-trained YOLO model
├── requirements.txt   # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ASL_YOLO
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python app.py
```

## Model

- **Model Type**: YOLOv8
- **Pre-trained Weight**: `best.pt`
- **Task**: Object Detection for ASL Recognition

## Features

- Real-time ASL sign detection and recognition
- Automatic text-to-speech conversion (Convert detected text to audio)
- High accuracy sign detection
- Easy-to-use interface through `app.py`
- Accessible communication tool for deaf and hard-of-hearing users

## Dependencies

All required packages are listed in `requirements.txt`. Key dependencies include:
- ultralytics (YOLOv8)
- OpenCV (cv2)
- PyTorch

## Getting Started

1. Ensure all dependencies are installed
2. Run `python app.py` to start the application
3. The model will use the pre-trained `best.pt` weights
4. Allow access to your camera when prompted
5. Perform ASL signs in front of the camera
6. The detected text will be automatically converted to speech

## Notes

- The model file `best.pt` is required to run the application
- Ensure your camera/input device is properly connected before running the app
- The project requires adequate GPU resources for optimal performance

## License

Yassin Ahmed


