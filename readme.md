📱 Mobile Phone Detection with YOLOv11 🚀
Welcome to an awesome project that harnesses the power of YOLOv11 to detect mobile phones in images and real-time video! 📸 This project uses a Kaggle dataset, converts XML annotations to YOLO format, trains a model, and even lets you detect phones live using your webcam. Whether you're new to AI or a pro, this guide is your ticket to a fun and rewarding experience! 😎

🎯 Project Overview
This project trains a YOLOv11 model to detect mobile phones with high accuracy. We start by grabbing the Mobile Phone Image Dataset from Kaggle, splitting it into training and validation sets, converting annotations, and training the model using the Ultralytics library. Plus, we’ll show you how to use the trained model for real-time detection with your webcam! 🎥
What you'll get:

A trained YOLOv11 model (best.pt).
A pipeline to process and train on custom datasets.
Real-time mobile phone detection using your webcam! 📹
A super fun learning experience! 🌟


🗂️ Dataset
We’re using the 🔥 Mobile Phone Image Dataset from Kaggle. It’s loaded with images of mobile phones (smartphones and feature phones) and XML annotations for bounding boxes.

Class: mobile_phone 📱
Split: 80% training, 20% validation
Format: Images (.jpg/.png) + XML annotations (converted to YOLO TXT)


🛠️ Requirements
To jump into this adventure, you’ll need:

Python 3.7+ 🐍
Libraries: ultralytics, kaggle, and Python basics (os, shutil, random, xml.etree.ElementTree)
Environment: Google Colab for training (free tier works, GPU recommended for speed ⚡)
Webcam: For real-time detection (local setup or Colab with webcam support)
Kaggle API Token: Download kaggle.json from your Kaggle account settings.

Install the magic:
pip install kaggle ultralytics --quiet


🚀 Setup Instructions
Ready to roll? Follow these steps in Google Colab to train the model:

Download the Dataset 📥:

Upload your kaggle.json when prompted in Colab.
Run these commands to download and unzip the dataset:kaggle datasets download -d dataclusterlabs/mobile-phone-image-dataset
unzip mobile-phone-image-dataset.zip -d /content/mobile_dataset




Organize the Data 🗂️:

The script creates train and val folders with images and labels subdirectories.
Images are split (80% train, 20% val), and XML annotations are copied over.


Convert Annotations 🔄:

XML annotations are converted to YOLO TXT format using a custom function.
Converted files are saved in train/labels and val/labels.


Create the YAML File 📝:

A data.yaml file is generated for YOLO:path: /content/mobile_dataset
train: /content/mobile_dataset/train/images
val: /content/mobile_dataset/val/images
nc: 1
names:
  0: mobile_phone




Train the Model 🏋️:

The script loads a pretrained yolo11s.pt model and trains for 15 epochs on CPU (use GPU for speed!).
The best model is saved as best.pt.




📂 Directory Structure
After setup, your project directory will look like this:
/content/mobile_dataset/
├── train/                # Training data
│   ├── images/           # Training images
│   └── labels/           # Training annotations (TXT)
├── val/                  # Validation data
│   ├── images/           # Validation images
│   └── labels/           # Validation annotations (TXT)
├── data.yaml             # YOLO configuration
├── Mobile_image/         # Original images
│   └── Mobile_image/
├── Annotations/          # Original XML annotations
│   └── Annotations/
└── yolo_train_v11/log/weights/best.pt  # Trained model 🎉


🧠 Training the Model
Here’s how the model is trained:
from ultralytics import YOLO

model = YOLO('yolo11s.pt')  # Load pretrained YOLOv11 small
results = model.train(
    data='/content/mobile_dataset/data.yaml',
    epochs=15,
    project='yolo_train_v11',
    name='log',
    device='cpu',  # Switch to '0' for GPU if available
    imgsz=640,     # Image size
    batch=16       # Batch size
)


Output: Training logs, metrics (mAP, precision, recall), and plots in yolo_train_v11/log/.
Model: Best weights saved as best.pt in yolo_train_v11/log/weights/.
Training Time: ~30-60 minutes on CPU, faster on GPU! 🚀


🎥 Real-Time Webcam Detection
Once your model is trained, you can use it to detect mobile phones in real-time using your webcam! Here’s the code snippet and what it does:
from ultralytics import YOLO

model = YOLO("best.pt")  # Load the trained model
model.predict(source=0, show=True, conf=0.6)  # Run real-time detection

Explanation of Webcam Detection Code 📹

from ultralytics import YOLO: Imports the YOLO class from the Ultralytics library, which provides tools for loading and running YOLO models.
model = YOLO("best.pt"): Loads your trained model weights (best.pt) from the training output (located in yolo_train_v11/log/weights/best.pt or specify the full path if running locally).
model.predict(source=0, show=True, conf=0.6):
source=0: Uses your webcam as the input (0 is the default webcam ID in OpenCV).
show=True: Displays the live video feed with bounding boxes drawn around detected mobile phones.
conf=0.6: Sets a confidence threshold of 0.6, meaning only detections with 60%+ confidence are shown (adjust this to balance sensitivity and accuracy).



How it works:

The model processes each frame from the webcam in real-time.
It draws bounding boxes around detected mobile phones, labeled with the class (mobile_phone) and confidence score.
The video feed is displayed in a window until you stop the script (e.g., press Ctrl+C or close the window).

Important Notes:

Running in Colab: Google Colab doesn’t support webcam access directly due to its cloud-based nature. To use this code, run it on a local machine with a webcam, Python, and Ultralytics installed (pip install ultralytics opencv-python).
Local Setup: Ensure OpenCV (opencv-python) is installed for webcam support.
Path to best.pt: If running locally, copy best.pt from Colab (download via files.download('/content/yolo_train_v11/log/weights/best.pt')) and update the path in the code.
Performance: Real-time detection requires a decent CPU/GPU. For smoother performance, use a GPU and adjust conf or imgsz (e.g., 320 for faster processing).


🎉 Usage
To test the model on images or videos:

Load the Model:
from ultralytics import YOLO
model = YOLO('yolo_train_v11/log/weights/best.pt')


Run Inference on an Image:
results = model.predict(source='path_to_image.jpg', save=True, conf=0.5)
results[0].show()  # Display the result with bounding boxes


Download the Model from Colab:
from google.colab import files
files.download('/content/yolo_train_v11/log/weights/best.pt')


Real-Time Webcam Detection (local machine):Use the code above (model.predict(source=0, show=True, conf=0.6)). Results are shown live, and you can save frames if needed (save=True).


Output images/videos are saved in runs/detect/. Check them out! 😍

🌟 Output

Trained Model: best.pt in yolo_train_v11/log/weights/.
Training Logs: Metrics and plots in yolo_train_v11/log/.
Inference Results: Images/videos with bounding boxes in runs/detect/.
Performance: Expect mAP@0.5 of ~0.7-0.9 (tune epochs or hyperparameters for better results).


🐞 Troubleshooting
Stuck? Here’s how to fix common issues:

Kaggle Download Fails 😕: Verify kaggle.json and dataset access.
Path Errors 🚫: Run !ls -R /content/mobile_dataset in Colab to check the dataset structure. Adjust img_dir or label_dir if needed.
Webcam Not Working 📹: Colab doesn’t support webcams. Run the detection code locally with OpenCV installed.
Memory Issues 💥: Reduce batch size or use GPU (device=0) for training. For webcam, lower imgsz (e.g., 320).
Missing Annotations ⚠️: Some images may lack labels. Filter them or verify dataset integrity.
Class Mismatch 🤔: Ensure mobile_phone matches the dataset’s class name (update classes list if needed).


🤝 Contributing
Love this project? Want to make it even cooler? Fork it, tweak it, and send a pull request! Let’s build something epic together! 🙌

📜 License

Dataset: CC0 (Public Domain) from Kaggle.
YOLOv11: Apache 2.0 License (Ultralytics).
This Project: Free to use and modify for educational purposes.


🎈 Final Notes
This project is a blast to work on, and you’ll end up with a model that can spot mobile phones in images and live video like a champ! 🦸 Whether you’re training in Colab or detecting phones with your webcam, you’re now part of the AI revolution. Keep experimenting and have fun! 🎉
Built with 💖 by a passionate coder (you!). Powered by YOLOv11, Ultralytics, and a sprinkle of curiosity.