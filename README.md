### **Summary for README**

---

# **Multi-Digit Image Recognition with PyTorch**

This project detects and recognizes multiple digits in an image using **PyTorch**, **OpenCV**, and **Gradio**. The model segments digits from an image, processes them, and predicts each digit using a CNN trained on the MNIST dataset.

---

## **How It Works**
1. **Digit Detection**: OpenCV detects individual digits in the image using contour detection.
2. **Preprocessing**: Detected digits are cropped, resized, and normalized for the model.
3. **Prediction**: Each digit is passed to a CNN model trained on MNIST to predict its class.
4. **Interactive App**: Gradio provides an interface for uploading images and viewing predictions.

---

## **How to Set Up and Run**

### **1. Install Python and Clone the Repository**
```bash
git clone https://github.com/your-username/multi-digit-recognition.git
cd multi-digit-recognition
```

### **2. Install Dependencies**
```bash
pip install torch torchvision pillow opencv-python gradio
```

### **3. Prepare the Model**
- Place the trained model (`model.pth`) in the `outputs/` folder.
- If the model is missing, train it using:
  ```bash
  python train.py
  ```

### **4. Launch the Gradio App**
Run the app:
```bash
python predict_digit_gradio.py
```
Open the URL provided by Gradio in your browser.

---

## **Features**
- Supports multiple digits in a single image.
- Interactive, easy-to-use web interface.
- Powered by PyTorch, OpenCV, and Gradio.

---

## **Next Steps**
1. Improve contour detection for overlapping or noisy digits.
2. Retrain the model with augmented datasets (e.g., colored or stylized digits).
3. Deploy the app to a cloud service like AWS, Heroku, or Hugging Face Spaces.

---

