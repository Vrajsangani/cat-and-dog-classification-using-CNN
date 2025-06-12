# cat-and-dog-classification-using-CNN
🐶🐱 Classification of Dogs and Cats using SVM
This project focuses on image classification using Support Vector Machine (SVM) to distinguish between images of dogs and cats. It uses traditional machine learning with feature extraction (not deep learning), offering a computationally efficient approach for binary image classification.

📌 Project Objective
To build a binary classifier that can:

Take an image as input

Identify whether it's a dog or a cat

Use Support Vector Machine (SVM) for classification

📁 Files Included
classification of dog and cat using SVM.ipynb – Main Jupyter notebook containing the code, visualizations, and model building process.

dataset/ – Folder containing images of dogs and cats (not included here, you need to provide it).

README.md – Project overview and instructions.

🧪 Technologies & Libraries Used
Python 3.x

Jupyter Notebook

OpenCV (cv2) – Image processing

NumPy – Numerical operations

Matplotlib – Visualization

Scikit-learn – Machine learning tools (SVM, train-test split, evaluation)

⚙️ Methodology
Data Collection

Load image data from a structured directory: dataset/dogs and dataset/cats.

Data Preprocessing

Resize all images to a uniform shape (e.g., 64x64).

Convert to grayscale or flatten RGB.

Normalize the pixel values.

Feature Extraction

Flatten image matrix into a 1D feature vector.

Combine all features and assign labels (0: Cat, 1: Dog).

Train-Test Split

80% for training and 20% for testing.

Model Training

Use SVM with RBF kernel (or linear) from sklearn.svm.SVC.

Evaluation

Accuracy Score

Confusion Matrix

Classification Report

Prediction

Test the model on unseen data or user-provided images.

🚀 How to Run
Clone the repository or download the files.

Install dependencies:

bash
Copy
Edit
pip install numpy opencv-python scikit-learn matplotlib
Run the notebook:

bash
Copy
Edit
jupyter notebook "classification of dog and cat using SVM.ipynb"
Ensure you have a dataset directory structured like:

markdown
Copy
Edit
dataset/
  ├── cats/
  │     ├── cat1.jpg
  │     ├── cat2.jpg
  │     └── ...
  └── dogs/
        ├── dog1.jpg
        ├── dog2.jpg
        └── ...
📊 Results
The model achieved an accuracy of around XX% (replace with actual accuracy) on the test dataset.

SVM performed well with simple pixel-based features for small image datasets.

📈 Future Improvements
Use Histogram of Oriented Gradients (HOG) or SIFT for better features.

Try deep learning models like CNN (e.g., using TensorFlow or PyTorch).

Deploy the model via a Flask or Streamlit web app.

📄 License
This project is open-source and available under the MIT License.
