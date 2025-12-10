# 🍅 Tomato Ripeness Detection System

A machine learning system to classify tomatoes as **ripe** or **unripe** with a user-friendly web interface.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Fast Local Training)
```bash
# Auto-download dataset and train (6 epochs for quick results)
python train.py --epochs 6

# Or specify your own dataset path
python train.py --data "path/to/Images" --epochs 10 --batch 16
```

### 3. Run the Web App
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and upload tomato images!

## Features

### 🎯 **Smart Classification**
- **Ripe**: Red/orange tomatoes
- **Unripe**: Green/yellow tomatoes  
- **Undefined**: Non-tomato images or low-confidence predictions

### 🖥️ **Web Interface**
- Drag & drop image upload
- Real-time predictions
- Confidence scores
- Automatic "not a tomato" detection

### ⚡ **Fast Training**
- Uses EfficientNetB0 (lighter than B3)
- Data augmentation for better accuracy
- Early stopping to prevent overfitting
- Typically trains in 5-15 minutes on CPU

## 📋 Requirements

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn pandas numpy tqdm kagglehub
```

## 🚀 Quick Start

### Method 1: Using the Jupyter Notebook (Recommended)

1. **Open the notebook**: `tomatoes-f1-score-98.ipynb`
2. **Run all cells** in order to:
   - Download the dataset
   - Train the model
   - Save the trained model
3. **Test with your images** using the prediction functions

### Method 2: Using the Python Script

1. **Train the model first** by running the notebook
2. **Use the standalone script**:
```bash
python tomato_predictor.py your_tomato_image.jpg
```

## 📖 How to Use

### For Training (First Time Setup)

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook tomatoes-f1-score-98.ipynb
   ```

2. **Run all cells** in sequence:
   - Cell 1-2: Download dataset from Kaggle
   - Cell 3-4: Import libraries and setup
   - Cell 5-15: Data preprocessing and model training
   - Cell 16-17: Model evaluation and saving

3. **The trained model** will be saved as `tomatoes-XX.XX.h5`

### For Prediction (After Training)

#### In Jupyter Notebook:
```python
# Method 1: Simple prediction
prediction, confidence = predict_tomato_ripeness('your_image.jpg', model)
print(f'Tomato is {prediction} with {confidence:.1%} confidence')

# Method 2: Display image with prediction
predict_and_display('your_image.jpg', model)

# Method 3: Multiple images
image_list = ['tomato1.jpg', 'tomato2.jpg', 'tomato3.jpg']
results = predict_multiple_images(image_list, model)
```

#### Using Python Script:
```bash
# Basic prediction
python tomato_predictor.py my_tomato.jpg

# With custom model
python tomato_predictor.py my_tomato.jpg --model my_model.h5

# Save result image
python tomato_predictor.py my_tomato.jpg --save
```

## 📁 Dataset

The system uses the **"Riped and Unriped Tomato Dataset"** from Kaggle:
- **Source**: https://www.kaggle.com/datasets/techkhid/riped-and-unriped-tomato-dataset
- **Classes**: Ripe and Unripe tomatoes
- **Format**: JPEG images
- **Automatic Download**: The notebook downloads this automatically

## 🏗️ Model Architecture

- **Base Model**: EfficientNetB3 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned for tomato classification
- **Input Size**: 224x300 pixels
- **Output**: Binary classification (ripe/unripe)
- **Training Features**:
  - Data augmentation for better generalization
  - Custom callback for learning rate adjustment
  - Early stopping to prevent overfitting

## 📊 Performance

- **F1-Score**: 98%+
- **Accuracy**: 98%+
- **Validation Accuracy**: High performance on unseen data
- **Inference Time**: <1 second per image

## 🔧 File Structure

```
├── tomatoes-f1-score-98.ipynb    # Main training notebook
├── tomato_predictor.py            # Standalone prediction script
├── README.md                      # This file
├── tomatoes-XX.XX.h5             # Trained model (created after training)
└── aug/                           # Augmented images folder (created during training)
```

## 💡 Usage Tips

### For Best Results:
- Use **clear, well-lit images**
- Ensure the **tomato is the main subject**
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Images are automatically resized to training dimensions

### Image Requirements:
- **Good**: Clear tomato images with good lighting
- **Avoid**: Blurry, dark, or heavily occluded images
- **Multiple tomatoes**: Works best with single tomato per image

## 🐛 Troubleshooting

### Common Issues:

1. **"Model not found" error**:
   - Run the training notebook first to create the model
   - Check that the `.h5` file exists in the folder

2. **"Image not found" error**:
   - Verify the image path is correct
   - Check file extension (.jpg, .jpeg, .png)

3. **Low prediction confidence**:
   - Ensure image is clear and well-lit
   - Check that the subject is clearly a tomato

4. **Import errors**:
   - Install required packages: `pip install -r requirements.txt`
   - Ensure Python 3.7+ is being used

## 🔄 Retraining the Model

To retrain with new data:

1. **Add new images** to the dataset folder
2. **Update the data loading** section in the notebook
3. **Run the training cells** again
4. **New model** will be saved with updated timestamp

## 📈 Model Evaluation

The notebook includes comprehensive evaluation:
- **Confusion Matrix**: Visual representation of predictions
- **Classification Report**: Precision, recall, F1-score per class
- **Training Plots**: Loss and accuracy curves
- **Sample Predictions**: Visual verification of model performance

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features
- Optimize the model

## 📄 License

This project is open source and available under the MIT License.

## 🎉 Ready to Use!

Your tomato ripeness detection system is ready for production use! 

**Start by running the notebook to train your model, then use either the notebook functions or the Python script to make predictions on new tomato images.**