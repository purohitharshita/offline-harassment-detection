# Harassment Detection System Setup Instructions

## 📁 Project Structure

After downloading the updated files, your project directory should look like this:

```
OfflineHarassmentDetection/
├── .streamlit/
│   └── config.toml
├── models/
│   ├── audio_harassment_model (3).pkl
│   └── video_harassment_model (3).pkl
├── ui.py
├── requirements.txt
└── README.md
```

## 🚀 Installation Steps

1. **Create the .streamlit folder** in your project root directory:
   ```bash
   mkdir .streamlit
   ```

2. **Move the config.toml file** into the .streamlit folder:
   - Place the provided `config.toml` file inside the `.streamlit` folder
   - This will remove the 200MB upload limit

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify your model files** are in the correct location:
   - Ensure your trained models are in the `models/` folder
   - Update the file paths in `ui.py` if your models are located elsewhere

## 🏃‍♂️ Running the Application

Start the application with:
```bash
streamlit run ui.py
```

## ✨ Features

- **Single File Upload**: Analyze individual audio/video files
- **Multiple Files Upload**: Process multiple files at once
- **ZIP Folder Upload**: Upload entire folders (any size)
- **Real-time Progress Tracking**: See processing status
- **Interactive Results**: Charts and detailed analysis
- **CSV Export**: Download results for further analysis

## 🔧 Troubleshooting

### Large File Upload Issues

If you still encounter upload size limits:

1. **Check config file location**: Ensure `config.toml` is in `.streamlit/` folder
2. **Restart the application**: Stop and restart Streamlit after adding config
3. **Verify config syntax**: Ensure no extra spaces or syntax errors in config.toml

### Memory Issues with Very Large Files

For files larger than your system RAM:
- The application uses chunked processing to handle large files
- Close other applications to free up memory
- Consider processing files in smaller batches

### Model Loading Errors

If models fail to load:
- Verify model file paths in `ui.py`
- Ensure model files are not corrupted
- Check that you have sufficient disk space

## 🆘 Support

For additional help:
1. Check the error messages in the Streamlit interface
2. Verify all file paths match your directory structure
3. Ensure all dependencies are properly installed