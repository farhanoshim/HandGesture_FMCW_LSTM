# HandGesture_FMCW_LSTM

# FMCW Radar Gesture Dataset Preprocessing

This project provides a preprocessing pipeline for gesture recognition data collected using Frequency-Modulated Continuous Wave (FMCW) radar. The data is used to prepare inputs for machine learning models (e.g., LSTM) by selecting the most relevant object points based on radar signal characteristics.

## Dataset
Dataset can be downloaded from: [Google Drive Link](https://drive.google.com/drive/folders/1FPZhGzkBPEEVMm2rtcAvdu52XoEuX5Mr?usp=sharing)

The gesture data consists of frames with multiple object points. Each object contains:
- Range
- Velocity
- Peak Value
- X and Y position

Data is stored in CSV format, where each row corresponds to one object in a frame.

## ⚙️ Preprocessing Pipeline

The Python script:
- Reads raw CSV radar data
- Extracts up to the top-2 objects with highest peak values per frame
- Pads the gesture sequence with zeros **at the beginning** so that the meaningful frames are aligned to the **end** of the fixed-size array
- Stores each gesture as a `(80, 4, 2)` numpy array (80 frames, 4 features, 2 top objects)

This preprocessing ensures that the input is suitable for models like LSTM which expect uniform input dimensions.

## Sample Input Shape

```python
(80, 4, 2)
# 80 frames per gesture
# 4 features: velocity, peak_value, x, y
# Top-2 objects per frame

## Citation

If you use this data or preprocessing approach, please cite the original dataset paper:

> Grobelny, Piotr, and Adam Narbudowicz.  
> ["MM-Wave radar-based recognition of multiple hand gestures using long short-term memory (LSTM) neural network."](https://doi.org/10.3390/electronics11050787)  
> *Electronics* 11.5 (2022): 787.  
> https://doi.org/10.3390/electronics11050787

The gesture recognition model used in this project is inspired by the LSTM-based architecture described in the paper above.  
You can find the official model and code here:  
🔗 [Original Model Repository (GitHub)](https://github.com/petergry/Hand-Gesture-Recognition-Radar-LSTM)
