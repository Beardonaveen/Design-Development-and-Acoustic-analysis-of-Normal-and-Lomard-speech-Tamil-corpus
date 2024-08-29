import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def extract_mfccs(audio, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def extract_chroma(audio, sr):
    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    return chroma_mean

def extract_spectral_contrast(audio, sr):
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
    return spectral_contrast_mean

def extract_features(file_path):
    audio, sr = load_audio(file_path)
    mfccs = extract_mfccs(audio, sr)
    chroma = extract_chroma(audio, sr)
    spectral_contrast = extract_spectral_contrast(audio, sr)
    features = np.hstack([mfccs, chroma, spectral_contrast])
    return features

# Example file paths and labels
file_paths = ["Normal speech.Wav", "Lombard speech.Wav"]
labels = ["normal", "lombard"]

# Load and preprocess dataset
def load_dataset(file_paths, labels):
    features = []
    for file_path, label in zip(file_paths, labels):
        feature = extract_features(file_path)
        features.append((feature, label))
    X = np.array([f[0] for f in features])
    y = np.array([f[1] for f in features])
    return X, y

X, y = load_dataset(file_paths, labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model for later use
model.save('speech_classification_model.h5')

# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('speech_classification_model.h5')

# Function to classify new audio file
def classify_audio(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    predicted_label = le.inverse_transform(np.round(prediction).astype(int).flatten())
    return predicted_label[0]

# Classify a new audio file
new_audio_file = "path_to_new_audio_file.Wav"
predicted_class = classify_audio(new_audio_file)
print(f"The given audio is classified as: {predicted_class}")
