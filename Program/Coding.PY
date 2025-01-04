import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import pandas as pd
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')
data = pd.read_excel('/content/drive/My Drive/proglanjut/ALDO.xlsx')
data


# Load dataset
# Ensure the dataset has the required columns
# Replace with your actual dataset loading logic
data = data[["timestamp", "open", "low", "high", "close"]]  # Replace with actual loading

# Convert and process dataset
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Add volatility and moving average features
data['volatility'] = data['high'] - data['low']
data['moving_avg'] = data['close'].rolling(window=10).mean()
data.fillna(0, inplace=True)  # Handle NaN values caused by rolling

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['open', 'low', 'high', 'close', 'volatility', 'moving_avg']])

# Prepare training and test data
sequence_length = 60  # Number of past days used for prediction
X, y = [], []
for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i])  # Use all features
    y.append(data_scaled[i, 3])  # Predict based on 'close'
X, y = np.array(X), np.array(y)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Build the GRU model with dropout for regularization
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Add Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with validation split
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=50,
    callbacks=[early_stopping],
    verbose=1
)

# Predict test set
y_pred_scaled = model.predict(X_test)
y_test_unscaled = scaler.inverse_transform(np.concatenate([np.zeros((len(y_test), 3)), y_test.reshape(-1, 1), np.zeros((len(y_test), 2))], axis=1))[:, 3]
y_pred_unscaled = scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred_scaled), 3)), y_pred_scaled, np.zeros((len(y_pred_scaled), 2))], axis=1))[:, 3]

# Evaluate the model
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
r2 = r2_score(y_test_unscaled, y_pred_unscaled)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Visualize historical and predicted stock prices
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['close'], label="Actual Prices", color="blue")
plt.plot(data.index[-len(y_test_unscaled):], y_pred_unscaled, label="Predicted Prices", color="orange")
plt.title("Stock Prices Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Plot moving average
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['close'], label="Actual Prices", color="blue")
plt.plot(data.index, data['moving_avg'], label="Moving Average (10 days)", color="green")
plt.title("Stock Prices with Moving Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Predict a future price
future_date = input("Enter the date to predict (YYYY-MM-DD): ")
future_date = datetime.strptime(future_date, "%Y-%m-%d")
predicted_price = scaler.inverse_transform(
    np.concatenate([
        np.zeros((1, 3)),
        model.predict(data_scaled[-sequence_length:].reshape(1, sequence_length, X_train.shape[2])),
        np.zeros((1, 2))
    ], axis=1)
)[0, 3]

# Add the predicted point to the chart
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['close'], label="Actual Prices", color="blue")
plt.plot(data.index[-len(y_test_unscaled):], y_pred_unscaled, label="Predicted Prices", color="orange")
plt.scatter(future_date, predicted_price, label=f"Predicted Price ({future_date.date()}): {predicted_price:.2f}", color="red")
plt.title("Stock Prices Prediction with Future Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(14, 7))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.figure(figsize=(14, 7))
data['close'].resample('D').mean().plot(label='Harga Harian', color='blue')
plt.title("Harga Saham Per Hari")
plt.xlabel("Tanggal")
plt.ylabel("Harga")
plt.legend()
plt.show()

# Grafik harga saham per minggu
plt.figure(figsize=(14, 7))
data['close'].resample('W').mean().plot(label='Harga Mingguan', color='orange')
plt.title("Harga Saham Per Minggu")
plt.xlabel("Tanggal")
plt.ylabel("Harga")
plt.legend()
plt.show()

# Grafik harga saham per bulan
plt.figure(figsize=(14, 7))
data['close'].resample('M').mean().plot(label='Harga Bulanan', color='green')
plt.title("Harga Saham Per Bulan")
plt.xlabel("Tanggal")
plt.ylabel("Harga")
plt.legend()
plt.show()