import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.set_page_config(page_title="Atmospheric Aerosol Forecast", layout="wide")

# ----------------------------
# MODEL DEFINITION
# ----------------------------
class BetterAerosolCNN(nn.Module):
    def __init__(self, in_channels=10, out_channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# LOAD MODEL + DATA
# ----------------------------
@st.cache_resource
def load_model():
    model = BetterAerosolCNN(in_channels=10, out_channels=2)
    state = torch.load("best_aerosol_cnn.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_data
def load_last_maps():
    return np.load("last_5_maps.npy").astype(np.float32)

model = load_model()
last_5_maps = load_last_maps()

# ----------------------------
# FORECAST FUNCTION
# ----------------------------
def recursive_forecast(model, history, days):
    history = history.copy()
    future_preds = []

    with torch.no_grad():
        for _ in range(days):
            x = np.transpose(history, (0, 3, 1, 2))   # [5,2,H,W]
            x = x.reshape(1, 10, history.shape[1], history.shape[2])
            x = torch.tensor(x, dtype=torch.float32)

            pred = model(x).numpy()[0]                # [2,H,W]
            pred_hwc = np.transpose(pred, (1, 2, 0)) # [H,W,2]

            future_preds.append(pred_hwc)
            history = np.concatenate([history[1:], pred_hwc[None, ...]], axis=0)

    return np.array(future_preds, dtype=np.float32)

# ----------------------------
# UI
# ----------------------------
st.title("AI-Based Atmospheric Aerosol Forecast")
st.write("Forecast future aerosol spatial maps from the trained CNN model.")

st.sidebar.header("Controls")
days = st.sidebar.slider("Forecast days", 1, 30, 7)
channel_name = st.sidebar.selectbox("Aerosol Variable", ["BCSMASS", "DUSMASS25"])
day_to_show = st.sidebar.slider("Predicted day to display", 1, days, 1)
predict_btn = st.sidebar.button("Predict")

channel_idx = 0 if channel_name == "BCSMASS" else 1

if predict_btn:
    with st.spinner("Generating forecast..."):
        future_preds = recursive_forecast(model, last_5_maps, days)

    selected_day_map = future_preds[day_to_show - 1, :, :, channel_idx]
    weekly_avg = future_preds[:min(7, days)].mean(axis=0)[:, :, channel_idx]
    period_avg = future_preds.mean(axis=0)[:, :, channel_idx]
    last_obs = last_5_maps[-1, :, :, channel_idx]

    st.success(f"Forecast created for {days} day(s).")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{channel_name} - Predicted Day {day_to_show}")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(selected_day_map)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    with col2:
        st.subheader(f"{channel_name} - Last Observed Map")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(last_obs)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader(f"{channel_name} - Weekly Average Forecast")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(weekly_avg)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    with col4:
        st.subheader(f"{channel_name} - Average Forecast for Selected Period")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(period_avg)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    st.subheader("Forecast Array Details")
    st.write("Forecast shape:", future_preds.shape)
    st.write("Last 5 maps shape:", last_5_maps.shape)

else:
    st.info("Choose settings from the sidebar and click Predict.")
