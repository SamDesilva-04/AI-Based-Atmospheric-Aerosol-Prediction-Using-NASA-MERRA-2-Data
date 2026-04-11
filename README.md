
# 🌍 AI-Based Atmospheric Aerosol Prediction Using NASA MERRA-2 Data

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-FF4B4B?logo=streamlit)
![Data](https://img.shields.io/badge/Data-NASA%20MERRA--2-0B3D91?logo=nasa)
![R2](https://img.shields.io/badge/R%C2%B2%20Score-~0.844-brightgreen)

An end-to-end machine learning pipeline to **predict atmospheric aerosol optical depth (AOD)** from NASA MERRA-2 global satellite reanalysis data, with an interactive Streamlit deployment for real-time spatial inference.

---

## 🎯 Results

| Metric | Value |
|---|---|
| R² Score | ~0.844 |
| Data Source | NASA MERRA-2 (M2T1NXAER) |
| Spatial Coverage | Global gridded (0.5° × 0.625°) |
| Variables Predicted | Black Carbon PM2.5, Dust PM2.5, AOD |
| Model | BetterAerosolCNN (custom PyTorch CNN) |
| Platform | Google Colab + Streamlit |

---

## 🏗️ Architecture — BetterAerosolCNN

```
Input: Gridded spatiotemporal array [C × H × W]
(multi-channel MERRA-2 atmospheric variables)
            │
            ▼
    Conv2d(in, 64, 3×3) + BN + ReLU
            │
    Conv2d(64, 128, 3×3) + BN + ReLU
            │
       MaxPool2d(2×2)
            │
    Conv2d(128, 256, 3×3) + BN + ReLU
            │
       AdaptiveAvgPool
            │
    Flatten → FC(256) → Dropout → FC(1)
            │
            ▼
    Regression Output (AOD value)
```

### Design Rationale
- **Spatial CNN** — captures local and regional spatial correlations in gridded atmospheric data better than tabular models
- **Batch Normalization** — stabilizes training on satellite data which has large value ranges across channels
- **Regression head** — continuous AOD prediction rather than classification
- **Multi-channel input** — stacks multiple MERRA-2 variables (BC, dust, SO4, etc.) as channels

---

## 📡 Data Pipeline

```
NASA GES DISC
     │  (download .nc4 files via wget/OPeNDAP)
     ▼
netCDF4 / xarray ingestion
     │  (parse spatiotemporal arrays)
     ▼
Preprocessing
     │  (normalization, NaN handling, channel stacking)
     ▼
PyTorch Dataset & DataLoader
     │  (batched gridded patches)
     ▼
BetterAerosolCNN training
     │  (MSE loss, Adam optimizer)
     ▼
Evaluation (R², MAE, RMSE)
     │
     ▼
Streamlit deployment
(real-time AOD map inference + visualization)
```

### MERRA-2 Variables Used
| Variable | Description |
|---|---|
| `BCSMASS` | Black Carbon Surface Mass Concentration |
| `DUSMASS25` | Dust PM2.5 Surface Mass Concentration |
| `SO4SMASS` | Sulfate Surface Mass Concentration |
| `OCSMASS` | Organic Carbon Surface Mass Concentration |
| `TOTEXTTAU` | Total Aerosol Extinction AOD (target) |

---

## 🚀 Getting Started

### Requirements
```bash
pip install torch torchvision xarray netCDF4 numpy pandas matplotlib streamlit scikit-learn
```

### Download MERRA-2 Data
Register at [NASA GES DISC](https://disc.gsfc.nasa.gov/) and download `M2T1NXAER` collection files.

```bash
wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies \
     --auth-no-challenge=on --keep-session-cookies \
     "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXAER.5.12.4/..."
```

### Run Training
```python
python train.py --data_dir ./merra2_data --epochs 50 --batch_size 16
```

### Launch Streamlit App
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Framework | PyTorch |
| Data I/O | xarray, netCDF4 |
| Preprocessing | NumPy, Pandas |
| Visualization | Matplotlib, Cartopy |
| Deployment | Streamlit |
| Platform | Google Colab + local |
| Data Source | NASA MERRA-2 (GES DISC) |

---

## 📄 Related Work

This project is part of a broader research direction in **atmospheric transition regime detection** using Conv-LSTM Autoencoders, HDBSCAN clustering, UMAP visualization, and SHAP attribution over ~20 years of multi-pollutant satellite data across Asia.

---

## 👤 Author

**Sam Desilva** — B.Tech Information Technology, Christ University, Bangalore
- GitHub: [@SamDesilva-04](https://github.com/SamDesilva-04)
- LinkedIn: [sam-desilva-434748319](https://linkedin.com/in/sam-desilva-sb)
