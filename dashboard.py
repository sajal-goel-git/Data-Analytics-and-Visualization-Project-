# Crop Recommendation Dashboard in Streamlit
import streamlit as st
import pandas as pd
import os
from PIL import Image
import joblib
import base64

# === PAGE CONFIG ===
st.set_page_config(page_title="\U0001F33E Crop Recommendation Dashboard", layout="wide")

# === LOAD PYTHON MODEL ===
from model_script import predict_crop  # DO NOT TOUCH MODEL

# === LOAD DATA ===
csv_path = "Crop_recommendation_corrected.csv"
img_dir = "images"
def show_image(filename, caption="", use_container_width=True):
    path = os.path.join(img_dir, filename)
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_column_width=use_container_width)
    else:
        st.warning(f"⚠ {filename} not found in {img_dir}")
# === IMAGE DESCRIPTIONS ===

# === STYLING TO REDUCE SPACING ===
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .main > div:first-child {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.title("\U0001F4D1 Navigation")
page = st.sidebar.radio("Go To", [
    "Overview", "CSV Visualizations", "Crop Prediction Model", "Model Evaluation & ANOVA Analysis", "Image Visualizations", "Dataset Preview"
])

# === PAGE ROUTING ===
if page == "Overview":
    st.title("🌿 Data Driven Crop Recommendation System")
    
    st.markdown("""
This project uses a machine learning model to recommend the most suitable crop based on soil nutrients and climate features.It also explores the dataset through statistical visualizations and image-based inspections.


#### Data Acquisition & Preprocessing  
- Crop dataset containing soil nutrients (N, P, K), temperature, humidity, pH, and rainfall  
- Collected crop leaf images (healthy & diseased)  
- Cleaned and standardized both datasets

#### Crop Recommendation Model  
- Trained a Random Forest classifier to predict suitable crops  
- Achieved strong performance and deployed for live use

#### CSV Data Visualizations & ANOVA Testing  
- Visualized key patterns using histograms, heatmaps, 3D plots,etc
- Conducted *One-Way ANOVA* → All 7 parameters showed *significant differences* across crops (p < 0.05)

#### Image Visualizations   
- Visualized healthy vs. diseased leaf patterns to highlight differences in color, structure, and distribution
- Extracted scientific insights from image data, revealing consistent and significant variations across all crops

#### Interactive Streamlit Dashboard  
- Unified all analysis into a navigable, responsive dashboard  
- Offers crop prediction, visual insights, and dataset previews
""")
    st.subheader("✨ Project Summary")
    show_image('overview.png')

elif page == "CSV Visualizations":
    st.title("📊 CSV-Based Visualizations")

    show_image("crop_distribution.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("This histogram shows that equal number of samples (100) for all 15 crops is taken.")
    with open(os.path.join(img_dir, "crop_distribution.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="crop_distribution.png", mime="image/png")
    st.markdown("---")

    show_image("correlation_matrix.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("🔍 Top Positive Correlation (> 0.5):\n- P & K: 0.837\n\n🔍 Top Negative Correlation (< -0.2):\n- N & K: -0.262\n- P & rainfall: -0.316\n\n✅ Key Observations:\n- Strong P–K correlation\n- Weak/Negative correlation of N with P & K\n- Rainfall positively correlates with N and humidity, but negatively with P.\n- pH is nearly uncorrelated with most variables.")
    with open(os.path.join(img_dir, "correlation_matrix.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="correlation_matrix.png", mime="image/png")
    st.markdown("---")

    show_image("better_nutrient_histograms.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("""
### 📊 Nutrient & Environmental Summary

A statistical breakdown of key soil and climate features used in modeling:

---

#### 🌱 *Nitrogen (N)*
- *Mean*: 50.62    
- *Median*: 37.00  
- *Standard Deviation*: 37.76  
- *Range*: 0.00 → 140.00  
- *Skewness: 0.49*(Symmetrical)  

---

#### 🌿 *Phosphorus (P)*
- *Mean*: 56.65  
- *Median*: 51.00  
- *Standard Deviation*: 36.79  
- *Range*: 5.00 → 145.00  
- *Skewness: 0.93*(Right-skewed)

---

#### 🪴 *Potassium (K)*
- *Mean*: 55.29  
- *Median*: 34.00  
- *Standard Deviation*: 59.11  
- *Range*: 5.00 → 205.00  
- *Skewness: 1.86*(Right-skewed)  
➡ *Highly skewed distribution – transformation may help.*

---

#### 🌡 *Temperature*
- *Mean*: 24.08°C  
- *Median*: 24.14°C  
- *Standard Deviation*: 4.69  
- *Range*: 8.83 → 41.95  
- *Skewness: 0.10*(Symmetrical)  
⚠ *Potential upper outliers.*

---

#### 💧 *Humidity*
- *Mean*: 70.06%  
- *Median*: 80.00%  
- *Standard Deviation*: 23.80  
- *Range*: 14.26 → 99.98  
- *Skewness: -1.10*(Left-skewed)  
➡ *Highly skewed – may require transformation.*

---

#### ⚗ *pH*
- *Mean*: 6.42  
- *Median*: 6.31  
- *Standard Deviation*: 0.71  
- *Range*: 4.51 → 8.87  
- *Skewness: 0.52*(Right-skewed)  
⚠ *Slight upper-end outliers possible.*

---

#### 🌧 *Rainfall*
- *Mean*: 116.08 mm  
- *Median*: 105.06 mm  
- *Standard Deviation*: 50.94  
- *Range*: 35.03 → 298.56  
- *Skewness: 1.14*(Right-skewed)  
➡ *Transformation may help skewness.*  
⚠ *Upper outliers detected.*

---
""")
    with open(os.path.join(img_dir, "better_nutrient_histograms.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="better_nutrient_histograms.png", mime="image/png")
    st.markdown("---")

    show_image("5_scatter_matrix.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("""
### Correlation Matrix Insights

**Strong Correlations (|r| > 0.7):**
- P & K: 0.84 (Strong Positive)

**Top 3 Most Correlated Pairs:**
- P & K: 0.84  
- P & Rainfall: -0.32  
- Humidity & Rainfall: 0.28

**3 Least Correlated Pairs:**
- N & Humidity: 0.06  
- pH & Rainfall: -0.05  
- P & Humidity: -0.02

---

### Descriptive Statistics & Distribution Shape

*Nitrogen (N)*  
- Mean: 50.62  Median: 37.00  Std: 37.76  
- Skewness: 0.49 → Approximately symmetric

*Phosphorus (P)*  
- Mean: 56.65  Median: 51.00  Std: 36.79  
- Skewness: 0.93 → Moderately skewed

*Potassium (K)*  
- Mean: 55.29  Median: 34.00  Std: 59.11  
- Skewness: 1.86 → Highly skewed

*Temperature*  
- Mean: 24.08°C  Median: 24.14°C  Std: 4.69  
- Skewness: 0.10 → Approximately symmetric

*Humidity*  
- Mean: 70.06  Median: 80.00  Std: 23.80  
- Skewness: -1.10 → Highly skewed

*pH*  
- Mean: 6.42  Median: 6.31  Std: 0.71  
- Skewness: 0.52 → Moderately skewed

*Rainfall*  
- Mean: 116.08 mm  Median: 105.06 mm  Std: 50.94  
- Skewness: 1.14 → Highly skewed
""")
    with open(os.path.join(img_dir, "5_scatter_matrix.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="5_scatter_matrix.png", mime="image/png")
    st.markdown("---")

    show_image("3d_nutrients.png")
    with st.expander("📌 Statistical Analysis"):

        st.markdown('''### 📊  Nutrient Requirement Analysis by Crop

#### Nitrogen
###### 🔼 Top 5 Crops by Nitrogen Requirement

| Crop     | Avg Nitrogen |
|----------|--------------|
| Cotton   | 117.77       |
| Coffee   | 101.20       |
| Banana   | 100.23       |
| Rice     | 79.89        |
| Jute     | 78.40        |


###### 🔽 Bottom 5 Crops by Nitrogen Requirement

| Crop        | Avg Nitrogen |
|-------------|--------------|
| Beans       | 20.75        |
| Mango       | 20.07        |
| Orange      | 19.58        |
| Pomegranate | 18.87        |
| Lentil      | 18.77        |


#### Phosphorus
###### 🔼 Top 5 Crops by Phosphorus Requirement

| Crop       | Avg Phosphorus |
|------------|----------------|
| Apple      | 134.22         |
| Grapes     | 132.53         |
| Banana     | 82.01          |
| Lentil     | 68.36          |
| Soya Bean  | 67.79          |


###### 🔽 Bottom 5 Crops by Phosphorus Requirement

| Crop        | Avg Phosphorus |
|-------------|----------------|
| Coffee      | 28.74          |
| Mango       | 27.18          |
| Pomegranate | 18.75          |
| Coconut     | 16.93          |
| Orange      | 16.55          |


#### Potassium
###### 🔼 Top 5 Crops by Potassium Requirement

| Crop        | Avg Potassium |
|-------------|----------------|
| Grapes      | 200.11         |
| Apple       | 199.89         |
| Soya Bean   | 79.92          |
| Banana      | 50.05          |
| Pomegranate | 40.21          |


###### 🔽 Bottom 5 Crops by Potassium Requirement

| Crop    | Avg Potassium |
|---------|----------------|
| Beans   | 20.05          |
| Maize   | 19.79          |
| Cotton  | 19.56          |
| Lentil  | 19.41          |
| Orange  | 10.01          |

### Nutrient Spread Summary (mean ± std)

| Crop        | N (mean ± std)     | P (mean ± std)     | K (mean ± std)     |
|-------------|---------------------|---------------------|---------------------|
| Rice        | 79.89 ± 11.92       | 47.58 ± 7.90        | 39.87 ± 2.95        |
| Maize       | 77.76 ± 11.95       | 48.44 ± 8.01        | 19.79 ± 2.94        |
| Soya Bean   | 40.09 ± 12.15       | 67.79 ± 7.50        | 79.92 ± 3.26        |
| Beans       | 20.75 ± 10.83       | 67.54 ± 7.57        | 20.05 ± 3.10        |
| Lentil      | 18.77 ± 12.20       | 68.36 ± 7.34        | 19.41 ± 2.97        |
| Pomegranate | 18.87 ± 12.62       | 18.75 ± 7.39        | 40.21 ± 3.03        |
| Banana      | 100.23 ± 11.11      | 82.01 ± 7.69        | 50.05 ± 3.38        |
| Mango       | 20.07 ± 12.33       | 27.18 ± 7.66        | 29.92 ± 3.10        |
| Grapes      | 23.18 ± 12.47       | 132.53 ± 7.62       | 200.11 ± 3.27       |
| Apple       | 20.80 ± 11.86       | 134.22 ± 8.14       | 199.89 ± 3.32       |
| Orange      | 19.58 ± 11.94       | 16.55 ± 7.69        | 10.01 ± 3.06        |
| Coconut     | 21.98 ± 11.76       | 16.93 ± 8.36        | 30.59 ± 3.00        |
| Cotton      | 117.77 ± 11.63      | 46.24 ± 7.35        | 19.56 ± 3.17        |
| Jute        | 78.40 ± 10.97       | 46.86 ± 7.20        | 39.99 ± 3.31        |
| Coffee      | 101.20 ± 12.35      | 28.74 ± 7.28        | 29.94 ± 3.25        |

---

### Cluster-Based Crop Grouping (Based on NPK Profiles)

| Cluster | Crops                                     |
|---------|-------------------------------------------|
| 0       | Coconut, Mango, Orange, Pomegranate       |
| 1       | Apple, Grapes                             |
| 2       | Coffee, Cotton (91%), Jute, Maize, Rice   |
| 3       | Banana, Cotton (9%)                       |
| 4       | Beans, Lentil, Soya Bean                  |
''')
    with open(os.path.join(img_dir, "3d_nutrients.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="3d_nutrients.png", mime="image/png")
    st.markdown("---")

    show_image("6_nutrient_heatmap.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("Heatmap showing Average Nutrient Levels by Crop.\nUseful for identifying high/low nutrient-demanding crops.")
    with open(os.path.join(img_dir, "6_nutrient_heatmap.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="6_nutrient_heatmap.png", mime="image/png")
    st.markdown("---")

    show_image("cropwise_boxplots.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("""
### 🔍 Automated Insights by Feature

---

#### *Feature: N*
*Top Crops by Median Value*
- Cotton: *117.0*
- Coffee: *103.0*
- Banana: *100.5*

*Crops with Most Outliers*
- Rice: 0
- Maize: 0
- Soya Bean: 0

---

#### *Feature: P*
*Top Crops by Median Value*
- Apple: *136.5*
- Grapes: *133.0*
- Banana: *81.0*

*Crops with Most Outliers*
- Rice: 0
- Maize: 0
- Soya Bean: 0

---

#### *Feature: K*
*Top Crops by Median Value*
- Grapes: *201.0*
- Apple: *200.0*
- Soya Bean: *79.0*

*Crops with Most Outliers*
- Rice: 0
- Maize: 0
- Soya Bean: 0

---

#### *Feature: Temperature*
*Top Crops by Median Value*
- Mango: *31.30*
- Banana: *27.44*
- Coconut: *27.39*

*Crops with Most Outliers*
- Rice: 0
- Maize: 0
- Soya Bean: 0

---

#### *Feature: Humidity*
*Top Crops by Median Value*
- Coconut: *94.96*
- Apple: *92.42*
- Orange: *91.96*

*Crops with Most Outliers*
- Rice: 0
- Maize: 0
- Soya Bean: 0

---

#### *Feature: pH*
*Top Crops by Median Value*
- Soya Bean: *7.36*
- Orange: *7.02*
- Lentil: *6.95*

*Crops with Most Outliers*
- Rice: 0
- Maize: 0
- Soya Bean: 0

---

#### *Feature: Rainfall*
*Top Crops by Median Value*
- Rice: *233.12*
- Jute: *175.59*
- Coconut: *172.00*

*Crops with Most Outliers*
- Rice: 0
- Maize: 0
- Soya Bean: 0

---
""")
    with open(os.path.join(img_dir, "cropwise_boxplots.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="cropwise_boxplots.png", mime="image/png")
    st.markdown("---")


elif page == "Crop Prediction Model":
    st.title("🌾Crop Prediction Model")
    st.markdown("""
    Enter the values below to receive a real-time prediction based on our trained machine learning model.
    """)

    st.header("\U0001F50D Enter Soil & Climate Parameters")
    N = st.number_input("Nitrogen (N)", 0, 140, 50)
    P = st.number_input("Phosphorus (P)", 0, 145, 50)
    K = st.number_input("Potassium (K)", 0, 205, 50)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    pH = st.number_input("pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)

    if st.button("\U0001F33F Recommend Crop"):
        prediction = predict_crop(N, P, K, temperature, humidity, pH, rainfall)
        st.success(f"✅ Recommended Crop: {prediction}")


# --- Section Title ---
elif page == "Model Evaluation & ANOVA Analysis":
    st.title("📊 Model Evaluation & ANOVA Analysis")
    
    # --- Evaluation Metrics Table ---
    st.subheader("✅ Overall Evaluation Metrics")

    eval_metrics = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Value": ["0.9967", "0.9968", "0.9967", "0.9967"]
    }

    # Create DataFrame
    eval_df = pd.DataFrame(eval_metrics)
    
    # Set index starting from 1
    eval_df.index = range(1, len(eval_df) + 1)
    
    # Display with left-aligned values and headers
    st.dataframe(
        eval_df.style.set_properties(subset=slice(None), **{'text-align': 'left'}),
        use_container_width=True
    )
    
    # --- Confusion Matrix Table (Simplified Example) ---
    st.subheader("🧮 Classification Summary")
    conf_df = pd.DataFrame({
        "Crop": ["apple", "banana", "beans", "coconut", "coffee", "cotton", "grapes", "jute", "lentil", "maize",
                 "mango", "orange", "pomegranate", "rice", "soya bean"],
        "Precision": ["1.00"] * 7 + ["0.95"] + ["1.00"] * 5 + ["1.00", "1.00"],
        "Recall":    ["1.00"] * 14 + ["1.00"],
        "F1-score":  ["1.00"] * 7 + ["0.98"] + ["1.00"] * 5 + ["0.97", "1.00"]}, index=range(1, 16))
    st.dataframe(conf_df.style.set_properties(subset=slice(None), **{'text-align': 'left'}).set_table_styles([{
        'selector': 'th',
        'props': [('text-align', 'left')]
    }]), use_container_width=True)

# -- confusion matrix--
    st.subheader("🧩 Confusion Matrix")
    show_image("confusion_matrix.png")

    
    # --- ANOVA Table ---
    st.subheader("🔬 ANOVA Results Across Crops")
    anova_data = {
        "Parameter": ["Rainfall", "pH", "Temperature", "Humidity", "N", "P", "K"],
        "F-value": ["868.431", "100.080", "69.727", "5656.071", "975.119", "2368.657", "37750.841"],
        "p-value": ["0.000"] * 7
    }
    anova_df = pd.DataFrame(anova_data, index=range(1, 8))
    st.dataframe(anova_df.style.set_properties(subset=slice(None), **{'text-align': 'left'}).set_table_styles([{
        'selector': 'th',
        'props': [('text-align', 'left')]
    }]), use_container_width=True)
    
    st.success("✅ Significant differences detected among crops for all parameters.")
    

elif page == "Image Visualizations":
    st.title("🧭 Image-Based Visual Insights")

    show_image("class_distribution.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("📊 This Bar Chart shows number of images per class for both healthy and diseased crops.")
    with open(os.path.join(img_dir, "class_distribution.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="class_distribution.png", mime="image/png")
    st.markdown("---")

    show_image("bean_samples.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("Sample images of bean leaves showing the comparison of healthy and diseased crops' leaves.")
    with open(os.path.join(img_dir, "bean_samples.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="bean_samples.png", mime="image/png")
    st.markdown("---")

    show_image("all_crops_color_comparison.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("Healthy VS Diseased Leaf Color Distribution (HSV Hue Channel).")
    with open(os.path.join(img_dir, "all_crops_color_comparison.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="all_crops_color_comparison.png", mime="image/png")
    st.markdown("---")

    show_image("scientific_insights.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("""
### Key Scientific Findings (Color Shift Analysis)

- *Strongest Color Shift:* Pomegranate (Δ = 0.14)  
- *Weakest Color Shift:* Cotton (Δ = 0.08)  
- *Statistical Significance:* All 15 crops showed statistically significant color shifts (p < 0.05)  

These findings highlight the potential of hue-based color metrics in accurately identifying disease presence across diverse crop types.
    """)
    with open(os.path.join(img_dir, "scientific_insights.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="scientific_insights.png", mime="image/png")
    st.markdown("---")

    show_image("banana_comparison.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("""
###  Banana Hue Analysis

- *Mean Hue (Healthy)*: 0.2188  
- *Mean Hue (Diseased)*: 0.2062  
- *T-test p-value*: 0.4827

⚪ No statistically significant hue difference between healthy and diseased banana leaf images.
    """)
    with open(os.path.join(img_dir, "banana_comparison.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="banana_comparison.png", mime="image/png")
    st.markdown("---")

    show_image("apple_anomaly_map.png")
    with st.expander("📌 Statistical Analysis"):
        st.markdown("""
### 🍎 Apple Leaf Disease Insight

- *Disease Severity:* Moderate  
  (Average hue difference: 0.34 ± 0.22)  
- *Affected Area:* 52.3% of the leaf surface  
- *Spread Pattern:* Disease is widespread across the leaf  
- *Max Hue Shift Intensity:* 1.00

These metrics indicate a noticeable and diffused disease manifestation in apple leaves, detectable via hue-based color analysis.
    """)
    with open(os.path.join(img_dir, "apple_anomaly_map.png"), "rb") as file:
        st.download_button("⬇️ Download Image", file, file_name="apple_anomaly_map.png", mime="image/png")
    st.markdown("---")

elif page == "Dataset Preview":
    st.title("\U0001F4C1 Dataset Preview")

    st.markdown("Here you can explore the complete csv dataset used for training the crop recommendation model.")

    # Load dataset
    df = pd.read_csv("Crop_recommendation_corrected.csv")
    df.index = range(1, len(df) + 1)  # Reset index to start from 1

    # User-selectable number of rows to display
    rows_to_show = st.selectbox("🔢 Select number of rows to display:", [10, 25, 50, 100, 200, len(df)], index=3)

    st.dataframe(df.head(rows_to_show), use_container_width=True)