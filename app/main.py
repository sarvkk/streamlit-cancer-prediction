import os
import streamlit as st
import pickle5 as pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def get_clean_data():
    """Load and clean the breast cancer data, trying multiple file paths."""
    # List all possible file paths in order of preference
    data_paths = [
        'data.csv',               # In app directory (for Streamlit Cloud)
        'app/data.csv',           # Alternative app directory path
        '../Data/data.csv',       # Local development path
        './Data/data.csv',        # Alternative local path
    ]
    
    # Try each path until one works
    for path in data_paths:
        try:
            # Try to read the file without debug messages
            df = pd.read_csv(path)
            
            # Process the data
            df.drop(columns=['Unnamed: 32','id'], inplace=True)
            df = df.dropna()
            df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
            return df
        except FileNotFoundError:
            continue
    
    # If no path worked, show error and stop
    st.error("Cannot find data file. Please make sure 'data.csv' is available in the repository.")
    st.stop()

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'],axis=1)
    y = data['diagnosis']
 
    scaled_dict={}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value-min_val) / (max_val-min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):
  
    input_data = get_scaled_values(input_data)
  
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
          r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
      polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
    )
  
    return fig

def add_sidebar():
    st.sidebar.header('Cell Nuclei Details')
    data=get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    input_dict ={}

    for label,key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def add_predictions(input_data):
    try:
        model = pickle.load(open('model.pkl','rb'))
        scaler = pickle.load(open('scaler.pkl','rb'))
    except FileNotFoundError:
        # Try alternative path
        model = pickle.load(open('Model/model.pkl','rb'))
        scaler = pickle.load(open('Model/scaler.pkl','rb'))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    st.subheader("Cell Cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0]== 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app may assist professionals in making a diagnosis, but shouldn't be used as a substitute for a professional diagnosis.")

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction",
        page_icon=':female-doctor:',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    try:
        with open('assets/styles.css') as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback 
        st.markdown("""
        <style>
        .diagnosis { font-weight: bold; padding: 8px 16px; border-radius: 4px; display: inline-block; }
        .diagnosis.benign { background-color: #5cb85c; color: white; }
        .diagnosis.malicious { background-color: #d9534f; color: white; }
        </style>
        """, unsafe_allow_html=True)
    
    input_data=add_sidebar()
    
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("""
        This educational tool demonstrates how machine learning can help classify cell nuclei characteristics as benign or malignant. 
        
        The model has been trained on the Wisconsin Breast Cancer dataset and analyzes 30 features extracted from digitized images of fine needle aspirates (FNA) of breast masses.
        
        Use the sliders in the sidebar to adjust cell measurements and observe how different parameters affect the prediction. The radar chart visualizes the relationship between different measurements.
        
        **Note:** This is a demonstration only and not intended for medical diagnosis. Always consult healthcare professionals for medical advice.
        """)

    col1, col2= st.columns([4,1])
    
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)


if __name__ == '__main__':
    main()
