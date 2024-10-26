import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle

st.set_page_config(page_title="MsbA-Pred", layout="wide")

def main_page():
    # Add logo image at the top (centered)
    logo_image = Image.open('LOGO.png')
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{img}" style="max-width: 50%; height: auto;">
        </div>
        """.format(img=base64.b64encode(open('LOGO.png', "rb").read()).decode()), unsafe_allow_html=True
    )

    # Add space after the logo
    st.markdown("<br>", unsafe_allow_html=True)  # Add a line break

    # Create title and subtitle
    html_temp = """
        <div style="background-color:teal">
        <h2 style="font-family:arial;color:white;text-align:center;">MsbA-Pred</h2>
        <h4 style="font-family:arial;color:white;text-align:center;">Artificial Intelligence Based Bioactivity Prediction Web-App</h4>       
        </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Add a marquee after the logo
    st.markdown(
        """
        <div style="background-color: yellow; padding: 0.1px;">
            <marquee behavior="scroll" direction="left" style="font-size:18px; color:blue;">
                Welcome to MsbA-Pred! This application allows you to predict the bioactivity towards inhibiting the "MsbA - ATP-dependent lipid A-core flippase" with respect to quinoline derivatives.  
            </marquee>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create two columns with specified widths
    col1, col2 = st.columns([1, 4])  # Column 1 is narrower than Column 2

    # Left column: Sidebar for file upload
    with col1:
        st.header('1. Upload your data')
        uploaded_file = st.file_uploader("Upload your input file as smile notation", type=['txt'])
        st.markdown("""[Example input file](https://raw.githubusercontent.com/MohanChemoinformaticsLab/MsbAPred/main/Sample_Smiles_File.txt)""")

        if st.button('Predict'):
            if uploaded_file is not None:
                # Load the uploaded data
                load_data = pd.read_table(uploaded_file, sep=' ', header=None, names=['Notation', 'Identity'])  # Read the uploaded data with specified column names
                load_data.reset_index(drop=True, inplace=True)
                load_data.index = load_data.index.astype(int) + 1  # Convert index to int and set to start from 1
                load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

                # Save data to session state for results page
                st.session_state['load_data'] = load_data
                st.session_state['show_results'] = True

            else:
                st.error("Error! Please upload a file.")

    # Right column: MsBAPred image and details
    with col2:
        # Logo image for MsBAPred
        image = Image.open('MsbAPred.png')
        st.image(image, use_column_width=True)

        # Application details
        st.markdown("""        
        # Application Details

        Here are some details about the app and its functionality:

            * Input  : Text file containing smile notation.
            * Output : Predictions based on the input data.
            * Usage  : Click on the button to predict after uploading your data.
            * Credits: Developed by Anupama Binoy, Dr. C. Gopi Mohan, Ratul Bhowmik, and Preena S Parvathy.

            **Note: Ensure that your input file is formatted correctly to avoid errors during processing.

        ***Credits***

        - Author affiliations: *Bioinformatics and Computational Biology Lab, Amrita School of Nanosciences and Molecular Medicine, Amrita Vishwa Vidyapeetham, Kochi*
        - Descriptor calculated using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Read the Paper]](https://doi.org/10.1002/jcc.21707).
        """)

    # Create a visual separation 
    st.markdown("---")  # Horizontal line for separation

# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
     # Check if molecule.smi exists before attempting to delete
    if os.path.exists('molecule.smi'):
        os.remove('molecule.smi')
    else:
        st.warning("The file 'molecule.smi' does not exist. Skipping deletion.")

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    # Reads in saved regression model
    load_model = pickle.load(open('msba_rf.pkl', 'rb'))
    
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    
    st.header('**Prediction output**')

    # Create a DataFrame for predictions
    prediction_output = pd.Series(prediction, name='pIC50')
    
    # Get the molecule names without extra indices
    molecule_name = st.session_state.load_data['Identity'].reset_index(drop=True)  # Reset index
    
    # Create a DataFrame combining molecule names and predictions
    df = pd.DataFrame({
        'molecule_name': molecule_name,
        'pIC50': prediction_output.reset_index(drop=True)  # Reset index for predictions as well
    })
    
    # Display the DataFrame in a table format
    df.index = df.index + 1
    st.write(df)  # Streamlit automatically renders DataFrame as a table

    # Provide a download link for the predictions
    st.markdown(filedownload(df), unsafe_allow_html=True)



# Function to display the results page
def results_page():
    load_data = st.session_state.load_data

    st.header('**Input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Apply trained model to make prediction on query compounds
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('descriptors_output.csv')
    desc.reset_index(drop=True, inplace=True)
    desc.index += 1
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header('**Subset of descriptors from previously built models**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    desc_subset.reset_index(drop=True, inplace=True)
    desc_subset.index += 1
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)

# Main loop
if 'show_results' not in st.session_state:
    st.session_state['show_results'] = False

if st.session_state.show_results:
    results_page()  # Show the results page
else:
    main_page()  # Show the main page initially
