import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as keras
from sklearn.preprocessing import MinMaxScaler


# give a title to the app
st.title('Influx Detector: Analyse Your Data')
st.text("Upload your CSV file to get started with influx detection.")
st.text("Our tool will analyse your data for any significant patterns or unusual activities.")

model = keras.models.load_model('/Users/nkosimate/Desktop/MSc LSTM-AE working.keras')
#This function is to preprocess the dataframe


def dataset_to_sequence(data):
    data = data.drop(df.columns[0], axis=1)
    data = data.drop(columns=['TIME','ARAMCO BLOCK POSITION (SURFACE)','ARAMCO FLOW IN (SURFACE)','ARAMCO DEPTH (SURFACE)','ARAMCO RPM (SURFACE)'])
    
    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Convert the data into sequences
    timesteps = 5  # Set the sequence length
    def create_sequences(data, timesteps):
        sequences = []
        for i in range(len(data) - timesteps + 1):
            seq = data[i:i+timesteps]
            sequences.append(seq)
        return np.array(sequences)

    X = create_sequences(data_scaled, timesteps)
    return(X)

#usign the model
def use_model(df,X,autoencoder):
    data = df.drop(df.columns[0], axis=1)
    data = data.drop(columns=['TIME','ARAMCO BLOCK POSITION (SURFACE)','ARAMCO FLOW IN (SURFACE)','ARAMCO DEPTH (SURFACE)','ARAMCO RPM (SURFACE)'])
    
    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    reconstructed_sequences = autoencoder.predict(X)
    # Reshape back to 2D (since we used sequences)
    n_samples = X.shape[0]
    n_timesteps = X.shape[1]
    n_features = X.shape[2]
    # Flatten the sequences into 2D
    X_flat = X.reshape(-1, n_features)
    reconstructed_flat = reconstructed_sequences.reshape(-1, n_features)

    # Inverse transform to get back to original scale
    data_reconstructed = scaler.inverse_transform(reconstructed_flat)
    data_actual = scaler.inverse_transform(X_flat)

    # Convert back to original shape and add to DataFrame
    df_actual = pd.DataFrame(data_actual, columns=data.columns)
    df_reconstructed = pd.DataFrame(data_reconstructed, columns=data.columns)
    return(df_actual,df_reconstructed)

def plotgraph1(df_a,df_r):
     # Clear the current figure
    plt.figure(figsize=(5, 3))
    plt.plot(df_a['ARAMCO FLOW OUT (SURFACE)'], label='Actual')
    plt.plot(df_r['ARAMCO FLOW OUT (SURFACE)'], label='Predicted')

    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Actual FLOW OUT vs Predicted FLOW OUT')
    plt.legend()
    # Return the figure
    return plt.gcf()


def plotgraph2(df_a,df_r):
     # Clear the current figure
    plt.figure(figsize=(5, 3))  
    plt.plot(df_a['ARAMCO TVA (SURFACE)'], label='Actual')
    plt.plot(df_r['ARAMCO TVA (SURFACE)'], label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Actual TVA vs Predicted TVA')
    plt.legend()
    # Return the figure
    return plt.gcf()

def plotgraph3(df_a,df_r):
     # Clear the current figure
    plt.figure(figsize=(5, 3))
    plt.plot(df_a['ARAMCO SPP (SURFACE)'], label='Actual')
    plt.plot(df_r['ARAMCO SPP (SURFACE)'], label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Actual SPP vs Predicted SPP')
    plt.legend()
    # Return the figure
    return plt.gcf()

def plotgraph4(df_a,df_r):
     # Clear the current figure
    plt.figure(figsize=(5, 3))
    plt.plot(df_a['ARAMCO ROP (SURFACE)'], label='Actual')
    plt.plot(df_r['ARAMCO ROP (SURFACE)'], label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Actual ROP vs Predicted ROP')
    plt.legend()
    # Return the figure
    return plt.gcf()

def show_possible_kick(drilling_df):
    drilling_df['flow_rate_prev'] = drilling_df['ARAMCO FLOW OUT (SURFACE)'].shift(2)
    drilling_df['TVA_prev'] = drilling_df['ARAMCO TVA (SURFACE)'].shift(2)
    drilling_df['ROP_prev'] = drilling_df['ARAMCO ROP (SURFACE)'].shift(2)
    drilling_df['SPP_prev'] = drilling_df['ARAMCO SPP (SURFACE)'].shift(2)
    drilling_df['WOB_prev'] = drilling_df['ARAMCO WOB (SURFACE)'].shift(2)
    drilling_df['Flow_in_prev'] = drilling_df['ARAMCO FLOW IN (SURFACE)'].shift(2)
    drilling_df['Hookload_prev'] = drilling_df['ARAMCO HOOKLOAD (SURFACE)'].shift(2)

    drilling_df['flow_rate_change'] = ((drilling_df['ARAMCO FLOW OUT (SURFACE)']- drilling_df['flow_rate_prev'])/drilling_df['flow_rate_prev']) * 100
    drilling_df['TVA_change'] = (drilling_df['ARAMCO TVA (SURFACE)']-drilling_df['TVA_prev'])
    drilling_df['ROP_change'] = ((drilling_df['ARAMCO ROP (SURFACE)']-drilling_df['ROP_prev'])/drilling_df['ROP_prev']) *100
    drilling_df['SPP_change'] = ((drilling_df['ARAMCO SPP (SURFACE)'] - drilling_df['SPP_prev'])/drilling_df['SPP_prev']) *100
    drilling_df['WOB_change'] = ((drilling_df['ARAMCO WOB (SURFACE)'] - drilling_df['WOB_prev'])/drilling_df['WOB_prev']) *100
    drilling_df['Flow_in_change'] = ((drilling_df['ARAMCO FLOW IN (SURFACE)'] - drilling_df['Flow_in_prev'])/drilling_df['Flow_in_prev']) *100
    drilling_df['Hookload_change'] = ((drilling_df['ARAMCO HOOKLOAD (SURFACE)'] - drilling_df['Hookload_prev'])/drilling_df['Hookload_prev']) *100

    # Initialize the score
    drilling_df['score'] = 0

    # Calculate the score based on the given conditions
    drilling_df['score'] += 90 * ((drilling_df['flow_rate_change'] >= 15) & (drilling_df['TVA_change'] >= 5))
    drilling_df['score'] += 40 * ((drilling_df['flow_rate_change'] >= 15) | (drilling_df['TVA_change'] >= 5))
    drilling_df['score'] += 20 * (drilling_df['ROP_change'] >= 50)
    drilling_df['score'] += 40 * (drilling_df['ROP_change'] >= 50)
    drilling_df['score'] += 20 * ((drilling_df['SPP_change'] >= 10) | (drilling_df['SPP_change'] <= -10))
    drilling_df['score'] += 20 * ((drilling_df['WOB_change'] >= 20) | (drilling_df['WOB_change'] <= -20))
    drilling_df['score'] += 20 * (drilling_df['Flow_in_change'] >= 10)
    drilling_df['score'] += 20 * (drilling_df['Hookload_change'] >= 5)

    # Assign possible kick based on the score
    drilling_df['possible_kick'] = 0
    drilling_df.loc[drilling_df['score'] >= 70, 'possible_kick'] = 2
    drilling_df.loc[(drilling_df['score'] >= 40) & (drilling_df['score'] < 70), 'possible_kick'] = 1

    possible_kicks2 = drilling_df[drilling_df['possible_kick']==2]
    possible_kicks2 = possible_kicks2.reset_index(drop = True)
    possible_kicks1 = drilling_df[drilling_df['possible_kick']==1]
    possible_kicks1 = possible_kicks1.reset_index(drop = True)
    return(possible_kicks2,possible_kicks1)

# Upload CSV file
st.text("Select a CSV file from your computer to begin the analysis")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Read the file into a DataFrame
    df = pd.read_csv(uploaded_file)
    if st.button('Analyse'):
        # Display the DataFrame
        st.write("DataFrame Preview:")
        # st.dataframe(df)
        # Display some statistics about the DataFrame
        st.write("Summary Statistics:")
        st.write(df.describe())
        # Display the first few rows of the DataFrame
        st.write("First 5 Rows:")
        st.write(df.head(5))
        X = dataset_to_sequence(df)
        dfa,dfr = use_model(df,X,model)
        plotgraph3(dfa,dfr)
        plotgraph4(dfa,dfr)
        # Layout with two columns
        col1, col2 = st.columns(2)
        # Plot graph 1 in the first column
        with col1:
            fig1 = plotgraph1(dfa, dfr)
            st.pyplot(fig1)

        # Plot graph 2 in the second column
        with col2:
            fig2 = plotgraph2(dfa, dfr)
            st.pyplot(fig2)
        
        col3, col4 = st.columns(2)
        # Plot graph 1 in the first column
        with col3:
            fig3 = plotgraph3(dfa, dfr)
            st.pyplot(fig3)

        # Plot graph 2 in the second column
        with col4:
            fig4 = plotgraph4(dfa, dfr)
            st.pyplot(fig3)


        
    if st.button('Show Kicks'):
        kick2,kick1 = show_possible_kick(df)
        st.write("Needs Immediate attention")
        st.write(kick2)
        st.write('Needs some checking')
        st.write(kick1)

    

