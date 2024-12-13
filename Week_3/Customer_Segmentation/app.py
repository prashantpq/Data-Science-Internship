from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Data/Clustered_Customer_Data.csv")
# st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Prediction of Cluster")

with st.form("my_form"):
    Credit_Limit=st.number_input(label='Credit Limit',step=0.1,format="%.1f")
    Total_Revolving_Bal=st.number_input(label='Total Revolving Bal',step=0.01,format="%.6f")
    Avg_Open_To_Buy=st.number_input(label='Avg Open To Buy',step=0.01,format="%.6f")
    Total_Trans_Amt=st.number_input(label='Total Trans Amt',step=1)
    Total_Trans_Ct=st.number_input(label='Total Trans Ct',step=1)

    data=[[Credit_Limit,Total_Revolving_Bal,Avg_Open_To_Buy,Total_Trans_Amt,Total_Trans_Ct]]

    submitted = st.form_submit_button("Submit")
    
if submitted:
    clust = loaded_model.predict(data)[0]
    st.write(f'Data Belongs to Cluster {clust}')

    cluster_df1 = df[df['Cluster'] == clust]
    plt.rcParams["figure.figsize"] = (20, 3)

    for c in cluster_df1.drop(['Cluster'], axis=1):
        fig, ax = plt.subplots()
        sns.histplot(cluster_df1[c], ax=ax, kde=True, bins=20)  # Using sns.histplot for clarity
        ax.set_title(f"Distribution of {c} in Cluster {clust}")
        st.pyplot(fig)


