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
    Attrition_Flag=st.number_input(label='Attrition_Flag',step=0.001,format="%.6f")
    Customer_Age=st.number_input(label='Customer_Age',step=0.001,format="%.6f")
    Gender=st.number_input(label='Gender',step=0.01,format="%.2f")
    Dependent_count=st.number_input(label='Dependent_count',step=0.01,format="%.2f")
    Education_Level=st.number_input(label='Education_Level',step=0.01,format="%.2f")
    Marital_Status=st.number_input(label='Marital_Status',step=0.01,format="%.6f")
    Income_Category=st.number_input(label='Income_Category',step=0.01,format="%.6f")
    Card_Category=st.number_input(label='Card_Category',step=0.1,format="%.6f")
    Months_on_book=st.number_input(label='Months_on_book',step=0.1,format="%.6f")
    Total_Relationship_Count=st.number_input(label='Total_Relationship_Count',step=0.1,format="%.6f")
    Months_Inactive_12_mon=st.number_input(label='Months_Inactive_12_mon',step=1)
    Contacts_Count_12_mon=st.number_input(label='Contacts_Count_12_mon',step=1)
    Credit_Limit=st.number_input(label='Credit Limit',step=0.1,format="%.1f")
    Total_Revolving_Bal=st.number_input(label='Total_Revolving_Bal',step=0.01,format="%.6f")
    Avg_Open_To_Buy=st.number_input(label='Avg_Open_To_Buy',step=0.01,format="%.6f")
    Total_Amt_Chng_Q4_Q1=st.number_input(label='Total_Amt_Chng_Q4_Q1',step=0.01,format="%.6f")
    Total_Trans_Amt=st.number_input(label='Total_Trans_Amt',step=1)
    Total_Trans_Ct=st.number_input(label='Total_Trans_Ct',step=1)
    Total_Ct_Chng_Q4_Q1=st.number_input(label='Total_Ct_Chng_Q4_Q1',step=1)
    Avg_Utilization_Ratio=st.number_input(label='Avg_Utilization_Ratio',step=1)

    data=[[Attrition_Flag,Customer_Age,Gender,Dependent_count,Education_Level,Marital_Status,Income_Category,Card_Category,Months_on_book,Total_Relationship_Count,Months_Inactive_12_mon,Contacts_Count_12_mon,Credit_Limit,Total_Revolving_Bal,Avg_Open_To_Buy,Total_Amt_Chng_Q4_Q1,Total_Trans_Amt,Total_Trans_Ct,Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio]]

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


