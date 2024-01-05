import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(initial_sidebar_state="expanded")

st.title("Random Forest Classifier")

with st.sidebar:
    st.title("Random Forest Classifier")
    dataset=st.selectbox("Choose Dataset",("Two spirals","Concentric Circles"))
    
    if dataset=="Concentric Circles" :
        df=pd.read_csv("concertriccir2.csv")

    if dataset=="Two spirals" :
        df=pd.read_csv("twoSpirals.csv")


def create_meshgrid():
    x_range = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    y_range = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    grid_x, grid_y = np.meshgrid(x_range, y_range)

    grid_array = np.array([grid_x.ravel(), grid_y.ravel()]).T

    return grid_x, grid_y, grid_array

X=df.iloc[:,0:2].values
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

fig, ax=plt.subplots()
ax.scatter(X.T[0],X.T[1],c=y)


col1,col2=st.columns([10,5])
with col1: 
    figures=st.pyplot(fig)

with col2:
    data=st.dataframe(df.head(8))

shapes=st.write("Overall DataFrame Shape",df.shape,)

with st.sidebar:
    estimators=int(st.number_input("N Estimators"))
    
    max_features=st.selectbox("Max Features",('auto', 'sqrt','log2','manual'))
    if max_features=="manual":
        max_features=int(st.number_input("Input Max Feat."))
    
    bootstrap = st.sidebar.selectbox(
    'Bootstrap',
    ('True', 'False')
)
   



max_sample = st.sidebar.slider('Max Samples', 1, X_train.shape[0], 1,key="1236")


if st.sidebar.button('Run Algorithm'):

    figures.empty()
    data.empty()
    
   if estimators==0:
        estimators=100
   boot=bool(bootstrap)

    if max_sample==1:
        st.write("Please increase max samples  to get better results, Selected sample size:",max_sample)
    classifier = RandomForestClassifier(n_estimators=estimators,random_state=42,bootstrap=boot,max_samples=max_sample,max_features=max_features)
    classifier.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)

    grid_x, grid_y, grid_array = create_meshgrid()
    predicted_labels = classifier.predict(grid_array)

    ax.contourf(grid_x, grid_y, predicted_labels.reshape(grid_x.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Column 1")
    plt.ylabel("Column 2")

    
    st.header("Accuracy - " + str(round(accuracy_score(y_test, y_predicted), 2)))

    original_plot = st.pyplot(fig)



    
