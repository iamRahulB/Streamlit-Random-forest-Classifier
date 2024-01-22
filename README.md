# Random Forest Classifier with Streamlit

This project presents a Random Forest Classifier implemented using Streamlit, an open-source app framework for Machine Learning and Data Science projects.

## Project Description

- The Random Forest Classifier in this project is trained on two types of datasets: Concentric Circles and Spiral data. These datasets are particularly challenging because the decision boundaries between classes are not linear, testing the flexibility of the model.

- A key feature of this project is the use of a meshgrid. The meshgrid allows us to visualize the decision boundaries of the classifier, providing insights into its performance and potential overfitting. Overfitting occurs when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data.

## Features

- **Interactive UI:** The Streamlit framework provides an interactive user interface, allowing users to input parameters and run the classifier in real-time.

- **Customizable Parameters:** Users can customize various parameters of the Random Forest Classifier including the number of estimators, maximum features, bootstrap option, and maximum samples.

- **Visualization:** The matplotlib library is used to plot the datasets and the decision boundaries of the classifier.

## Screenshots
![random forest classifier](https://github.com/iamRahulB/Streamlit-Random-forest-Classifier/assets/108116259/ae429d85-3cf9-4e6b-bb75-3979d57b84d1)


## How to Use
1. **Fork or Clone the Project:**
   ```
     git clone https://github.com/iamRahulB/Streamlit-Random-forest-Classifier.git
   ```
2. Navigate to the Project Directory:
   ```
   cd Streamlit-Random-forest-Classifier
   ```
3. Create a Virtual Environment (Optional but Recommended):
   ```
   python -m venv venv
   ```
4. Activate the Virtual Environment:
   1. On Windows:
     ```
     .\venv\Scripts\activate
     ```
   2. On macOS/Linux:
     ```
     source venv/bin/activate
5. Install the Required Dependencies:
   ```
   pip install -r requirements.txt
   ```
6. Run the Application Locally Using Streamlit:
   ```
   streamlit run rfc.py
   ```
   

<h2>Usage</h2>

<p>To use the web app, click on the following link: <a href="https://randomforestclassifier.streamlit.app/">Demo Link</a></p>
