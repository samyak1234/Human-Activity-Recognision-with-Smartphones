
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score , recall_score

def main():
	st.title('Human Action Classifier')
	st.sidebar.title('Perform Parameter Variation')
	st.markdown('Predict Human Action Using mobile Sensors data 📱')

	@st.cache(persist = True)
	def load_train_data():
		data = pd.read_csv('train.csv')
		return data
	def load_test_data():
		test = pd.read_csv('test.csv')
		return test

	#plot performance data
	#@st.cache(persist = True, hash_funcs=hash())
	def plot_metrics(metrics_list):
		if 'Confusion Matrix' in metrics_list:
			st.subheader('Confusion Matrix')
			## cm plot
			plot_confusion_matrix(model, X_test, y_test, labels = class_names)
			plt.xticks(rotation=90)
			st.pyplot()
			## cm plot

		if 'ROC Curve' in metrics_list:
			st.subheader('ROC Curve')
			## ROC plot
			plot_roc_curve(model, X_test, y_test)
			st.pyplot()
			## ROC plot

		if 'Precision-Recall Curve' in metrics_list:
			st.subheader('PR Curve')
			## PR curve
			plot_precision_recall_curve(model, X_test, y_test)
			st.pyplot()
			## PR curve


	df = load_train_data()
	test = load_test_data()

	X_train = df.drop(['subject', 'Activity'], axis=1)
	y_train = df.Activity
	X_test = test.drop(['subject', 'Activity'], axis=1)
	y_test = test.Activity
	class_names = ['SITTING', 'STANDING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
	
	st.sidebar.subheader('Choose Classifier')
	classifier = st.sidebar.selectbox('Classifier', ('Logistic Regression (LR)', 'SVM' ,'Random Forest (RF)'))

	if classifier == 'Logistic Regression (LR)' :
		st.sidebar.subheader('Model Hyperparameter')
		C = st.sidebar.number_input('C (Regularization Parameter)', .01, 10.0, step=.01, key= 'C')

	
	if classifier == 'SVM' :
		C = st.sidebar.number_input('C (Regularization Parameter)', .01, 10.0, step=.01, key= 'C')
		kernel = st.sidebar.radio('Kernel', ('rbf', 'linear'), key='kernel')
		gamma = st.number_input('Gamma', .01, 1.0, step=.01, key= 'gamma')	

		metrics = st.sidebar.multiselect('What matrix to plot?', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

		if st.sidebar.button('Classify', key = 'classify'):
			st.subheader('SVM Results')
			model = SVC(C=C, kernel=kernel, gamma=gamma)
			model.fit(X_train, y_train)
			accuracy = model.score(X_test, y_test)
			y_pred = model.predict(X_test)
			st.write('Accuracy: ', accuracy.round(2))
			st.write('Precision: ', precision_score(y_test, y_pred, labels=class_names, average = 'macro').round(2))
			st.write('Recall: ', recall_score(y_test, y_pred, labels=class_names, average = 'macro').round(2))
			plot_metrics(metrics)


	if st.sidebar.checkbox('Show Raw data', False):
		st.subheader('Mobile Sensors Dataset')
		st.write(df)











if __name__ == '__main__':
    main()