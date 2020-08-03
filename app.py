'''
'''
'@host: samyak jain'
'mail: samyaj@iitk.ac.in'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score , recall_score
from sklearn.manifold import TSNE
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
	
	#visualisation
	#st.sidebar.subheader('Visualise data in 2D')
	st.cache(persist= True)
	if st.sidebar.button('Visualize Data in 2D', key = 't-SNE'):
		st.subheader('Dimensionality reduction to 2D using t-SNE')
		X_for_tsne = df.drop(['subject', 'Activity'], axis = 1)
		tsne = TSNE(random_state = 42, n_components=2, verbose=1, perplexity=50, n_iter=1000).fit_transform(X_for_tsne)
		plt.figure(figsize=(12,8))
		sns.scatterplot(x =tsne[:, 0], y = tsne[:, 1], hue = df['Activity'],palette='bright')
		st.pyplot()

	#choose classifier
	st.sidebar.subheader('Choose Classifier')
	classifier = st.sidebar.selectbox('Classifier', ('Logistic Regression (LR)', 'SVM' ,'Random Forest (RF)'))

	############################################################################################
	if classifier == 'Logistic Regression (LR)' :
		st.sidebar.subheader('Model Hyperparameter')
		C = st.sidebar.number_input('C (Regularization Parameter)', .01, 10.0, step=.01, key= 'C')
		penalty = st.sidebar.radio('Penalty Function', ('l2', 'l1', 'elasticnet','none'), key='penalty')
		l1_ratio = st.sidebar.number_input('l1 to l2 ratio  (if using elasticnet)',.01, .99, step = .01, key = 'l1 ratio')
		#matrix to plot
		metrics = st.sidebar.multiselect('What matrix to plot?', ('Confusion Matrix',))
		# button for running the model
		if st.sidebar.button('Classify', key = 'classify_lr'):
			st.subheader('LR Results')
			model = LogisticRegression(C=C, penalty=penalty, solver='saga', l1_ratio=l1_ratio)
			model.fit(X_train, y_train)
			accuracy = model.score(X_test, y_test)
			y_pred = model.predict(X_test)
			st.write('Accuracy: ', accuracy.round(2))
			st.write('Precision: ', precision_score(y_test, y_pred, labels=class_names, average = 'macro').round(2))
			st.write('Recall: ', recall_score(y_test, y_pred, labels=class_names, average = 'macro').round(2))
			plot_metrics(metrics)

	############################################################################################
	
	if classifier == 'SVM' :
		C = st.sidebar.number_input('C (Regularization Parameter)', .01, 10.0, step=.01, key= 'C')
		kernel = st.sidebar.radio('Kernel', ('rbf', 'linear'), key='kernel')
		gamma = st.sidebar.number_input('Gamma', .01, 1.0, step=.01, key= 'gamma')	

		metrics = st.sidebar.multiselect('What matrix to plot?', ('Confusion Matrix',))
		# button nfor running the model
		if st.sidebar.button('Classify', key = 'classify_svm'):
			st.subheader('SVM Results')
			model = SVC(C=C, kernel=kernel, gamma=gamma)
			model.fit(X_train, y_train)
			accuracy = model.score(X_test, y_test)
			y_pred = model.predict(X_test)
			st.write('Accuracy: ', accuracy.round(2))
			st.write('Precision: ', precision_score(y_test, y_pred, labels=class_names, average = 'macro').round(2))
			st.write('Recall: ', recall_score(y_test, y_pred, labels=class_names, average = 'macro').round(2))
			plot_metrics(metrics)

	##################################################################################

	if classifier == 'Random Forest (RF)' :
		n_estimators = st.sidebar.number_input('Number of Trees', 1, 200, step=1, key= 'n_estimators')
		max_depth = st.sidebar.number_input('The maximum depth of the tree', min_value = 2, max_value =20, step=1, key='max_depth')
		
		metrics = st.sidebar.multiselect('What matrix to plot?', ('Confusion Matrix',))

		if st.sidebar.button('Classify', key = 'classify_rf'):
			st.subheader('Random Forest Results')
			model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
			model.fit(X_train, y_train)
			accuracy = model.score(X_test, y_test)
			y_pred = model.predict(X_test)
			st.write('Accuracy: ', accuracy.round(2))
			st.write('Precision: ', precision_score(y_test, y_pred, labels=class_names, average = 'macro').round(2))
			st.write('Recall: ', recall_score(y_test, y_pred, labels=class_names, average = 'macro').round(2))
			plot_metrics(metrics)

	##################################################################################
	#raw data
	if st.sidebar.checkbox('Show Raw data', False):
		st.subheader('Mobile Sensors Dataset')
		st.write(df)




if __name__ == '__main__':
    main()
