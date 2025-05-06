# we import all the libraries we need 
import pandas as pd
from sklearn.model_selection import train_test_split #for testing and training

from sklearn.linear_model import LogisticRegression #for the model
from sklearn.metrics import classification_report, confusion_matrix #we use these to get accurate predictions
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay #evaluation p

from sklearn.metrics import roc_auc_score #evaluation metric
import matplotlib.pyplot as plt #for visual representation
import seaborn as sns # for visual representation as well
import warnings # to ignore warnings , and live a happy life
warnings.filterwarnings("ignore")
# we load the sample data that I've created , absolutely fake data
df = pd.read_csv("student_burnout_dataset.csv")
print(df.columns)
print(df.dtypes) #to know which rows have to be mapped to numeric values
df['burnout_risk'] = df['burnout_risk'].map({'Yes':1 , 'No':0})
# this step we are just slicing the dataframe to get just what is enough for prediction , more than enough
df = df[['avg_sleep_hours','attendance_percent','social_activity_score','mental_health_rating','burnout_risk']]

# we print the first few rows of the dataset
print(df.head())

# to get more info about the dataset
print(df.info())

# now we check if we have any null values at all 
print(df.isnull().sum())

#to get a summary of the numeric values , we basically get  mean max average and also standard deviation stuff like that 
print(df.describe())

#now we will check of there are any missing values and fill them up with the mean value of that column , so the function fillna will fill the missing values with the mean of that column
df.fillna(df.mean(), inplace=True)

X =df[['avg_sleep_hours','attendance_percent','social_activity_score','mental_health_rating','burnout_risk']]
y = df['burnout_risk']
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# we split the model 80:20 , 80 for training and 20 for testing
# we create the model
model = LogisticRegression(max_iter=1000) # we set the max iterations to 1000 , so that the model can converge as much as possible
# we fit the model for the data that we have
model.fit(X_train, y_train)
# we make predictions of our very own data 
y_pred = model.predict(X_test)
# we print the predictions this model , lets see
print("Predictions: ", y_pred)
# we print the classification report , very important , accuracy of the model depends on this
print(classification_report(y_test, y_pred))
# we print the confusion matrix , very important as well , yk the most important part ever
print(confusion_matrix(y_test, y_pred))
# we plot the confusion matrix , like we should be 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(13,9)) #we just doing this to print a plot thingy and set the figure size to be big enough to see the confusion matrix 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes']) 
#now thats a heatmap , we can see various statistics in it , this will be printed on the figure that we just created
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#now we evaluate the model? ofc we have to 
#we first find the accuracy of the model
accuracy = (cm[0][0] + cm[1][1]) / cm.sum() #this adds the sum of TN and TP and divides it by the total number of predictions , so hence we get the accuracy
print(f"Accuracy: {accuracy:.2f}") #upto two decimal points

#then we find the precision of the model
precision = cm[1][1] / (cm[1][1] + cm[0][1]) #this adds the sum of TP and FP and divides it by the total number of predictions , so hence we get the precision
print(f"Precision: {precision:.2f}") #upto two decimal points

#then we find the recall of the model , can be called sensitivity as well
recall = cm[1][1] / (cm[1][1] + cm[1][0]) #this adds the sum of TP and FN and divides it by the total number of predictions , so hence we get the recall
print(f"Recall: {recall:.2f}") #upto two decimal points

#then we find the f1 score of the model , this is a very important metric for evaluating the model , VERY IMPORTANT
f1_score = 2 * (precision * recall) / (precision + recall) #this adds the sum of TP and FN and divides it by the total number of predictions , so hence we get the f1 score
print(f"F1 Score: {f1_score:.2f}") #upto two decimal points


#then we find the ROC AUC score of the model , this is a very important metric for evaluating the model
from sklearn.metrics import roc_auc_score  
roc_auc = roc_auc_score(y_test, y_pred) 
print(f"ROC AUC Score: {roc_auc:.2f}") #upto two decimal points
# we plot the ROC curve , this is a very important metric for evaluating the model
fpr, tpr, thresholds = roc_curve(y_test, y_pred) #this gives us the false positive rate , true positive rate and the thresholds for the ROC curve
plt.figure(figsize=(13, 9)) #we just doing this to print a plot thingy and set the figure size to be big enough to see the ROC curve
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc) #this plots the ROC curve
plt.plot([0, 1], [0, 1], color='red', linestyle='--') #this plots the diagonal line for the ROC curve
plt.xlim([0.0, 1.0]) #this sets the x axis limit for the ROC curve
plt.ylim([0.0, 1.05]) #this sets the y axis limit for the ROC curve
plt.xlabel('False Positive Rate') #this sets the x axis label for the ROC curve
plt.ylabel('True Positive Rate') #this sets the y axis label for the ROC curve
plt.title('Receiver Operating Characteristic') #this sets the title for the ROC curve
plt.legend(loc='lower right') #this sets the legend for the ROC curve
plt.show() #this shows the ROC curve
# we plot the ROC curve using the RocCurveDisplay class
RocCurveDisplay.from_estimator(model, X_test, y_test, name='ROC Curve', alpha=0.8) #this plots the ROC curve using the RocCurveDisplay class





