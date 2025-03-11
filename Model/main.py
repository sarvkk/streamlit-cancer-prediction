import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle5 as pickle

def create_model(data):
    y=data['diagnosis']
    X=data.drop(['diagnosis'],axis=1)

    #scale the data
    scaler= StandardScaler()
    X=scaler.fit_transform(X)

    #split the data
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

    #train
    model = LogisticRegression()
    model.fit(X_train,y_train)

    #test model
    y_pred=model.predict(X_test)    
    print('Accuracy of the model:',accuracy_score(y_test,y_pred))
    print('Classification report: \n',classification_report(y_test,y_pred))

    return model,scaler


def get_clean_data():
    df=pd.read_csv('../Data/data.csv')
    df.drop(columns=['Unnamed: 32','id'] ,inplace=True)
    df = df.dropna()
    df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
    return df
    
    
def main():
    data = get_clean_data()
    model,scaler= create_model(data)
    
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)
    with open('scaler.pkl','wb') as f:
        pickle.dump(scaler,f)
if __name__=="__main__":
    main()