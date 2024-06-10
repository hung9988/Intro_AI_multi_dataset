import pandas as pd
import numpy as np
import seaborn as sns
from category_encoders import OneHotEncoder, OrdinalEncoder, BinaryEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler 
from sklearn.ensemble import RandomForestClassifier
class PreProcessor():
    def __init__(self,dataset):
        self.dataset=dataset.replace({' ':np.nan})
        
    
    def remove_nans(self):
        self.dataset.dropna(inplace=True)
        return self
    
    def remove_duplicates(self):
        self.dataset.drop_duplicates(inplace=True)
        return self
    
    def binary_encode(self,binary_columns):
        binary_encoder = BinaryEncoder(cols=binary_columns)
        self.dataset = binary_encoder.fit_transform(self.dataset)
        return self
    
    def ordinal_encode(self,ordinal_columns):
        ordinal_encoder = OrdinalEncoder(cols=ordinal_columns)
        self.dataset = ordinal_encoder.fit_transform(self.dataset)
        return self
    
    def onehot_encode(self,categorical_columns):
        onehot_encoder = OneHotEncoder(cols=categorical_columns)
        self.dataset = onehot_encoder.fit_transform(self.dataset)
        return self
    
    def rename_label(self,label_column):
        self.dataset.rename(columns={label_column: 'churn'}, inplace=True)
        return self
    
    def split_features_labels(self):
        self.X=self.dataset.drop('churn',axis=1)
        self.y=self.dataset['churn'].astype(float)
        return self
    
    
    def scale(self,scaler_columns):
        scaler = StandardScaler()
        self.dataset[scaler_columns] = scaler.fit_transform(self.dataset[scaler_columns])
        return self
    
    def oversample(self):
        ros = RandomOverSampler(sampling_strategy='minority')
        self.split_features_labels()
        self.X, self.y = ros.fit_resample(self.X, self.y)
        self.dataset=pd.concat([self.X,self.y],axis=1)
        return self
    
    def calculate_feature_importance(self):
        clf=RandomForestClassifier()
        self.split_features_labels()
        clf.fit(self.X,self.y)
        self.feature_importances=pd.DataFrame(clf.feature_importances_,
                                                                 index=self.X.columns,
                                                                 columns=['importance']).sort_values('importance',ascending=False)
        
        return self.feature_importances
    
    def remove_features_by_importance(self,threshold):
        columns_to_drop=list(self.feature_importances[self.feature_importances['importance']<threshold].index)
        self.dataset=self.dataset.drop(columns_to_drop,axis=1)
        return self
    
        
    def nan_stats(self):
        nans = self.dataset.isna().sum().sort_values(ascending=False)
        pct = 100 * nans / self.dataset.shape[0]
        nan_stats = pd.concat([nans, pct], axis=1)
        nan_stats.columns = ['num_of_nans', 'percentage_of_nans']
        return nan_stats
    
    def plot_imbalance(self):
        
        plt.bar(['Churn','No Churn'],[self.y[self.y==1].shape[0],self.y[self.y==0].shape[0]])
    
    def plot_correlation(self):
        plt.figure(figsize=(10,10))
        sns.heatmap(self.dataset.corr(),annot=True)
        
    
    def train_test_split(self,test_size=0.2):
        return train_test_split(self.X,self.y,test_size=test_size)
    
    
    # def remove_columns(self, columns):
    #     self.dataset = self.dataset.drop(columns, axis=1)
        
    # def nans_stat(self, dataset):
    
    #     nans = dataset.isna().sum().sort_values(ascending=False)
    #     pct = 100 * nans / dataset.shape[0]
    #     nan_stats = pd.concat([nans, pct], axis=1)
    #     nan_stats.columns = ['num_of_nans', 'percentage_of_nans']
    #     return nan_stats
    
    # def drop_nans(self):
    #     self.dataset=self.dataset.replace({' ':np.nan})
    #     self.dataset.dropna(inplace=True)
        
    # def drop_duplicates(self):
    #     self.dataset.drop_duplicates(inplace=True)
        
    # def split_data_and_labels(self):
    #     X=self.dataset.drop('churn',axis=1)
    #     y=self.dataset['churn']
    #     return X,y
    
    # def plot_imbalance(self,y):
    #     plt.bar(['Churn','No Churn'],[y[y==1].shape[0],y[y==0].shape[0]])
        
    # def plot_correlation(self,dataset):
    #     plt.figure(figsize=(10,10))
    #     sns.heatmap(self.dataset.corr(),annot=True)
    
    # def encoding(self):
    #     binary_encoder = BinaryEncoder(cols=self.binary_columns)
    #     self.dataset = binary_encoder.fit_transform(self.dataset)
    #     ordinal_encoder = OrdinalEncoder(cols=self.ordinal_columns)
    #     self.dataset = ordinal_encoder.fit_transform(self.data)
    #     onehot_encoder = OneHotEncoder(cols=self.categorical_columns)
    #     self.data = onehot_encoder.fit_transform(self.data)
    
    # def scaling(self):
    #     scaler = StandardScaler()
    #     self.dataset[self.scaler_columns] = scaler.fit_transform(self.dataset[self.scaler_columns])
    
    # def base_process(self, dataset, label_column, categorical_columns=[], ordinal_columns=[], binary_columns=[], scaler_columns=[]):
    #     self.dataset = dataset
    #     self.drop_nans()
    #     self.drop_duplicates()
    #     self.dataset.rename(columns={label_column: 'churn'}, inplace=True)

    #     self.label_column = label_column
    #     self.category_columns = categorical_columns
    #     self.ordinal_columns = ordinal_columns
    #     self.binary_columns = binary_columns
    #     self.scaler_columns = scaler_columns
        
    #     self.encoding()
        
        
        
    #     return self.data