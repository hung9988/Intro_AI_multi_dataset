import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, classification_report
from sklearn.metrics import f1_score
import pickle
import matplotlib.pyplot as plt 
import joblib

from sklearn.ensemble import VotingClassifier

class Model_selection():
    def __init__(self):
        pass   

    def rf_objective(self,trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
        rf.fit(self.X_train,self.y_train)
    
        return cross_val_score(rf,self.X_train,self.y_train,cv=5).mean()

    def boost_objective(self,trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
        max_depth = trial.suggest_int('max_depth', 4, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        boost=GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
        boost.fit(self.X_train,self.y_train)
        return cross_val_score(boost,self.X_train,self.y_train,cv=5).mean()

    def mlp_objective(self,trial):

        num_layers = trial.suggest_int('num_layers', 1, 3)
        layer1_size = trial.suggest_int('layer1_size', 32, 64)
        layer2_size = trial.suggest_int('layer2_size', 32, 64)
        activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
        solver = trial.suggest_categorical('solver', ['adam'])
        alpha = trial.suggest_float('alpha', 0.0001, 0.1)
        
        hidden_layer_sizes = [layer1_size,layer2_size][:num_layers]
        mlp=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation=activation,solver=solver,alpha=alpha,max_iter=10000, early_stopping=True)
        mlp.fit(self.X_train,self.y_train)

        return cross_val_score(mlp,self.X_train,self.y_train,cv=5).mean()
    
    def svm_objective(self,trial):
        C = trial.suggest_float('C', 0.1, 10)
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        svm=SVC(C=C,kernel=kernel,gamma=gamma)
        svm.fit(self.X_train,self.y_train)
        return cross_val_score(svm,self.X_train,self.y_train,cv=5).mean()
    
    def knn_objective(self,trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 10)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        knn=KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights)
        knn.fit(self.X_train,self.y_train)
        return cross_val_score(knn,self.X_train,self.y_train,cv=5).mean()
    
    def lr_objective(self,trial):
        C = trial.suggest_float('C', 0.1, 10)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
        lr=LogisticRegression(C=C,solver=solver)
        lr.fit(self.X_train,self.y_train)
        return cross_val_score(lr,self.X_train,self.y_train,cv=5).mean()
    


        
        
        
    
    def train(self, X_train, y_train, X_test, y_test, n_trials_=100):
        self.X_train = X_train  
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(self.rf_objective, n_trials=n_trials_)
        rf_best_params = study_rf.best_params
        self.rf_best_params = rf_best_params
        
        study_mlp = optuna.create_study(direction='maximize')
        study_mlp.optimize(self.mlp_objective, n_trials=n_trials_)
        nn_best_params = study_mlp.best_params
        self.nn_best_params = nn_best_params
        
        study_gb = optuna.create_study(direction='maximize')
        study_gb.optimize(self.boost_objective, n_trials=n_trials_)
        gb_best_params = study_gb.best_params
        self.gb_best_params = gb_best_params
        
        
        
        study_svm = optuna.create_study(direction='maximize')
        study_svm.optimize(self.svm_objective, n_trials=n_trials_)
        svm_best_params = study_svm.best_params
        self.svm_best_params = svm_best_params
        
        
        study_knn = optuna.create_study(direction='maximize')
        study_knn.optimize(self.knn_objective, n_trials=n_trials_)
        knn_best_params = study_knn.best_params
        self.knn_best_params = knn_best_params
        
        study_lr = optuna.create_study(direction='maximize')
        study_lr.optimize(self.lr_objective, n_trials=n_trials_)
        lr_best_params = study_lr.best_params
        self.lr_best_params = lr_best_params
        
        
        rf=RandomForestClassifier(n_estimators=rf_best_params['n_estimators'],
                                  max_depth=rf_best_params['max_depth'],
                                  min_samples_split=rf_best_params['min_samples_split'],
                                  min_samples_leaf=rf_best_params['min_samples_leaf'])
        nn=MLPClassifier(hidden_layer_sizes=[nn_best_params['layer1_size'],nn_best_params['layer2_size']][:nn_best_params['num_layers']],
                         activation=nn_best_params['activation'],
                         solver=nn_best_params['solver'],
                         alpha=nn_best_params['alpha'],
                         max_iter=10000, early_stopping=True)
        
        gb=GradientBoostingClassifier(n_estimators=gb_best_params['n_estimators'],
                                        learning_rate=gb_best_params['learning_rate'],
                                        max_depth=gb_best_params['max_depth'],
                                        min_samples_split=gb_best_params['min_samples_split'],
                                        min_samples_leaf=gb_best_params['min_samples_leaf'])
        
        svm=SVC(C=svm_best_params['C'],
                kernel=svm_best_params['kernel'],
                gamma=svm_best_params['gamma'],
                probability=True)
        
        knn=KNeighborsClassifier(n_neighbors=knn_best_params['n_neighbors'],
                                    weights=knn_best_params['weights'])
        
        lr=LogisticRegression(C=lr_best_params['C'],
                                solver=lr_best_params['solver'])
        
        
        
        
        voting = VotingClassifier(estimators=[('rf', rf), ('nn', nn), ('gb', gb), ('knn', knn)], voting='soft')
        voting.fit(self.X_train,self.y_train)
        

        
        
        
        
        rf.fit(self.X_train,self.y_train)
        nn.fit(self.X_train,self.y_train)
        gb.fit(self.X_train,self.y_train)
        svm.fit(self.X_train,self.y_train)
        knn.fit(self.X_train,self.y_train)
        lr.fit(self.X_train,self.y_train)
        
        
        self.rf_model = rf
        self.nn_model = nn
        self.gb_model = gb
        self.svm_model = svm
        self.knn_model = knn
        self.lr_model = lr
        self.voting_model=voting
        
        
            # Save the models
        joblib.dump(self.rf_model, 'rf_model.joblib')
        joblib.dump(self.nn_model, 'nn_model.joblib')
        

    
    def classification_reports(self):
        print("Random Forest")
        print(classification_report(self.y_test,self.rf_model.predict(self.X_test)))
        print("Neural Network")
        print(classification_report(self.y_test,self.nn_model.predict(self.X_test)))
        print("Gradient Boosting")
        print(classification_report(self.y_test,self.gb_model.predict(self.X_test)))
        print("SVM")
        print(classification_report(self.y_test,self.svm_model.predict(self.X_test)))
        print("KNN")
        print(classification_report(self.y_test,self.knn_model.predict(self.X_test)))
        print("Logistic Regression")
        print(classification_report(self.y_test,self.lr_model.predict(self.X_test)))
        print("Voting Classifier")
        print(classification_report(self.y_test,self.voting_model.predict(self.X_test)))
        
        
        

    def roc_curve(self):

        ax = plt.gca()
        rf_disp = RocCurveDisplay.from_estimator(self.rf_model, self.X_test, self.y_test, ax=ax)
        mlp_disp = RocCurveDisplay.from_estimator(self.nn_model, self.X_test, self.y_test, ax=ax)
        gb_disp = RocCurveDisplay.from_estimator(self.gb_model, self.X_test, self.y_test, ax=ax)
        svm_disp = RocCurveDisplay.from_estimator(self.svm_model, self.X_test, self.y_test, ax=ax)
        knn_disp = RocCurveDisplay.from_estimator(self.knn_model, self.X_test, self.y_test, ax=ax)
        lr_disp = RocCurveDisplay.from_estimator(self.lr_model, self.X_test, self.y_test, ax=ax)
        voting_disp = RocCurveDisplay.from_estimator(self.voting_model, self.X_test, self.y_test, ax=ax)
        
        self.roc_plot = ax
        plt.show()
        
    def precision_recall_curve(self):
      
        ax = plt.gca()
        rf_disp = PrecisionRecallDisplay.from_estimator(self.rf_model, self.X_test, self.y_test, ax=ax)
        mlp_disp = PrecisionRecallDisplay.from_estimator(self.nn_model, self.X_test, self.y_test, ax=ax)
        gb_disp = PrecisionRecallDisplay.from_estimator(self.gb_model, self.X_test, self.y_test, ax=ax)
        svm_disp = PrecisionRecallDisplay.from_estimator(self.svm_model, self.X_test, self.y_test, ax=ax)
        knn_disp = PrecisionRecallDisplay.from_estimator(self.knn_model, self.X_test, self.y_test, ax=ax)
        lr_disp = PrecisionRecallDisplay.from_estimator(self.lr_model, self.X_test, self.y_test, ax=ax)
        voting_disp = PrecisionRecallDisplay.from_estimator(self.voting_model, self.X_test, self.y_test, ax=ax)
        self.pr_plot = ax
        plt.show()
        
    def choose_best_model(self, model):
        self.best_model = model
                
    def calculate_max_f1_threshold_for_best_model(self):
        
        y_pred_rf = self.best_model.predict(self.X_test)
        f1_scores = []
        thresholds = np.linspace(0, 1, 100)

        for threshold in thresholds:
            y_pred_threshold = (self.best_model.predict_proba(self.X_test)[:, 1] >= threshold).astype(int)
            f1_scores.append(f1_score(self.y_test, y_pred_threshold))

        self.max_f1_score = max(f1_scores)
        self.max_f1_threshold = thresholds[f1_scores.index(self.max_f1_score)]

        return {"max_f1_score": self.max_f1_score, "max_f1_threshold": self.max_f1_threshold}
    
    def save_best_model(self):
        with open('best_model.pkl', 'wb') as file:
            pickle.dump({"best_model": self.best_model, "threshold": self.max_f1_threshold}, file)
        
    
    
