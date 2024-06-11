import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, classification_report
import matplotlib.pyplot as plt 
import joblib
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
    
    def train(self, X_train, y_train, X_test, y_test, n_trials_=100):
        self.X_train = X_train  
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(self.rf_objective, n_trials=n_trials_)
        rf_best_params = study_rf.best_params

        study_mlp = optuna.create_study(direction='maximize')
        study_mlp.optimize(self.mlp_objective, n_trials=n_trials_)
        nn_best_params = study_mlp.best_params
        
        rf=RandomForestClassifier(n_estimators=rf_best_params['n_estimators'],
                                  max_depth=rf_best_params['max_depth'],
                                  min_samples_split=rf_best_params['min_samples_split'],
                                  min_samples_leaf=rf_best_params['min_samples_leaf'])
        nn=MLPClassifier(hidden_layer_sizes=[nn_best_params['layer1_size'],nn_best_params['layer2_size']][:nn_best_params['num_layers']],
                         activation=nn_best_params['activation'],
                         solver=nn_best_params['solver'],
                         alpha=nn_best_params['alpha'],
                         max_iter=10000, early_stopping=True)
        
        rf.fit(self.X_train,self.y_train)
        nn.fit(self.X_train,self.y_train)
        self.rf_model = rf
        self.nn_model = nn
        
            # Save the models
        joblib.dump(self.rf_model, 'rf_model.joblib')
        joblib.dump(self.nn_model, 'nn_model.joblib')
        

    
    def classification_reports(self):
        print("Random Forest")
        print(classification_report(self.y_test,self.rf_model.predict(self.X_test)))
        print("Neural Network")
        print(classification_report(self.y_test,self.nn_model.predict(self.X_test)))
        

    def roc_curve(self):

        ax = plt.gca()
        rf_disp = RocCurveDisplay.from_estimator(self.rf_model, self.X_test, self.y_test, ax=ax)
        mlp_disp = RocCurveDisplay.from_estimator(self.nn_model, self.X_test, self.y_test, ax=ax)
        plt.show()
        
    def precision_recall_curve(self):
      
        ax = plt.gca()
        rf_disp = PrecisionRecallDisplay.from_estimator(self.rf_model, self.X_test, self.y_test, ax=ax)
        mlp_disp = PrecisionRecallDisplay.from_estimator(self.nn_model, self.X_test, self.y_test, ax=ax)
        plt.show()
                
            
    
    
    
