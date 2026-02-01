import pandas as pd
from src.load_kaggle import load_kaggle_data
from src.load_extra import load_extra_data
from src.load_data import load_heart_data
from src.preprocess import preprocess_heart
from src.models.baseline import baseline_rule
from src.models.svm_model import train_svm
from src.models.rf_model import train_rf
from src.models.logistic_regression import train_logistic_regression
from src.evaluate import evaluate_model, plot_roc_curves
from sklearn.model_selection import train_test_split

def run():
    df_cleveland = load_heart_data("data/processed.cleveland.data")
    df_kaggle = load_kaggle_data("data/heart.csv")
    df_extra = load_extra_data("data/dataset.csv")

    print("Cleveland rows:", len(df_cleveland))
    print("Kaggle rows:", len(df_kaggle))
    print("Extra rows:", len(df_extra))

    df = pd.concat([df_cleveland, df_kaggle, df_extra], ignore_index=True)

    print("Total combined rows:", len(df))

    X,y=preprocess_heart(df)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

    models_results={}

    base_pred=baseline_rule(df.loc[y_test.index])
    print("\nBaseline:",evaluate_model(y_test,base_pred))

    svm=train_svm(X_train,y_train)
    svm_pred=svm.predict(X_test)
    svm_proba = svm.predict_proba(X_test)[:, 1]
    svm_results=evaluate_model(y_test,svm_pred,svm_proba)
    print("\nSVM:",svm_results)
    print("SVM Probability (%):", [f"{p*100:.2f}%" if isinstance(p, float) else f"{p:.2f}%" for p in (svm_proba - svm_proba.min()) / (svm_proba.max() - svm_proba.min())][:5])
    models_results['SVM']={'y_true':y_test,'y_proba':svm_proba,'roc_auc':svm_results.get('roc_auc',0)}

    rf=train_rf(X_train,y_train)

    import joblib
    joblib.dump(rf, "best_model.pkl")
    print("Model saved as best_model.pkl")

    rf_pred=rf.predict(X_test)
    rf_proba=rf.predict_proba(X_test)[:,1]
    rf_results=evaluate_model(y_test,rf_pred,rf_proba)
    print("\nRandom Forest:",rf_results)
    print("Random Forest Probability (%):", [f"{p*100:.2f}%" for p in rf_proba[:5]])
    models_results['Random Forest']={'y_true':y_test,'y_proba':rf_proba,'roc_auc':rf_results.get('roc_auc',0)}

    lr=train_logistic_regression(X_train,y_train)
    lr_pred=lr.predict(X_test)
    lr_proba=lr.predict_proba(X_test)[:,1]
    lr_results=evaluate_model(y_test,lr_pred,lr_proba)
    print("\nLogistic Regression:",lr_results)
    print("Logistic Regression Probability (%):", [f"{p*100:.2f}%" for p in lr_proba[:5]])
    models_results['Logistic Regression']={'y_true':y_test,'y_proba':lr_proba,'roc_auc':lr_results.get('roc_auc',0)}

    plot_roc_curves(models_results)

if __name__=='__main__':
    run()
