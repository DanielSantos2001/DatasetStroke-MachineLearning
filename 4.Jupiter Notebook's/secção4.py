import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
import itertools
from sklearn import svm

df = pd.read_csv('stroke.csv')

while True:
    print("\nMenu:")
    print("0. Sair")

    print("Algoritmos:")
    print("1. KNN")
    print("2. Decision Trees")
    print("3. Logistic Regression")
    print("4. SVM")

    opcao = input("\nOpção: ")
    if opcao == "0":

        exit(0)
        
    elif opcao == "1":

        print("Head:")
        print(df.head())
        input("")

        print("Quantia de strokes:")
        print(df['stroke'].value_counts())
        input("")

        print("Quantia de strokes em gráfico:")
        df.hist(column='stroke', bins=50)
        plt.show()
        input("")

        print("Features:")
        print(df.columns)
        print(df.dtypes)
        input("")

        print("Normalizar a data:")
        varint = ['avg_glucose_level', 'bmi']
        df[varint] = df[varint].astype("float")
        varfloat = ['id', 'age', 'hypertension', 'heart_disease']
        df[varfloat] = df[varfloat].astype("int")
        avg_norm_loss = df['bmi'].astype("float").mean(axis=0)
        df['bmi'].replace(np.nan, avg_norm_loss, inplace=True)
        input("")

        print("Converter a data frame do Pandas para vetor Numpy:")
        X = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].values
        print(X[0:5])
        y = df['stroke'].values
        print(y[0:5])
        input("")

        print("Train Test Split:")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
        print('Train set:', X_train.shape, y_train.shape)
        print('Test set:', X_test.shape, y_test.shape)
        input("")

        print("Training (K=4):")
        k = 4
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)
        input("")

        print("Usar o modelo para prever o teste:")
        yhat = neigh.predict(X_test)
        print(yhat[0:5])
        input("")

        print("Accuracy evaluation:")
        print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
        print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
        input("")

    elif opcao == "2":

        print("Head:")
        print(df.head())
        input("")

        print("Preencher missing values e remover colunas não numéricas:")
        avg_norm_loss = df['bmi'].astype("float").mean(axis=0)
        df['bmi'].replace(np.nan, avg_norm_loss, inplace=True)
        X = df[['gender', 'age', 'hypertension', 'ever_married', 'work_type', 'Residence_type', 'heart_disease',
                'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']].values
        print(X[0:5])
        input("")

        print("Transformar variáveis categóricas em dummies:")
        sex = preprocessing.LabelEncoder()
        sex.fit(['Male', 'Female', 'Other'])
        X[:, 0] = sex.transform(X[:, 0])
        married = preprocessing.LabelEncoder()
        married.fit(['Yes', 'No'])
        X[:, 3] = married.transform(X[:, 3])
        work = preprocessing.LabelEncoder()
        work.fit(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
        X[:, 4] = work.transform(X[:, 4])
        residence = preprocessing.LabelEncoder()
        residence.fit(['Urban', 'Rural'])
        X[:, 5] = residence.transform(X[:, 5])
        smoke = preprocessing.LabelEncoder()
        smoke.fit(['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
        X[:, 9] = smoke.transform(X[:, 9])
        print(X[0:5])
        input("")

        print("Definir a label:")
        y = df['stroke']
        print(y[0:5])
        input("")

        print("Train Test Split:")
        X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
        print('Train set:', X_trainset, y_trainset)
        print('Test set:', X_testset, y_testset)
        input("")

        print("Criar a instância DecisionTreeClassifier chamada strokeTree:")
        strokeTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
        print(strokeTree)  # it shows the default parameters
        input("")

        print("Training strokeTree:")
        strokeTree.fit(X_trainset, y_trainset)
        input("")

        print("Prediction com o testing set:")
        predictTree = strokeTree.predict(X_testset)
        print(predictTree[0:5])
        print(y_testset[0:5])
        input("")

        print("Accuracy:")
        print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predictTree))
        input("")

    elif opcao == "3":

        print("Head:")
        print(df.head())
        input("")

        print("Preencher missing values:")
        avg_norm_loss = df['bmi'].astype("float").mean(axis=0)
        df['bmi'].replace(np.nan, avg_norm_loss, inplace=True)
        print(df.head())
        input("")

        print("Converter a variável target em int e modelar o set:")
        df['stroke'] = df['stroke'].astype('int')
        X = df[['gender', 'age', 'hypertension', 'ever_married', 'work_type', 'Residence_type', 'heart_disease',
                'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']].values
        print(X[0:5])
        input("")

        print("Definir o X e y para o dataset:")
        X = np.asarray(df[['gender', 'age', 'hypertension', 'ever_married', 'work_type', 'Residence_type',
                           'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']])
        print(X[0:5])
        y = np.asarray(df['stroke'])
        print(y[0:5])
        input("")

        print("Normalizar o dataset:")
        # cast
        varfloat = ['age', 'hypertension', 'heart_disease']
        df[varfloat] = df[varfloat].astype("int")
        # dummies
        sex = preprocessing.LabelEncoder()
        sex.fit(['Male', 'Female', 'Other'])
        X[:, 0] = sex.transform(X[:, 0])
        married = preprocessing.LabelEncoder()
        married.fit(['Yes', 'No'])
        X[:, 3] = married.transform(X[:, 3])
        work = preprocessing.LabelEncoder()
        work.fit(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
        X[:, 4] = work.transform(X[:, 4])
        residence = preprocessing.LabelEncoder()
        residence.fit(['Urban', 'Rural'])
        X[:, 5] = residence.transform(X[:, 5])
        smoke = preprocessing.LabelEncoder()
        smoke.fit(['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
        X[:, 9] = smoke.transform(X[:, 9])
        # transform
        X = preprocessing.StandardScaler().fit(X).transform(X)
        print(X[0:5])
        input("")

        print("Train Test Split:")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
        print('Train set:', X_train.shape, y_train.shape)
        print('Test set:', X_test.shape, y_test.shape)
        input("")

        print("Training:")
        LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
        print(LR)
        input("")

        print("Predict:")
        yhat = LR.predict(X_test)
        print(yhat)
        input("")

        print("Predict probability:")
        yhat_prob = LR.predict_proba(X_test)
        print(yhat_prob)
        input("")

        print("Jaccard:")
        print(jaccard_score(y_test, yhat))
        input("")

        print("Confusion matrix:")
        def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        print(confusion_matrix(y_test, yhat, labels=[1, 0]))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
        plt.show()
        #classificação (f1)
        print(classification_report(y_test, yhat))
        input("")

        print("LogLoss:")
        print(log_loss(y_test, yhat_prob))
        input("")

    elif opcao == "4":
        print("Preencher missing values e head:")
        avg_norm_loss = df['bmi'].astype("float").mean(axis=0)
        df['bmi'].replace(np.nan, avg_norm_loss, inplace=True)
        print(df.head())                      
        input("")
        
        print("Distribuição de stroke em função do average glucose level e bmi:")
        ax = df[df['stroke'] == 1][0:1000].plot(kind='scatter', x='avg_glucose_level', y='bmi', color='DarkBlue', label='stroke');
        df[df['stroke'] == 0][0:1000].plot(kind='scatter', x='avg_glucose_level', y='bmi', color='Yellow', label='no stroke', ax=ax);
        plt.show()
        input("")

        print("Transformar o pandas dataframe em numpy array:")
        #transformar
        X = np.asarray(df[['gender', 'age', 'hypertension', 'ever_married', 'work_type', 'Residence_type',
                           'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']])
        #dummies
        sex = preprocessing.LabelEncoder()
        sex.fit(['Male', 'Female', 'Other'])
        X[:, 0] = sex.transform(X[:, 0])
        married = preprocessing.LabelEncoder()
        married.fit(['Yes', 'No'])
        X[:, 3] = married.transform(X[:, 3])
        work = preprocessing.LabelEncoder()
        work.fit(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
        X[:, 4] = work.transform(X[:, 4])
        residence = preprocessing.LabelEncoder()
        residence.fit(['Urban', 'Rural'])
        X[:, 5] = residence.transform(X[:, 5])
        smoke = preprocessing.LabelEncoder()
        smoke.fit(['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
        X[:, 9] = smoke.transform(X[:, 9])
        print(X[0:5])
        # label
        y = np.asarray(df['stroke'])
        print(y[0:5])
        input("")

        print("Train Test Split:")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
        print('Train set:', X_train.shape, y_train.shape)
        print('Test set:', X_test.shape, y_test.shape)
        input("")

        print("Modeling:")
        clf = svm.SVC(kernel='rbf')
        clf.fit(X_train, y_train)
        input("")

        print("Predict:")
        yhat = clf.predict(X_test)
        print(yhat[0:5])
        input("")

        print("Jaccard:")
        print(jaccard_score(y_test, yhat))
        input("")

        print("Confusion Matrix:")
        def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        print(confusion_matrix(y_test, yhat, labels=[1, 0]))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, yhat, labels=[0, 1])
        np.set_printoptions(precision=1)
        print(cnf_matrix)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['Stroke(1)', 'NoStroke(0)'], normalize=False,title='Confusion matrix')
        plt.show()
        input("")

        print("F1:")
        print(f1_score(y_test, yhat, average='weighted'))
        input("")
        
    else:
        print("ERROR")
