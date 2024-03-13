import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

#1
df_train = pd.read_csv("train.csv")
headers = df_train.columns

while True:
        print("\nMenu:")
        print("0. Sair")

        print("Aquisição de dados:")
        print("1. Visualizar dataset\n2. Visualizar as 15 primeiras linhas\n3. Obter a dimensão do dataframe\n4. Remover a primeira linha do dataframe que contém os headers")
        print("5. Adicionar os mesmos headers que retirou anteriormente ao dataframe\n6. Mostrar o nome das colunas\n7. Renomear a coluna Food and drink para Food\n8. Renomear a primeira coluna como Indice ")

        print("Estatística Descritiva:")
        print("9. Obter a estatística descritiva para cada coluna\n10. Obter a estatística descritiva somente para variáveis categóricas\n11. Obter a estatística descritiva somente para a coluna Flight Distance\n12. Obter os valores máximos e mínimos da coluna Departure Delay in Minutes")
        print("13. Apresentar a frequência de cada um dos valores possíveis da coluna Class\n14. Criar uma dataframe contendo cada um dos valores possíveis da coluna Class\n15. Apresentar graficamente o histograma da coluna Departure Delay in Minutes\n16. Apresentar as categorias existentes na coluna satisfaction")
        print("17. Apresentar em caixa de bigodes(boxplot) a relação entre as colunas satisfaction e Departure Delayin Minutes")

        print("Missing Values:")
        print("18. Substituir na coluna Departure Delay in Minutes os valores 70 por NaN, e na coluna Arrival Time in Minutes os valores 100 por NaN\n19. Obter o número de missing values em cada coluna\n20. Descrever as várias formas de lidar com missing values\n21. Substituir os missing values presentes na coluna Departure Delay in Minutes, pela média da coluna")
        print("22. Eliminar os missing values da coluna Arrival Delay in Minutes")

        print("Normalização:")
        print("23. Caracterizar os tipos de dados utilizados nos dataframes do Pandas\n24. Obter o tipo de dados de cada coluna\n25. Converter os tipos das colunas para o formato adequado\n26. Normalizar as colunas Departure Delay in Minutes e Arrival Delay in Minutes, através da criação de duas novas colunas Departure Delay in Seconds e Arrival Delay in Seconds com os respetivos tempos em segundos.")
        print("27. Normalizar as colunas Departure Delay in Seconds e Arrival Delay in Seconds de forma a que os seus valores variem entre 0 e 1")

        print("Binning:")
        print("28. Criar uma nova coluna (feature) designada por Departure Delay in Seconds_Binned que divida os tempos de atraso em três categorias Baixa, Media, Elevada\n29. Comparar a coluna Departure Delay in Seconds_Binned com aquela que lhe deu origem mostrando simultaneamente as primeiras 5 linhas de cada uma\n30. Apresentar graficamente o histograma da coluna Departure Delay in Seconds_Binned")

        print("Estatística Inferencial:")
        print("31. Obter o coeficiente de correlação entre as diferentes colunas (features) númericas do dataset\n32. Para as duas features mais fortemente correlacionadas produza o respetivo scatterplot\n33. Apresentar uma pivot table considerando a média de idades em função das colunas Type of Travel e Class")
        print("34. Apresentar graficamente a pivot table da alínea anterior no formato de um heatmap\n35. Apresentar a média de idades para cada grupo da coluna Satisfation\n36. Realizar o teste one-way ANOVA para os grupos da coluna Satisfation considerando a Age como variável dependente")

        opcao = input("\nOpção: ")
        if opcao == "0":
            exit(0)
        elif opcao == "1":
            print(df_train)

        elif opcao == "2":
            print(df_train.head(15))

        elif opcao == "3":
            print(df_train.shape)

        elif opcao == "4":
            df_train = df_train.rename(columns=df_train.iloc[0]).drop(df_train.index[0])

        elif opcao == "5":
            df_train = pd.read_csv("train.csv", names=headers, low_memory=False)
            df_train = df_train.iloc[1:]
            df_train.reset_index(drop=True, inplace=True)

        elif opcao == "6":
            print(df_train.columns)

        elif opcao == "7":
            df_train.rename(columns={'Food and drink': 'Food'}, inplace=True)

        elif opcao == "8":
            df_train.index.name = 'Indice'

        elif opcao == "9":
            print(df_train.describe(include = "all"))

        elif opcao == "10":
            print(df_train.describe(include=['object']))

        elif opcao == "11":
            print(df_train[['Flight Distance']].describe())

        elif opcao == "12":
            print(df_train['Departure Delay in Minutes'].min())
            print(df_train['Departure Delay in Minutes'].max())

        elif opcao == "13":
            print(df_train['Class'].value_counts())

        elif opcao == "14":
            print(df_train['Class'].value_counts().to_frame())

        elif opcao == "15":
            plt.hist(df_train["Departure Delay in Minutes"])
            plt.xlabel("Departure Delay in Minutes")
            plt.ylabel("count")
            plt.title("Histogram")
            plt.show()

        elif opcao == "16":
            print(df_train['satisfaction'].unique())

        elif opcao == "17":
            sns.boxplot(x="satisfaction", y="Departure Delay in Minutes", data=df_train)
            plt.show()

        elif opcao == "18":
            df_train['Departure Delay in Minutes'] = df_train['Departure Delay in Minutes'].replace(70, np.NaN)
            df_train['Arrival Delay in Minutes'] = df_train['Arrival Delay in Minutes'].replace(100, np.NaN)
            print(df_train['Departure Delay in Minutes'].head())
            print(df_train['Arrival Delay in Minutes'].head())

        elif opcao == "19":
            missing_data = df_train.isnull().sum()
            print(missing_data)

        elif opcao == "20":
            print("Missing values")

        elif opcao == "21":
            avg_norm_loss = df_train["Departure Delay in Minutes"].astype("float").mean(axis=0)
            df_train["Departure Delay in Minutes"].replace(np.nan, avg_norm_loss, inplace=True)
            print(df_train['Departure Delay in Minutes'].head())

        elif opcao == "22":
            df_train.dropna(subset=["Arrival Delay in Minutes"],axis=0, inplace=True)
            df_train.reset_index(drop=True, inplace=True)
            print(df_train['Arrival Delay in Minutes'].head())

        elif opcao == "23":
            print("Caracterizar os tipos de dados utilizados nos dataframes do Pandas")

        elif opcao == "24":
            print(df_train.dtypes, df_train.info())

        elif opcao == "25":
            varint = ['Flight Distance','Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes']
            df_train[varint] = df_train[varint].astype("float")

        elif opcao == "26":
            df_train['Departure Delay in Seconds'] = df_train['Departure Delay in Minutes'] * 60
            df_train['Arrival Delay in Seconds'] = df_train['Arrival Delay in Minutes'] * 60
            print(df_train['Arrival Delay in Minutes'].head())
            print(df_train['Arrival Delay in Seconds'].head())

        elif opcao == "27":
            df_train['Departure Delay in Seconds'] = df_train['Departure Delay in Seconds']/df_train['Departure Delay in Seconds'].max()
            df_train['Arrival Delay in Seconds'] = df_train['Arrival Delay in Seconds'] / df_train['Arrival Delay in Seconds'].max()
            print(df_train['Arrival Delay in Seconds'].head())

        elif opcao == "28":
            bins = np.linspace(min(df_train['Departure Delay in Seconds']), max(df_train['Departure Delay in Seconds']), 4)
            group_names = ['Low', 'Medium', 'High']
            df_train['Departure Delay in Seconds_Binned'] = pd.cut(df_train['Departure Delay in Seconds'], bins, labels=group_names, include_lowest=True)

        elif opcao == "29":
            print(df_train[['Departure Delay in Seconds','Departure Delay in Seconds_Binned']].head(5))

        elif opcao == "30":
            plt.hist(df_train["Departure Delay in Seconds_Binned"])
            plt.xlabel("Departure Delay in Seconds_Binned")
            plt.ylabel("count")
            plt.title("Histogram")
            plt.show()

        elif opcao == "31":
            print(df_train.corr())

        elif opcao == "32":
            sns.regplot(x="Departure Delay in Minutes", y="Arrival Delay in Minutes", data=df_train)
            plt.ylim(0, )
            plt.show()

        elif opcao == "33":
            df_new = df_train[['Age', 'Type of Travel', 'Class']]
            grouped = df_new.groupby(['Type of Travel', 'Class'], as_index=False).mean()
            grouped_pivot = grouped.pivot(index='Type of Travel', columns='Class')
            print(grouped_pivot)

        elif opcao == "34":
            plt.pcolor(grouped_pivot, cmap='RdBu')
            plt.colorbar()
            plt.show()

        elif opcao == "35":
            df_groups = df_train[['Age', 'satisfaction']]
            grouped_by_mean = df_groups.groupby(['satisfaction'], as_index=False).mean()
            print(grouped_by_mean)

        elif opcao == "36":
            grouped = df_groups.groupby(['satisfaction'])
            df_groups['Age'].unique()
            f_val, p_val = f_oneway(grouped.get_group('neutral or dissatisfied')['Age'], grouped.get_group('satisfied')['Age'])
            print(f_val,p_val)

        else:
            print("ERROR")




