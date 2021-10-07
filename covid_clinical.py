import c3aidatalake
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# dataLineList = c3aidatalake.fetch(
#   "linelistrecord",
#   {
#       "spec" : {
#           "filter" : "",
#           "limit" : 1000000
#       }
#   }
# )


# dataTherapeutic = c3aidatalake.fetch(
#     "therapeuticasset",
#     {
#         "spec" : {
#             "filter" : "!contains(clinicalTrialsCovid19, 'NaN')",
#             "limit" : 100000
#         }
#     }
# )


dataClinicalTrial = c3aidatalake.fetch(
    "clinicaltrial",
    {
        "spec" : {
            "filter" : "",
            "limit" : 100000
        }
    }
)

# print(clinicalTrial.get('outcome'), clinicalTrial.get('treatmentType'))




outcome_df = dataClinicalTrial.copy()
outcome_df["outcome"] = outcome_df["outcome"].str.lower()
outcome_df["outcome"] = outcome_df["outcome"].str.split(", ")
outcome_df = outcome_df.explode("outcome")
outcome_df = outcome_df.dropna(subset = ["outcome"])
outcome_freq = outcome_df.groupby(["outcome"]).agg("count")[["id"]].sort_values("id")

outcome_freq = outcome_freq[-30:]

outcomes = ['other mild symptoms',
       'dyspnea', 'cough', 'pneumonia or ards', 'organ failure or dysfunction (sofa)',
       'respiratory rate', 'pao2-fio2', 'il-6',
       'treatment-emergent adverse events', 'radiographic findings', 'spo2',
       'c-reactive protein', 'fever', 'serious adverse events',
       'viral load or clearance', 'adverse events',
       'non-invasive ventilation', 'icu admission', 'hospitalization',
       'invasive mechanical ventilation or ecmo', 'mortality']

outcome_df["result"] = 1
for o in outcomes:
    outcome_df.loc[outcome_df["outcome"].str.contains(o, na=False), "result"] = 0
    print(outcome_df[outcome_df["outcome"].str.contains(o, na=False) == True].shape)
    print("__________________________________________________________________________ ", o)

# print(dataClinicalTrial["treatmentType"].to_string())

treatment_df = dataClinicalTrial.copy()
# treatment_df["treatmentType"] = treatment_df["treatmentType"].str.lower()
# treatment_df["treatmentType"] = treatment_df["treatmentType"].str.split(", ")
# treatment_df = treatment_df.explode("treatmentType")
# treatment_df = treatment_df.dropna(subset = ["treatmentType"])
# treatment_freq = treatment_df.groupby(["treatmentType"]).agg("count")[["id"]].sort_values("id")

# treatment_freq = treatment_freq[-20:]
# print(treatment_freq.index.to_list())

treatments = ['corticosteroids', 'acalabrutinib', 'fpv', 'sarilumab', 'non-invasive respiratory support', 'anticoagulants', 'stem cells', 'tcm', 'lpv/r', 'jaki', 'vaccine', 'alternative therapy', 'tcz', 'plasma based therapy', 'mab', 'remdesivir', 'hcq', 'soc']

treatment_df["treatmentType"] = treatment_df["treatmentType"].str.lower()
for t in treatments:
    treatment_df[t] = 0
    treatment_df.loc[treatment_df["treatmentType"].str.contains(t, na=False), t] = 1

final_df = treatment_df[treatments].copy()
result_df = outcome_df["result"]
print(final_df.to_string())

#
# data = dataLineList
#
# symptom_freq = symptom_freq[-20:]

# symptoms = ['phlegm', 'chills', 'fatigue', 'headache', 'sneezing', 'mild', 'malaise', 'sore throat', 'cough', 'fever']
#
# symptom_df = dataLineList.copy()
# for s in symptoms:
#     symptom_df[s] = 0
#     symptom_df.loc[symptom_df["symptoms"].str.contains(s, na=False), s] = 1
#     print("__________________________________________________________________________ ", s)
#
# symptom_df["result"] = 0
# symptom_df.loc[symptom_df["didDie"] == True, "result"] = 1 #Death
# symptom_df.loc[symptom_df["didDie"] == False, "result"] = 0 #Life
#
# symptom_df.loc[symptom_df["symptoms"].str.contains("pneumonia", na=False), "result"] = 1
# symptom_df.loc[symptom_df["symptoms"].str.contains("pneumonitis", na=False), "result"] = 1
# symptom_df.loc[symptom_df["symptoms"].str.contains("pain", na=False), "result"] = 1
# symptom_df.loc[symptom_df["symptoms"].str.contains("dyspnea", na=False), "result"] = 1
# symptom_df.loc[symptom_df["didRecover"] == False, "result"] = 1 #Death
#
# print(symptom_df[symptom_df["result"] == 1].shape)
#
# # print(symptom_df.columns.to_list())
#
# inputs = ['gender', 'age', 'phlegm', 'chills', 'fatigue', 'headache', 'sneezing', 'mild', 'malaise', 'sore throat', 'cough', 'fever', 'result']
#
# final_df = symptom_df[inputs].copy()
#
#
# print(final_df.columns)


# print(symptom_freq.index.to_list())


# data['treatmentType'].value_counts().plot(kind='bar')

# data['date'] = pd.to_datetime(data['caseConfirmationDate'])
# data['date_delta'] = (data['date'] - data['date'].min())  / np.timedelta64(1,'D') / 0.5
# print(data['date'])

# Plot the data
# plt.figure(figsize = (10, 6))
# j = 0
# for i in ['date_delta']:
#     plt.subplot(1, 1, j+1)
#     # print(data[i][data['didRecover']==True])
#     sns.distplot(data[i][data['didRecover']==True], color='g', label = 'Recovered')
#     sns.distplot(data[i][data['didDie']==True], color='b', label = 'Critical')
#     plt.legend(loc='best')
#     j+=1

# print(symptom_freq)
# symptom_freq = symptom_freq[-40:]
#
# plt.bar(symptom_freq.index, symptom_freq["id"])
# plt.xticks(rotation = 90)
# plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.5)
#
# plt.xlabel("Treatment Type in Clinical Trial")
# plt.ylabel("Number of Patients")
# plt.title("Clinical Trials: Treatment Type")
# plt.show()


# label_encoder = LabelEncoder()

# print(data.index, data.get('productType'))
# print(data.get('chronicDisease')[data.get('chronicDisease') != ''])

# For Line List Data
# for col in ['didDie', 'didRecover']:
#     data.iloc[:,data.columns.tolist().index(col)] = label_encoder.fit_transform(data.get(col)).astype('float64')

# For Therapeutic Asset Data
# for col in ['stageOfDevelopment', 'developer', 'description', 'lastUpdatedNotes', 'productType', ]:
#     data.iloc[:,data.columns.tolist().index(col)] = label_encoder.fit_transform(data.get(col)).astype('float64')

# For Clinical Trial Datasets
# for col in ['country', 'design', 'trialStatus', 'blinding', 'location.id']:
#     data.iloc[:,data.columns.tolist().index(col)] = label_encoder.fit_transform(data.get(col)).astype('float64')


# corr = data.corr()
# sns.heatmap(corr)
# plt.xticks(rotation=50)
# plt.subplots_adjust(left=0.4, right=0.9, top=0.9, bottom=0.4)
# plt.show()

# print(type(lineList))




