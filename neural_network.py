import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import c3aidatalake


def get_linelist_data():
    dataLineList = c3aidatalake.fetch(
        "linelistrecord",
        {
            "spec": {
                "filter": "",
                "limit": 1000000
            }
        }
    )

    print(type(dataLineList))

    print("Data", dataLineList['symptoms'].to_string())

    symptoms = ['phlegm', 'chills', 'fatigue', 'headache', 'sneezing', 'mild', 'malaise', 'sore throat', 'cough',
                'fever']

    symptom_df = dataLineList.copy()
    for s in symptoms:
        symptom_df[s] = 0
        symptom_df.loc[symptom_df["symptoms"].str.contains(s, na=False), s] = 1

    symptom_df["result"] = 0
    symptom_df.loc[symptom_df["didDie"] == True, "result"] = 1  # Death
    symptom_df.loc[symptom_df["didDie"] == False, "result"] = 0  # Life

    symptom_df.loc[symptom_df["symptoms"].str.contains("pneumonia", na=False), "result"] = 1
    symptom_df.loc[symptom_df["symptoms"].str.contains("pneumonitis", na=False), "result"] = 1
    symptom_df.loc[symptom_df["symptoms"].str.contains("pain", na=False), "result"] = 1
    symptom_df.loc[symptom_df["symptoms"].str.contains("dyspnea", na=False), "result"] = 1
    symptom_df.loc[symptom_df["didRecover"] == False, "result"] = 1  # Death

    inputs = ['age', 'phlegm', 'chills', 'fatigue', 'headache', 'sneezing', 'mild', 'malaise', 'sore throat',
              'cough', 'fever']

    final_df = symptom_df[inputs].copy()
    final_df.loc[final_df["age"].isnull(), "age"] = final_df["age"].mean()

    result_df = symptom_df["result"].copy()

    return final_df, result_df


def get_clinical_data():
    dataClinicalTrial = c3aidatalake.fetch(
        "clinicaltrial",
        {
            "spec": {
                "filter": "",
                "limit": 100000
            }
        }
    )

    outcomes = ['dyspnea', 'cough', 'pneumonia or ards', 'organ failure or dysfunction (sofa)',
                'treatment-emergent adverse events', 'radiographic findings',
                'c-reactive protein', 'fever', 'serious adverse events',
                'viral load or clearance', 'adverse events',
                'non-invasive ventilation', 'icu admission', 'hospitalization',
                'invasive mechanical ventilation or ecmo', 'mortality']

    temp_df = dataClinicalTrial.copy()

    temp_df["result"] = 1
    temp_df["outcome"] = temp_df["outcome"].str.lower()
    for o in outcomes:
        temp_df.loc[temp_df["outcome"].str.contains(o, na=False), "result"] = 0

    treatment_df = dataClinicalTrial.copy()

    treatments = ['corticosteroids', 'acalabrutinib', 'fpv', 'sarilumab', 'non-invasive respiratory support',
                  'anticoagulants', 'stem cells', 'tcm', 'lpv/r', 'jaki', 'vaccine', 'alternative therapy', 'tcz',
                  'plasma based therapy', 'mab', 'remdesivir', 'hcq', 'soc']

    treatment_df["treatmentType"] = treatment_df["treatmentType"].str.lower()
    for t in treatments:
        treatment_df[t] = 0
        treatment_df.loc[treatment_df["treatmentType"].str.contains(t, na=False), t] = 1

    final_df = treatment_df[treatments].copy()
    result_df = temp_df["result"].copy()

    return final_df, result_df


class Net(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

        # return F.log_softmax(x, dim=1)


# X = torch.rand((5,))
# X = X.view(-1, 5)

x_df, y_df = get_linelist_data()
X, Y = x_df.to_numpy(), y_df.to_numpy()

# data = load_breast_cancer()
# X, Y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print("Data:", X.shape, "Train:", X_train.shape, "Test:", X_test.shape, "Target:", Y.shape, "Target Train:",
      y_train.shape, "Target Test", y_test.shape)

print(y_df.to_string())

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

net = Net(X_train.shape[1])
print(net)

EPOCHS = 700

train_losses = np.zeros(EPOCHS)
test_losses = np.zeros(EPOCHS)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = net(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    outputs_test = net(X_test)
    loss_test = criterion(outputs_test, y_test)

    train_losses[epoch] = loss.item()
    test_losses[epoch] = loss_test.item()

    # if epoch % 50 == 0:
    #     print(f'In this epoch {epoch+1}/{EPOCHS}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')

    if epoch % 50 == 0:
        print(loss)

plt.plot(train_losses, label='train_loss')
plt.plot(test_losses, label='test_loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Total Loss")
plt.title("Train and Test Errors (Severity Risk Prediction)")
plt.legend()
plt.show()
