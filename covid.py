import pandas as pd

from pandasgui import show
import icd10

filename = 'covid_deaths.csv'


def load_dataset():
    code = icd10.find("J02")
    print(code.description)
    # df = pd.read_csv(filename, index_col=0, header=0)
    # show(df, settings={'block': True})

if __name__ == '__main__':
    load_dataset()

# datatable = QtGui.QTableWidget(parent=self)
# datatable.setColumnCount(len(df.columns))
# datatable.setRowCount(len(df.index))
# for i in range(len(df.index)):
#     for j in range(len(df.columns)):
#         self.datatable.setItem(i,j,QtGui.QTableWidgetItem(str(df.iget_value(i, j))))
