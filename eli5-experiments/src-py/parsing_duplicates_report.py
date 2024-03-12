import pandas as pd
from docx.api import Document

document = Document('../../data/eli5_ds/duplicates_report/identical_dialogues.docx')

duplicates = []
for table in document.tables:
    data = []
    for i, clm in enumerate(table.columns):
        clms = [cell.text for cell in clm.cells[1:] if cell.text != '']
        if len(clms) > 1:
            duplicates.append([clms[0], clms[1:]])

    print(duplicates[-1])


df = pd.DataFrame(duplicates, columns=('dlg_id', 'duplicate_ids'))
df.to_pickle('../../data/eli5_ds/duplicates_report/identical_dialogues.pkl')