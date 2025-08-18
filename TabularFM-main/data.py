import relbench
from relbench.datasets import get_dataset_names
from relbench.datasets import get_dataset
print (relbench.__version__)

# ['rel-amazon',
#  'rel-avito',
#  'rel-event',
#  'rel-f1',
#  'rel-hm',
#  'rel-stack',
#  'rel-trial']


dataset = get_dataset(name="rel-amazon", download=True)
dataset = get_dataset(name="rel-avito", download=True)
dataset = get_dataset(name="rel-event", download=True)
dataset = get_dataset(name="rel-f1", download=True)
dataset = get_dataset(name="rel-hm", download=True)
dataset = get_dataset(name="rel-stack", download=True)
dataset = get_dataset(name="rel-trial", download=True)
db = dataset.get_db()
print (db.table_dict.keys())

# table1 = db.table_dict["AdsInfo"]
# print (table1.df.iloc[0])
# print ('*'*100)
# table2 = db.table_dict["SearchStream"]
# print (table2.df.iloc[0])
# print ('*'*100)
# table3 = db.table_dict["PhoneRequestsStream"]
# print (table3.df.iloc[0])
# print ('*'*100)
# table4 = db.table_dict["UserInfo"]
# print (table4.df.iloc[0])
# print ('*'*100)
# table5 = db.table_dict["Location"]
# print (table5.df.iloc[0])
# print ('*'*100)
