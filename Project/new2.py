import pickle
from new import vehicle
from new import person
with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript10.pkl', 'rb') as f:
    loaded_veh = pickle.load(f)
    loaded_per = pickle.load(f)
print(loaded_per.get())
print(loaded_veh.get())