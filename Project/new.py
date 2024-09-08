import pickle
class person:
    def set(self,name):
        self.name=name
    def get(self):
        return self.name
    
class vehicle:
    def set(self,model):
        self.model=model
    def get(self):
        return self.model
veh=vehicle()
per=person()
print("bbbbbbbbbbb")
veh.set("bmw")
per.set("aaaaa")
with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript10.pkl', 'wb') as f:
    pickle.dump(veh,f)
    pickle.dump(per,f)


