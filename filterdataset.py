import glob
import os
import pickle

a = glob.glob("testB/*")
a = a + (glob.glob("trainB/*"))

mask = pickle.load(open("maskB.p", "rb"))
for f in a:
    if f.split("/")[-1] not in mask.keys():
        print(f)
        os.remove(f)
