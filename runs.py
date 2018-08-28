import pickle

data1 = pickle.load(open("./datasets/horse2zebra/bboxA.p", "rb"))
print(data1)
# mergedata = {**data, **data1}
# pickle.dump(mergedata, open("./datasets/horse2zebra/bboxB.p", "wb"))