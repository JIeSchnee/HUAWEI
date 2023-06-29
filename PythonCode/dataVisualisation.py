import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('test.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)

Klist = list(data.keys())

key_list = list(data[1][1].keys())
d_dict = {}

for k in Klist:
    for key in key_list:
        d = data[k][1][key]
        if key in d_dict:
            d_dict[key].append(d)
        else:
            d_dict[key] = [d]

for m in ["Logistic Regression", 'Support Vector Machine', 'Random Forest']:
    plt.figure()
    for i in d_dict:
        y_list = [v[m] for v in d_dict[i]]
        plt.plot(Klist, y_list, label=i + " " + m)

    plt.xlabel('Number of Features')
    plt.ylabel('The accuracy classifier')
    plt.title(f'The accuracy of {m} classifier')
    plt.legend()
plt.show()