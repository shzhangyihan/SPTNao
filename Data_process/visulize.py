import numpy as np
import matplotlib.pyplot as plt
import csv

plt.cla()
major_ticks = np.arange(0, 1200, 100)                                              
minor_ticks = np.arange(0, 1200, 1)                                               

# both
f = open('nao_data_both.csv', 'rb')
dataReader = csv.reader(f)
array = []
i = 1
last = np.asfarray(np.zeros([8]))
for row in dataReader:
    array.append(row)
data = np.asarray(array)

for i in range(8):
    p = plt.subplot(6, 4, i + 1)
    plt.cla()
    plt.grid(True)
    plt.axis([0, 1200, 0, 1])
    p.plot(data[:, i], color = 'b')

# right
f = open('nao_data_right.csv', 'rb')
dataReader = csv.reader(f)
array = []
for row in dataReader:
    array.append(row)
data = np.asarray(array)

for i in range(8):
    p = plt.subplot(6, 4, i + 9)
    plt.cla()
    plt.grid(True)
    plt.axis([0, 1200, 0, 1])
    p.plot(data[:, i], color = 'b')

# left
f = open('nao_data_left.csv', 'rb')
dataReader = csv.reader(f)
array = []
for row in dataReader:
    array.append(row)
data = np.asarray(array)

for i in range(8):
    p = plt.subplot(6, 4, i + 17)
    plt.cla()
    plt.grid(True)
    plt.axis([0, 1200, 0, 1])
    p.plot(data[:, i], color = 'b')

plt.draw()
plt.ioff()
plt.show()
