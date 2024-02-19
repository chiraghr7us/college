import matplotlib.pyplot as plt
# Plot test versus prediction
x = [2,4,3,2,1,10,5,5,1,2,9,10,6,3,4]
print(len(x))
y_new=[]
yog=[20,60,46,41,12,137,68,89,4,32,144,156,93,36,72]
for i in x:
    y = -1.58 + (15.43 * i)
    y_new.append(y)
print(y_new)


# print(len(y))
# print("-----------")
# print(sum(x))
# print(sum(y))

plt.scatter(x, yog)
plt.plot(x, y_new, color='r', label="Estimated regression line")
plt.legend()
plt.xlabel("Number of copiers serviced")
plt.ylabel("Time for service")
plt.show()
