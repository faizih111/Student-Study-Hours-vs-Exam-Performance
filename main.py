import numpy
from scipy import stats
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scale = StandardScaler()

StudyHours = [2, 3, 4, 5, 6, 7, 8, 9]
SleepHours = [8, 7, 7, 6, 6, 5, 5, 4]
ExamScore = [50, 55, 60, 65, 70, 78, 85, 90]

x = numpy.random.normal(60, 80, 250)
plt.hist(x)
plt.show()

Mean = numpy.mean(ExamScore)
Median = numpy.median(ExamScore)
Mode = stats.mode(ExamScore)

print(f"Your mean value : {Mean}")
print(f"Your median value: {Median}")
print(f"Your mode value : {Mode}\n")

stddev = numpy.std(ExamScore)
print(f"Standard value : {stddev}\n")

perc = numpy.percentile(ExamScore, 75)
print(f"75th percentile → 75% of students scored below this : {perc}\n")

plt.scatter(StudyHours, ExamScore)
plt.show()

slope, intercept, r, p, std_err = stats.linregress(ExamScore, StudyHours)


def myfunc(StudyHours):
    return slope * StudyHours + intercept


mymodel = list(map(myfunc, StudyHours))

plt.scatter(StudyHours, ExamScore)
plt.plot(StudyHours, mymodel)
plt.show()

mymodel = numpy.poly1d(numpy.polyfit(StudyHours, ExamScore, 3))
myline = numpy.linspace(1, 22, 100)

plt.scatter(StudyHours, ExamScore)
plt.plot(myline, mymodel(myline))
plt.show()

X = [[70, 6], [80, 7], [85, 5], [90, 8], [95, 6]]
y = [3, 4, 3, 5, 4]

regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr, "\n")

scaledX = scale.fit_transform(X)
print(scaledX)

x = [1, 3, 5, 7, 9]
y = [2, 4, 6, 8, 10]

train_x = x[:3]
train_y = y[:3]

test_x = x[3:]
test_y = y[3:]
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))


myline = numpy.linspace(0, 6, 100)
plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()
