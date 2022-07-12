import numpy as np
from sklearn import svm

#import numpy as np
#import matplotlib.pyplot as plt
def sigmoid(z):
    a=np.exp(-z)
    a=1+a
    a=1/a
    return a

data=np.loadtxt("../input/train.csv",delimiter=',',skiprows=1)

m=np.shape(data)[0]
n=np.shape(data)[1]
x=data[:,1:n-1]            # excluded ids and final ans
y=data[:,n-1]             #  contains final ans

x_2=x**2            #MODEL2 X+X**2
x=np.column_stack((x,x_2))
#x_3=x**3            #MODEL2 X+X**2
#x=np.column_stack((x,x_3))
#x_4=x**4            #MODEL2 X+X**2
#x=np.column_stack((x,x_4))
n=np.shape(x)[1]
#MODEL 4 IS MODIFIED MORE COMPLEX MODEL 2
print ("reached 1")
for q in range(0,n-1):
#    for w in range(q+1):
#        temp=(x[:,q])*(x[:,w])
#        x=np.column_stack((x,temp))
     temp=(x[:,q])*(x[:,q+1])
     x=np.column_stack((x,temp))
                    

temp=x[:,0]
for i in range(1,n):
    temp=temp*x[:,i]

x=np.column_stack((x,temp))
n=np.shape(x)[1]

#n=np.shape(x)[1]
print ("reached 2")

y_orig=y
# feature scale the data
x_avg=np.mean(x,0)
x_std=np.std(x,0)
a=(x_std==0)
#b=(a==False)
#x=x[:,b]
x_std[a]=1
#x_avg=np.mean(x,0)
#x_std=np.std(x,0)
x=(x-x_avg)/x_std

#y1=(y==1)
#modify y into m*7
#for i in range(2,8):
#    y2=(y==i)
#    y1=np.column_stack((y1,y2))

#y=y1

# divide 80-20
a=np.arange(0,m)
np.random.shuffle(a)
m_train=int(0.8*m)
a1=a[0:m_train]
a2=a[m_train:]
x_train=x[a1]
x_test=x[a2]
y_train=y[a1]
y_test=y[a2]
m_test=np.shape(x_test)[0]

# add coloumn of 1
o=np.ones(m_train)
x_train=np.column_stack((o,x_train))
o=np.ones(m_test)
x_test=np.column_stack((o,x_test))
n=np.shape(x_train)[1]

print ("reached 3")

clf=svm.SVC(gamma=0.01,C=10)

clf.fit(x_train,y_train)

print ("reached 4")

pred=clf.predict(x_test)
acc=sum(pred==y_test)
acc=float(acc)
acc=(acc/m_test)*100
print ("svm acc is" ,acc,"%")
'''
from sklearn.neighbors import KNeighborsClassifier as knn
model=knn()
model.fit(x_train,y_train)
pred=model.predict(x_cv)
acc=sum(pred==y_cv)
acc=float(acc)
acc=(acc/m_cv)*100
print 'knn acc is' ,acc,'%'
'''