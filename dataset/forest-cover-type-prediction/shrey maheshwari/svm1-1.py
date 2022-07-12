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
test=np.loadtxt("../input/test.csv",delimiter=',',skiprows=1)

m=np.shape(data)[0]
n=np.shape(data)[1]
x=data[:,1:n-1]            # excluded ids and final ans

m2=np.shape(test)[0]
n2=np.shape(test)[1]
x2=test[:,1:n]            # excluded ids and final ans

y=data[:,n-1]             #  contains final ans

x_2=x**2            #MODEL2 X+X**2
x=np.column_stack((x,x_2))
#x_3=x**3            #MODEL2 X+X**2
#x=np.column_stack((x,x_3))
#x_4=x**4            #MODEL2 X+X**2
#x=np.column_stack((x,x_4))
n=np.shape(x)[1]

x_2=x2**2            #MODEL2 X+X**2
x2=np.column_stack((x2,x_2))
n2=np.shape(x2)[1]
#MODEL 4 IS MODIFIED MORE COMPLEX MODEL 2
print ("reached 1")
#for q in range(0,n-1):
#    for w in range(q+1):
#        temp=(x[:,q])*(x[:,w])
#        x=np.column_stack((x,temp))
#     temp=(x[:,q])*(x[:,q+1])
#     x=np.column_stack((x,temp))
                    

temp=x[:,0]
for i in range(1,n):
    temp=temp*x[:,i]

x=np.column_stack((x,temp))
n=np.shape(x)[1]

temp=x2[:,0]
for i in range(1,n2):
    temp=temp*x2[:,i]


x2=np.column_stack((x2,temp))
n2=np.shape(x2)[1]


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

x_avg=np.mean(x2,0)
x_std=np.std(x2,0)
a=(x_std==0)
#b=(a==False)
#x=x[:,b]
x_std[a]=1
#x_avg=np.mean(x,0)
#x_std=np.std(x,0)
x2=(x2-x_avg)/x_std

#y1=(y==1)
#modify y into m*7
#for i in range(2,8):
#    y2=(y==i)
#    y1=np.column_stack((y1,y2))

#y=y1
'''
 divide 80-20
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
'''
o=np.ones(m)
x=np.column_stack((o,x))
o=np.ones(m2)
x2=np.column_stack((o,x2))
n2=np.shape(x2)[1]

print ("reached 3")

clf=svm.SVC(gamma=0.01,C=10)

clf.fit(x,y)

print ("reached 4")
#x2_1=x2[0:80000]
#x2_1=x2[80000:160000]
#x2_1=x2[160000:240000]
#x2_1=x2[240000:320000]
#x2_1=x2[320000:420000]
x2_1=x2[420000:]
pred=clf.predict(x2_1)
np.set_printoptions(threshold=np.nan)
print (pred)
#acc=sum(pred==y_test)
#acc=float(acc)
#acc=(acc/m_test)*100
#print ("svm acc is" ,acc,"%")
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