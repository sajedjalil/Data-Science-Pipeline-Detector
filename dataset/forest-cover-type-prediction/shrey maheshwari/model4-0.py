# forest cover
import numpy as np
import matplotlib.pyplot as plt
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
#for q in range(0,n-1):
#    for w in range(q+1):
#        temp=(x[:,q])*(x[:,w])
#        x=np.column_stack((x,temp))
#     temp=(x[:,q])*(x[:,q+1])
#     x=np.column_stack((x,temp))

                    
print ("reached 2")


n=np.shape(x)[1]
temp=x[:,0]
for i in range(1,n):
    temp=temp*x[:,i]

x=np.column_stack((x,temp))
n=np.shape(x)[1]
y_orig=y
# feature scale the data
x_avg=np.mean(x,0)
x_std=np.std(x,0)
a=(x_std==0)
b=(a==False)
#x=x[:,b]
#x_std[a]=1
x_avg=np.mean(x[:,b],0)
x_std=np.std(x[:,b],0)
x[:,b]=(x[:,b]-x_avg)/x_std

y1=(y==1)
#modify y into m*7
for i in range(2,8):
    y2=(y==i)
    y1=np.column_stack((y1,y2))

y=y1
n_y=np.shape(y)[1]
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


# m_train,m_test,n,n_y

theta=np.zeros((n,n_y))
h=sigmoid(np.dot(x_train,theta))
lamb=0
cost=(sum((-y_train)*(np.log(h))-(1-y_train)*np.log(1-h)))/m_train
x_train_t=np.transpose(x_train)
grad=(np.dot(x_train_t,h-y_train))/m_train
alpha=0.01
j_hist=[]           # empty list so that can append

i=0


while(True):                
    theta[1:]=theta[1:]*(1-((alpha*lamb)/m_train))-alpha*grad[1:] #update rule
    theta[0]=theta[0]-alpha*grad[0]                               #update rule for theta0
    h=sigmoid(np.dot(x_train,theta))
    grad=(np.dot(x_train_t,h-y_train))/m_train
    #j_hist[i]=(sum((-y_train)*(np.log(h))-(1-y_train)*np.log(1-h)))/m_train
    #j_hist.append((sum((-y_train)*(np.log(h))-(1-y_train)*np.log(1-h)))/m_train)
    j_hist.append((sum((y_train)*(-np.log(h))-(1-y_train)*np.log(1-h),0))/m_train)
    if(i==600):
        itera=i    
        break
    i=i+1

print ("reached 3")    
            
j_hist=np.array(j_hist)    # converting list to array
x_axis=np.linspace(0,itera-1,200)
x_axis=x_axis.astype(int)        
plt.plot(x_axis,j_hist[x_axis],label='MODEL 1')
plt.legend(loc='upper right')
plt.show()

# prediction code on training data (80%)  MODEL 1

pred1=sigmoid(np.dot(x_train,theta))
ind=np.argmax(pred1,1)
for i in range(m_train):
    pred1[i]=0

for i in range(m_train):
    pred1[i,ind[i]]=1
qq=np.zeros(m_train)
for i in range(m_train):
    qq[i]=np.array_equal(pred1[i],y_train[i])

acc=sum(qq)
acc=float(acc)
acc=(acc/m_train)*100
print ("MODEL 1: train accuracy is" ,acc,"%")


# prediction code on test data (20%)  MODEL 1

pred2=sigmoid(np.dot(x_test,theta))
ind=np.argmax(pred2,1)
for i in range(m_test):
    pred2[i]=0

for i in range(m_test):
    pred2[i,ind[i]]=1
qq=np.zeros(m_test)
for i in range(m_test):
    qq[i]=np.array_equal(pred2[i],y_test[i])       
acc=sum(qq)
acc=float(acc)
acc=(acc/m_test)*100
print ("MODEL 1: test accuracy is" ,acc,"%")
np.set_printoptions(threshold=np.nan)
print (theta)
