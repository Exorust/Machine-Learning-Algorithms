'''**********************************************
CODE TO IMPLEMENT FISHER'S LDA -
Given two dimensional dataset with two classes 0 and 1,
Perform Fisher's LDA on the dataset, 
Perform dimensionality reduction and find the suitable vector to project it onto,
Find the threshold value for separation of the two classes
***********************************************'''
import numpy as np
import matplotlib.pyplot as plt
import time


#  to calculate the execution time of th clustering
start_time = time.time()

# reading data csv file 
my_data = np.genfromtxt('datasets/dataset_3.csv', delimiter=',')

# deleting the serial number column
data=np.delete(my_data,0,1)

# separating the two classes and deleting the target variable column
class0 = data[np.nonzero(data[:,2] == 0)]
class1=data[np.nonzero(data[:,2]==1)]
class0=np.delete(class0,2,1)
class1=np.delete(class1,2,1)

# finding the mean of the the two classes ​
mean0=np.mean(class0,0)
mean1=np.mean(class1,0)

''' calculating the variability of the two classes using the formula :
    variability=summation over points belonging to class 1((xi-mean)(xi-mean)tanspose)
'''
var0=np.zeros(1)
temp=np.array(mean0)
for i in range (class0.shape[0]) :
    temp=(class0[i,:]-mean0)
    var0+=np.dot(temp, temp.T)
var1=np.zeros(1)
temp=np.array(mean1)
for i in range (class1.shape[0]) :
    temp=(class1[i,:]-mean1)
    var1+=np.dot(temp, temp.T)
sw=var1+var0

# calculating the inverse of Sw matrix
invsw=np.array([(1/sw[0])])

# calculating the w vector using below formula
w=invsw*(mean1-mean0)

# declaring arrays for storing points' distance from the vector
dist0=np.zeros((class0.shape[0],1))
dist1=np.zeros((class1.shape[0],1))

# finding the the vector to project the points on;
# such that the means are farthest from each other
wperp=np.array([-w[1],w[0]])

# finding the norm of the w vector
norm_w=np.linalg.norm(wperp)

''' calculating the distance of original data points from the vector using the formula:
    r=w.T/norm(w)
'''
for i in range(dist0.shape[0]):
    dist0[i]=np.dot(wperp.T,class0[i,:])/norm_w
for i in range(dist1.shape[0]):
    dist1[i]=np.dot(wperp.T,class1[i,:])/norm_w

''' declaring the arrays to store the projected points data using formula:
    x_projected = x_actual-r*w/norm(w)
'''
class0proj=np.zeros((class0.shape[0],2))
class1proj=np.zeros((class1.shape[0],2))
for i in range(class0.shape[0]):
    class0proj[i,:]=np.subtract((class0[i,:]),(dist0[i]*wperp.T/norm_w))
for i in range(class1.shape[0]):
    class1proj[i,:]=np.subtract((class1[i,:]),(dist1[i]*wperp.T/norm_w))

# displaying the plot with the original data , projected points and line​
plt.scatter(class0[:,0],class0[:,1])
plt.scatter(class1[:,0],class1[:,1])

plt.scatter(class0proj[:,0],class0proj[:,1],color='blue')
plt.scatter(class1proj[:,0],class1proj[:,1],color='red')

#concatenating the two classes into a single array
pointsproj=np.concatenate((class0proj,class1proj),axis=0)
plt.plot(pointsproj[:,0],pointsproj[:,1],'m')

# storing dimensionally reduced projected points in array using formula:
#  y(x) = w.T*x
newproj0=np.zeros((class0.shape[0],1))
newproj1=np.zeros((class1.shape[0],1))

for i in range(class0.shape[0]):
    newproj0[i,:]=np.dot(wperp.T,class0[i,:])
for i in range(class1.shape[0]):
    newproj1[i,:]=np.dot(wperp.T,class1[i,:])

# storing the means and standard deviations of the projected points
proj0mean=np.mean(newproj0)
proj1mean=np.mean(newproj1)

proj0std=np.std(newproj0)
proj1std=np.std(newproj1)

'''
 Below function "solve" to finds the threshold value separating the two 
 classes when dimensionally reduced -
 input : m1, m2 - means of the two classes whose point of intersection needs to be found
         std1, std2 - the standard deviations of the two classes
'''
def solve(m1,m2,std1,std2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    roots= np.roots([a,b,c])
   
    # since two possible points of intersection , we select the one which lies in between the two means 
    if roots.shape[0]>1:
        for i in range(2):
            if roots[i] !=max(m1,m2,roots[i]) or roots[i]!=min(m1,m2,roots[i]):
                return roots[i]
    else:
        return roots
        
threshold=solve(proj0mean,proj1mean,proj0std,proj1std)

print("Threshold value =", threshold)
print("Time taken = ",(time.time()-start_time))
plt.savefig('Results/Result3.png')


