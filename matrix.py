'''
arr1=[1,2,3,4,5]
print(arr1)
arr1.append(6)
print(arr1)
'''
# below is just the matrix basics

import numpy as np

'''
mat1=[[1,2,3],[4,5,6],[7,8,9]]  # this is a list in python
#print(mat1, '\n')
mat2 = np.array(mat1)  # this is an array in python using numpy module
#print(mat2.shape)
mat2
'''

'''
#np.array attributes. ndim, shape, size, dtype
print(mat2.ndim)   # shows the n dimensions of the array. like 1d,2d,3d...
print(mat2.shape)  # shows the number or elements along the dimensions. like 1x1,2x2,3x3...
print(mat2.size)   # total elements in the array
print(mat2.dtype)  # data type of the array
'''

# Basic array creation

# zeros, ones, empty, arange, linspace
'''
print(np.zeros(2))
print(np.ones(2,dtype=np.int64))
print(np.empty(2))
print(np.arange(4))
print(np.arange(1,10,2))
print(np.linspace(1,10,num=4,dtype=np.int64))
'''

# one way for array creation is list -> array using np.array(list name)
'''
matrix = [[1,2],[3,4]]

matrix_array = np.array(matrix)
print(matrix_array)
'''
'''
num = 1
matrix = []

for i in range(5):
	matrix.append([])
	for j in range(5):
		matrix[i].append(num)
		num+=1

matrix_array = np.array(matrix)
print(matrix_array)
'''

# another way is empty array -> filling it
'''
num=1
matrix = np.empty((5,5),dtype='int64')  
for i in range(5):
	for j in range(5):
		matrix[i][j]=num
		num+=1
print(matrix)
'''

# matrix multiplication
'''
num1 = 1
num2 = 16
row = 2
col = 2

mat1 = np.empty((row,col),dtype='int64')
mat2 = np.empty((row,col),dtype='int64')

for i in range(row):
	for j in range(col):
		mat1[i][j] = num1
		mat2[i][j] = num2
		num1+=1
		num2-=1

# print("matrix 1:\n",mat1)
# print("matrix 2:\n",mat2)

mat3 = mat1*mat2         # this is multiplying same elemets as it is. not real matrix multiplication
#print("matrix 3:\n",mat3)

#manual calculation of matrix multiplication
result=np.empty([row,col],dtype='int64')

for i in range(len(mat1)):   # length of row mat1
	for j in range(len(mat2[0])):  # length of col mat2
		for k in range(len(mat2)):   # length of row of mat2
			result[i][j]+=mat1[i][k]*mat2[k][j]

print("matrix mul by 3 for loops:\n",result)
'''


# using inbuild functions and operators in numpy
'''
mat_mul = np.matmul(mat1,mat2)           # using matmul()
mat_mul = mat1@mat2                      # using @ operator
mat_mul = np.dot(mat1,mat2)              # using dot()
print("matrix multiplication using inbuild functions of numpy:\n",mat_mul)
'''








# convolution of a matrix using another matrix

def dim_of_conved_matrix(image,kernal,stride=1,padding=0):
	image = np.array(image)
	kernal = np.array(kernal)
	dim1 = image.shape
	dim2 = kernal.shape

	dimx = int(((dim1[0]-dim2[0]+(2*padding))/stride)+1)
	dimy = int(((dim1[1]-dim2[1]+(2*padding))/stride)+1)

	print(f"your given image dimension: {dim1}\nyour given kernal dimension: {dim2}\nresulting convoluted dimension: {dimx}x{dimy} \nstride: {stride} \npadding: {padding}")

mat1 = [
    [4, 7, 2, 0, 5],
    [6, 3, 9, 8, 2],
    [1, 5, 4, 7, 3],
    [8, 2, 0, 9, 4],
    [7, 1, 3, 6, 0]
]
mat1 = np.array(mat1)

mat2 = [
    [3, 7, 1],
    [0, 5, 8],
    [6, 2, 4]
]
mat2 = np.array(mat2)


# it is working!!!
dim_of_conved_matrix(mat1,mat2,1,1)

np.convolve((1,2,3),(4,5,6))
