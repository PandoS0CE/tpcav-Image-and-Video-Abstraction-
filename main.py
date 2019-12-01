import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

def normalize(v: np.ndarray):
    return v/(np.linalg.norm(v)+0.00001)

def structure_tensor_calculation(image_input):
    height,width = image_input.shape
    # compute gaussian + sobel filter
    p1 = 0.183
    matrice = np.array([[p1, 0, -p1], [1-2*p1, 0, 2*p1-1], [p1, 0, -p1]], float)
    dx = 1/2 * matrice
    dy = np.transpose(dx)
    
    # apply filter
    fx = cv2.filter2D(image_input,-1,dx)
    fy = cv2.filter2D(image_input,-1,dy)

    # compute the structure tensor
    # pour chaque pixel structure de tenseur = tableau 2*2 avec produit scalaire pour chaque pixel
    E = np.zeros(image_input.shape)
    F = np.zeros(image_input.shape)
    G = np.zeros(image_input.shape)
    
    for i in range(height):
        for j in range(width):
            E[i][j] = np.dot(fx[i][j], fx[i][j])
            F[i][j] = np.dot(fx[i][j], fy[i][j])
            G[i][j] = np.dot(fy[i][j], fy[i][j])

    S = [[E,F],[F,G]]

    return S

def compute_eigenvalue(S):

    E = S[0][0]
    F = S[1][0]
    G = S[1][1]

    lambda1 = np.zeros(E.shape)
    lambda2 = np.zeros(E.shape)

    # maximum and minimum eigenvalues respectively
    height, width = E.shape
    for i in range(height):
        for j in range(width):
            lambda1[i][j] = (E[i][j] + G[i][j] + math.sqrt((E[i][j] - G[i][j])**2 + 4 * F[i][j]**2))/2
            lambda2[i][j] = (E[i][j] + G[i][j] - math.sqrt((E[i][j] - G[i][j])**2 + 4 * F[i][j]**2))/2

    return lambda1, lambda2

def compute_eigenvector(S, lambda1, lambda2):

    E = S[0][0]
    F = S[1][0]

    # maximum and minimum eigenvectors respectively
    height, width = E.shape
    eta = np.zeros((height,width,2))
    xi  = np.zeros((height,width,2))
    
    for i in range(height):
        for j in range(width):
            eta[i][j] = normalize(np.array([F[i][j], (lambda1[i][j] - E[i][j])]))
            xi[i][j] = normalize([(lambda1[i][j] - E[i][j]),-F[i][j]])

    return eta, xi

def adaptative_smoothing(img_input,lambda1,lambda2,xi,sigma = 6):
    # compute the adaptative gaussian sigma
    A = np.zeros(lambda1.shape)
    adapted_sigma = np.zeros(lambda1.shape)

    # compute an adaptive sigma for each pixel
    height, width = lambda1.shape
    for i in range(height):
        for j in range(width):
            A[i][j] = (lambda1[i][j] - lambda2[i][j])/(lambda1[i][j] + lambda2[i][j])
            adapted_sigma[i][j] = 1/4 * sigma * (1+A[i][j])**2

    # 

def save_image(image, image_name):
    cv2.imwrite('./result_'+image_name, image)

def show_image(image):
    #print the image
    cv2.imshow("image",image)
    cv2.waitKey()

def main():
    image_name = 'dogs.jpg'
    img = cv2.imread('./images/'+image_name)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    S = structure_tensor_calculation(img_gray)

    lambda1,lambda2 = compute_eigenvalue(S)

    eta, xi = compute_eigenvector(S,lambda1, lambda2)
    
    # show_image(img_gray)
    # save_image(img_gray,image_name)

# ---------------------------------------------------------------------
start = timeit.default_timer()

main()

stop = timeit.default_timer()

print('Time :', stop - start, "s")  

# TODO

# continuer adaptative smoothing pour obtenir une image