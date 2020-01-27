import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

def normalize(v: np.ndarray):
    return v/(np.linalg.norm(v)+0.00001)

def print_debug(array: np.ndarray):
    print('value min : ' + str(np.amin(array)))
    print('value max : ' + str(np.amax(array)))
    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.show()

def structure_tensor_calculation(image_input):

    image_input = image_input.astype(float)

    # print(type(image_input[0][0]))

    # print_debug(image_input)

    height,width = image_input.shape
    # compute rotational symmetric derivative filter
    p1 = 0.183
    matrice = np.array([[p1, 0, -p1], [1-2*p1, 0, 2*p1-1], [p1, 0, -p1]], float)
    dx = 1.0/2.0 * matrice
    dy = np.transpose(dx)

    # apply filter
    fx = cv2.filter2D(image_input,-1,dx)
    fy = cv2.filter2D(image_input,-1,dy)

    # compute the structure tensor
    E = np.zeros(image_input.shape)
    F = np.zeros(image_input.shape)
    G = np.zeros(image_input.shape)
    
    for i in range(height):
        for j in range(width):
            E[i][j] = fx[i][j] * fx[i][j]
            F[i][j] = fx[i][j] * fy[i][j]
            G[i][j] = fy[i][j] * fy[i][j]

    S = [[E,F],[F,G]]

    # print_debug(fx)
    # print_debug(fy)

    # print_debug(E)
    # print_debug(F)
    # print_debug(G)

    return S

def compute_eigenvalue(S):

    E = S[0][0]
    F = S[1][0]
    G = S[1][1]

    # print(E[288][185])
    # print(F[288][185])
    # print(G[288][185])

    lambda1 = np.zeros(E.shape)
    lambda2 = np.zeros(E.shape)
    coeff1 = np.zeros(E.shape)
    determinant = np.zeros(E.shape)

    # maximum and minimum eigenvalues respectively
    height, width = E.shape
    for i in range(height):
        for j in range(width):
            if(F[i][j] > -1.0 and F[i][j] < 1.0):
                lambda1[i][j] = 0.0
                lambda2[i][j] = 0.0
            else:
                coeff1[i][j] = E[i][j] + G[i][j]
                determinant[i][j] = float((E[i][j] - G[i][j])**2 + 4 * F[i][j]**2)

                lambda1[i][j] = (coeff1[i][j] + math.sqrt(determinant[i][j])) / 2.0
                lambda2[i][j] = (coeff1[i][j] - math.sqrt(determinant[i][j])) / 2.0
            

            # lambda1[i][j] = (E[i][j] + G[i][j] + math.sqrt((E[i][j] - G[i][j])**2 + 4 * F[i][j]**2))/2.0
            # lambda2[i][j] = (E[i][j] + G[i][j] - math.sqrt((E[i][j] - G[i][j])**2 + 4 * F[i][j]**2))/2.0

    # print_debug(coeff1)
    # print_debug(determinant)

    # print_debug(lambda1)
    # print_debug(lambda2)

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

def add_padding(img,size):
    borderType = cv2.BORDER_REPLICATE

    top = int(size)
    bottom = top
    left = int(size)
    right = left

    img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, None)

def euler_integration(img, xi, a_sigma):
    img_result = img.copy()

    size = 15
    add_padding(img,size)

    height, width,_ = img.shape
    l = np.zeros((img_result.shape[0], img_result.shape[1]), dtype=np.uint8)
    for i in range(size,height+size):
        for j in range(size,width+size):
            i_pad = i - size
            j_pad = j - size

            # compute l
            l[i_pad][j_pad] = math.ceil(2*a_sigma[i_pad][j_pad])

            # compute each t_k and x_k
            t_minus = []
            t_plus = []
            t_minus.append(-xi[i_pad][j_pad])
            t_plus.append(xi[i_pad][j_pad])
            x_minus = []
            x_plus = []
            x_minus.append(np.array([i,j]))
            x_plus.append(np.array([i,j]))

            for k in range(1,l[i_pad][j_pad]):
                # sign<t_k-1,xi> * xi
                t_minus.append(np.dot(np.sign(np.dot(t_minus[k-1],xi[i_pad][j_pad])),xi[i_pad][j_pad]))
                t_plus.append(np.dot(np.sign(np.dot(t_plus[k-1],xi[i_pad][j_pad])),xi[i_pad][j_pad]))
                x_minus.append(np.add(x_minus[k-1], t_minus[k-1]*10).astype("int"))
                x_plus.append(np.add(x_plus[k-1], t_plus[k-1]*10).astype("int"))

            
            np.asarray(x_minus,dtype=np.int)
            np.asarray(x_plus,dtype=np.int)
            # print(x_minus)

            # create 1 dimensional gaussian with a_sigma
            kernel = cv2.getGaussianKernel( 2*l[i_pad][j_pad]+1, a_sigma[i_pad][j_pad])
            # apply the gaussian kernel with the x_k
            K_norm = np.sum(kernel)
            v_result = kernel[l[i_pad][j_pad]][0]*img[x_minus[0][0]][x_minus[0][1]]
            # print(x_minus)
            for k in range(1,l[i_pad][j_pad]):
                v_result = v_result + kernel[l[i_pad][j_pad]-k][0]*img[x_minus[k][0]][x_minus[k][1]] #probleme d'acces a cet endroit
                v_result = v_result + kernel[l[i_pad][j_pad]+k][0]*img[x_plus[k][0]][x_plus[k][1]]
            v_result = v_result / K_norm
            img_result[i_pad][j_pad] = v_result
    return img_result

def adaptative_smoothing(img_input,lambda1,lambda2,xi,eta,S,sigma = 6):
    # compute the adaptative gaussian sigma
    A = np.zeros(lambda1.shape)
    adapted_sigma = np.zeros(lambda1.shape)

    # Debug area --------------
    # E = S[0][0]
    # F = S[1][0]
    # G = S[1][1]

    height, width = lambda1.shape
    alpha = 0.1
    for i in range(height):
        for j in range(width):
            if(lambda1[i][j]-lambda2[i][j] == 0):
                A[i][j] = alpha
            else:
                A[i][j] = alpha + (1-alpha) * math.exp((-1000.0)/(lambda1[i][j]-lambda2[i][j]))
            adapted_sigma[i][j] = 1/4 * sigma * (1+A[i][j])**2

    #--------------------------

    # compute an adaptive sigma for each pixel
    # height, width = lambda1.shape
    # for i in range(height):
    #     for j in range(width):
    #         if(lambda1[i][j] + lambda2[i][j] != 0.0):
    #             A[i][j] = (lambda1[i][j] - lambda2[i][j])/(lambda1[i][j] + lambda2[i][j])
    #         else:
    #             A[i][j] = 1.0
    #         adapted_sigma[i][j] = 1/4 * sigma * (1+A[i][j])**2

    print_debug(A)

    # euler integration
    # euler_integration(img_input, xi, adapted_sigma)

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
    print("[INFO] Compute tensor structure -- done")

    lambda1,lambda2 = compute_eigenvalue(S)
    print("[INFO] Compute eigenvalues -- done")

    # cv2.imshow("kes",cv2.normalize(lambda1,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U))
    # cv2.waitKey(0)

    eta, xi = compute_eigenvector(S,lambda1, lambda2)
    print("[INFO] Compute eigenvectors -- done")

    adaptative_smoothing(img,lambda1,lambda2,xi,eta,S)
    print("[INFO] adaptive_smoothing -- done")
    
    # show_image(S[1][1])
    save_image(img_gray,image_name)

# ---------------------------------------------------------------------

start = timeit.default_timer()

main()

stop = timeit.default_timer()

print('Time :', stop - start, "s")  

# TODO

# continuer adaptative smoothing pour obtenir une image