import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

def normalize(v:np.array):
    return v / (np.linalg.norm(v) + 0.0000001)

def normalize_img(img, new_max = 255.0, new_min = 0.0):
    min = np.amin(img)
    max = np.amax(img)
    
    return (img-min)*((new_max - new_min)/(max-min)) + new_min

def print_debug(array: np.ndarray):
    print('value min : ' + str(np.amin(array)))
    print('value max : ' + str(np.amax(array)))
    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.show()

def structure_tensor_calculation(image_input):

    image_input = image_input.astype(float)

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

    S = [E,F,G]
    
    return S

def merge(S1, S2):
    # Merge 2 structures tensor
    E1 = S1[0]
    F1 = S1[1]
    G1 = S1[2]

    E2 = S2[0]
    F2 = S2[1]
    G2 = S2[2]

    result = S1.copy()
    result[0] = (E1 + E2) / 2.0
    result[1] = (F1 + F2) / 2.0
    result[2] = (G1 + G2) / 2.0

    return result

def structure_tensor_smoothing(S):
    # Apply Gaussian kernel on sturcture tensor
    matrice = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], float)
    kernel = 1.0/16.0 * matrice

    S[0] = cv2.filter2D(S[0],-1,kernel)
    S[1] = cv2.filter2D(S[1],-1,kernel)
    S[2] = cv2.filter2D(S[2],-1,kernel)
    
    return S

def compute_eigenvalue(S):

    E = S[0]
    F = S[1]
    G = S[2]

    lambda1 = np.zeros(E.shape)
    lambda2 = np.zeros(E.shape)
    coeff1 = np.zeros(E.shape)
    determinant = np.zeros(E.shape)

    # maximum and minimum eigenvalues respectively
    height, width = E.shape
    for i in range(height):
        for j in range(width):
            coeff1[i][j] = float(E[i][j] + G[i][j])
            determinant[i][j] = float((E[i][j] - G[i][j])*(E[i][j] - G[i][j]) + 4 * F[i][j]*F[i][j])

            lambda1[i][j] = (coeff1[i][j] + np.sqrt(determinant[i][j])) / 2.0
            lambda2[i][j] = (coeff1[i][j] - np.sqrt(determinant[i][j])) / 2.0

    return lambda1, lambda2

def compute_eigenvector(S, lambda1, lambda2):

    E = S[0]
    F = S[1]

    # maximum and minimum eigenvectors respectively
    height, width = E.shape
    eta = np.zeros((height,width,2))
    xi  = np.zeros((height,width,2))
    
    for i in range(height):
        for j in range(width):
            eta[i][j] = normalize(np.array([F[i][j], (lambda1[i][j] - E[i][j])]))
            xi[i][j] = normalize([(lambda1[i][j] - E[i][j]),-F[i][j]])

    return eta, xi

def euler_integration(img, xi, a_sigma):
    img_result = np.zeros(img.shape)
    height, width,_ = img.shape
    for i in range(height):
        for j in range(width):
            # compute l
            l = math.ceil(2*a_sigma[i][j])

            # compute each t_k and x_k
            t_minus = []
            t_plus = []
            t_minus.append(-xi[i][j])
            t_plus.append(xi[i][j])
            x_minus = []
            x_plus = []
            x_minus.append(np.array([i,j],dtype=int))
            x_plus.append(np.array([i,j],dtype=int))

            for k in range(1,l+1):
                # sign<t_k-1,xi> * xi
                t_minus.append(np.sign(np.dot(t_minus[k-1],xi[x_minus[k-1][0]][x_minus[k-1][1]])) * xi[x_minus[k-1][0]][x_minus[k-1][1]])
                t_plus.append(np.sign(np.dot(t_plus[k-1],xi[x_plus[k-1][0]][x_plus[k-1][1]])) * xi[x_plus[k-1][0]][x_plus[k-1][1]])
                x_minus.append(np.add(x_minus[k-1], t_minus[k-1]).astype("int"))
                x_plus.append(np.add(x_plus[k-1], t_plus[k-1]).astype("int"))

            np.asarray(x_minus)
            np.asarray(x_plus)
            
            # create 1 dimensional gaussian with a_sigma
            kernel = cv2.getGaussianKernel( 2*l+1, a_sigma[i][j])
            # apply the gaussian kernel with the x_k
            v_result = kernel[l]*img[x_minus[0][0]][x_minus[0][1]]
            for k in range(1,l+1):
                v_result = v_result + kernel[l+k]* img[x_plus[k][0]][x_plus[k][1]]
                v_result = v_result + kernel[l-k]* img[x_minus[k][0]][x_minus[k][1]]
            img_result[i][j] = np.uint8(v_result)
      
    return img_result

def adaptive_smoothing(img_input,lambda1,lambda2,xi,n = 1000.0,sigma = 6.0):
    # compute the adaptive gaussian sigma
    A = np.zeros(lambda1.shape)
    adapted_sigma = np.zeros(lambda1.shape)

    height, width = lambda1.shape
    alpha = 0.1
    for i in range(height):
        for j in range(width):
            if(lambda1[i][j]-lambda2[i][j] == 0):
                A[i][j] = alpha
            else:
                A[i][j] = alpha + (1-alpha) * math.exp((-n)/(lambda1[i][j]-lambda2[i][j]))
            adapted_sigma[i][j] = 1/4 * sigma * (1+A[i][j])**2

    # euler integration
    img_input = img_input.astype(float)

    img_input = euler_integration(img_input, xi, adapted_sigma)

    # print_debug(np.uint8(img_input))
    return img_input

def laplacian_of_gaussian(img):
    img_grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img_grey,(3,3),0)
    laplacian = cv2.Laplacian(blur,cv2.CV_64F)
    return laplacian

def gradient(img):

    img_grey = cv2.cvtColor(np.uint8(img),cv2.COLOR_RGB2GRAY)
    img_grey = img_grey.astype(float)

    kernel = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]],float)
    dx = kernel/2.0
    dy = kernel.transpose()/2.0
    fx = cv2.filter2D(img_grey,-1,dx)
    fy = cv2.filter2D(img_grey,-1,dy)

    return (abs(fx)+abs(fy))/2.0

def sharpening(img_input, nb_iter = 3):
    img = img_input.copy()
    img_new = np.zeros(img.shape, float)

    for k in range(0,nb_iter):
        LoG_sign = np.sign(laplacian_of_gaussian(np.uint8(img)))
        grad = gradient(img)

        height, width,_ = img.shape
        img_coeff = np.zeros(img.shape, float)

        for i in range(height):
            for j in range(width):
                img_coeff[i][j] = -LoG_sign[i][j] * grad[i][j] * 0.4
                temp = img[i][j] + img_coeff[i][j]
                img_new[i][j][0] = max(0.0,min(255.0,temp[0]))
                img_new[i][j][1] = max(0.0,min(255.0,temp[1]))
                img_new[i][j][2] = max(0.0,min(255.0,temp[2]))
        
        img = img_new.copy()
        print("   [INFO] sharpening -- "+str((k+1)/nb_iter*100) + "%")

    return img_new

def save_image(image, image_name):
    image = np.uint8(image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./result_'+image_name, image)

def show_image(image):
    #print the image
    cv2.imshow("image",image)
    cv2.waitKey()

def print_eigen_vector(vector_field):

    height, width, _ = vector_field.shape

    X, Y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
    
    U = np.zeros(X.shape)
    V = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U[i,j] = vector_field[i][j][0]*20.0
            V[i,j] = vector_field[i][j][1]*20.0

    _, ax = plt.subplots()
    skip_scalar = 5
    skip = (slice(None, None, skip_scalar), slice(None, None, skip_scalar))
    ax.quiver(X[skip], Y[skip], U[skip], V[skip], units='xy' ,scale=2, color='red')

    ax.set(aspect=1, title='Quiver Plot')

    plt.show()

def main():
    # Fill with the name of the image
    image_name = 'lesinge.png'
    # Choose the number of iteration
    nb_iter = 3

    img = cv2.imread('./images/'+image_name)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    result_img = img.copy()

    for i in range(0,nb_iter):
        print("\n[INFO] Iteration "+ str(i+1))

        if(i != 0):
            img_gray = cv2.cvtColor(np.uint8(result_img),cv2.COLOR_RGB2GRAY)
            S_new = structure_tensor_calculation(img_gray)
            S = merge(S, S_new)
            S = structure_tensor_smoothing(S)
            print("[INFO] Merge tensor structure -- done")
        else :
            img_gray = cv2.cvtColor(np.uint8(result_img),cv2.COLOR_RGB2GRAY)
            S = structure_tensor_calculation(img_gray)
            S = structure_tensor_smoothing(S)
            print("[INFO] Compute tensor structure -- done")

        lambda1,lambda2 = compute_eigenvalue(S)
        print("[INFO] Compute eigenvalues -- done")

        eta, xi = compute_eigenvector(S,lambda1, lambda2)
        print("[INFO] Compute eigenvectors -- done")

        # print_eigen_vector(xi)

        result_img = adaptive_smoothing(result_img,lambda1,lambda2,xi)
        print("[INFO] adaptive_smoothing -- done")

        result_img = sharpening(result_img,1)
        print("[INFO] sharpening -- done")
    
    save_image(result_img, image_name)

# ---------------------------------------------------------------------

start = timeit.default_timer()

main()

stop = timeit.default_timer()

print('Time :', stop - start, "s")