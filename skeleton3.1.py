# %%
from PIL import Image
import numpy as np
import math
np.random.seed(seed=2023)

# %%
# convert a RGB image to grayscale
# input (rgb): numpy array of shape (H, W, 3)
# output (gray): numpy array of shape (H, W)


def rgb2gray(rgb):
    ##############################################################################################
    #										IMPLEMENT											 #
    ##############################################################################################
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.1913 * r + 0.4815 * g + 0.3272 * b
    
    return gray

# %%
# load the data
# input (i0_path): path to the first image
# input (i1_path): path to the second image
# input (gt_path): path to the disparity image
# output (i_0): numpy array of shape (H, W, 3)
# output (i_1): numpy array of shape (H, W, 3)
# output (g_t): numpy array of shape (H, W)


def load_data(i0_path, i1_path, gt_path):

    ##############################################################################################
    #										IMPLEMENT											 #
    ##############################################################################################
    i_0 = np.array(Image.open(i0_path), dtype = np.float64) / 255
    i_1 = np.array(Image.open(i1_path), dtype = np.float64) / 255
    g_t = np.array(Image.open(gt_path),dtype = np.float64)

    return i_0, i_1, g_t

# %%
# image to the size of the non-zero elements of disparity map
# input (img): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (img_crop): numpy array of shape (H', W')


def crop_image(img, d):

    ##############################################################################################
    #										IMPLEMENT											 #
    ##############################################################################################
    x_min, x_max = np.min(np.nonzero(d)[0]), np.max(np.nonzero(d)[0])
    y_min, y_max = np.min(np.nonzero(d)[1]), np.max(np.nonzero(d)[1])
    img_crop = img[x_min:x_max , y_min:y_max]
    
    return img_crop



# %%
# shift all pixels of i1 by the value of the disparity map
# input (i_1): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (i_d): numpy array of shape (H, W)


def shift_disparity(i_1, d):
    i_d = np.zeros_like(i_1)
    for index, x in np.ndenumerate(d):
        i_d[index] = i_1[(index[0],index[1]+int(d[index]))] 
        
    ##############################################################################################
    #										IMPLEMENT											 #
    ##############################################################################################
    
    return i_d

# %%
# compute the negative log of the Gaussian likelihood
# input (i_0): numpy array of shape (H, W)
# input (i_1_d): numpy array of shape (H, W)
# input (mu): float
# input (sigma): float
# output (nll): numpy scalar of shape ()


def gaussian_nllh(i_0, i_1_d, mu, sigma):
    N , M =  i_0.shape
    nll =  (1/(2*(np.power(sigma,2))) ) * np.sum(np.power(i_0 - i_1_d - mu,2))
    # We can probably irgnore the constant 
    const = N*M*np.log(2*np.pi*sigma)
        
    return nll

# %%
# compute the negative log of the Laplacian likelihood
# input (i_0): numpy array of shape (H, W)
# input (i_1_d): numpy array of shape (H, W)
# input (mu): float
# input (s): float
# output (nll): numpy scalar of shape ()


def laplacian_nllh(i_0, i_1_d, mu,s):
    N,M  = i_0.shape
    nll = np.sum(np.absolute(i_0-i_1_d-mu)) * 1/s 
    const = N*M*np.log(1/2*s)
    
    return nll

# %%
# replace p% of the image pixels with values from a normal distribution
# input (img): numpy array of shape (H, W)
# input (p): float
# output (img_noise): numpy array of shape (H, W)


def make_noise(img, p):
    w, h = img.shape
    img = img.flatten()
    chosen_indexes = np.random.choice(img.shape[0], int(img.shape[0] * p) )
    rand_vals = np.random.normal(0.45, 0.14, size= int(img.shape[0] * p))
    img[chosen_indexes] = rand_vals
    img_noise = img.reshape(w, h)
    
    return img_noise

# %%
# apply noise to i1_sh and return the values of the negative lok-likelihood for both likelihood models with mu, sigma, and s
# input (i0): numpy array of shape (H, W)
# input (i1_sh): numpy array of shape (H, W)
# input (noise): float
# input (mu): float
# input (sigma): float
# input (s): float
# output (gnllh) - gaussian negative log-likelihood: numpy scalar of shape ()
# output (lnllh) - laplacian negative log-likelihood: numpy scalar of shape ()
def get_nllh_for_corrupted(i_0, i_1_d, noise, mu, sigma, s):

	# Make noisy img 
	img_noise = make_noise(i_1_d, noise)

	# Gaussian and Laplacian likelihood
	gnllh = gaussian_nllh(i_0, img_noise, mu, sigma)
	lnllh = laplacian_nllh(i_0, img_noise, mu, s)

	return gnllh, lnllh

# %%
# load images
i0, i1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
i0, i1 = rgb2gray(i0), rgb2gray(i1)
# shift i1
i1_sh = shift_disparity(i1, gt)

# crop images
i0 = crop_image(i0, gt)
i1_sh = crop_image(i1_sh, gt)

mu = 0.0
sigma = 1.4
s = 1.4
for noise in [0.0, 14.0, 27.0]:
    print("Noise : " , noise)
    gnllh, lnllh = get_nllh_for_corrupted(i0, i1_sh, noise, mu, sigma, s)
    print("GNLLH :" , gnllh)
    print("LNLLH :" , lnllh)

# %%


# %%



# %%



# %%


# %%


# %%


# %%



