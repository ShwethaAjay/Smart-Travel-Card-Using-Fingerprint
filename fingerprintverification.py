
# Downloading the dataset
# !pip install kaggle
# !mkdir ~/.kaggle
# !cp kaggle.json ~/.kaggle
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d ruizgara/socofing
# !unzip /content/socofing.zip

# Downloading samples and helper functions
from os import path
# if not path.exists('utils.py'): # If running on colab: the first time download and unzip additional files
#     !wget https://biolab.csr.unibo.it/samples/fr/files.zip
#     !unzip files.zip

# Importing libraries
import os
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *
import shutil

# Transferring samples to the dataset
# shutil.copyfile("/content/samples/sample_1_2.png",'/content/socofing/SOCOFing/Real/sample_1_2.png')
# shutil.copyfile("/content/samples/sample_2.png",'/content/socofing/SOCOFing/Real/sample_2.png')

# os.chdir("/content/socofing/SOCOFing/Real")

"""### Helper functions"""

def compute_crossing_number(values):
    return np.count_nonzero(values < np.roll(values, -1))

def compute_next_ridge_following_directions(previous_direction, values):    
    next_positions = np.argwhere(values!=0).ravel().tolist()
    if len(next_positions) > 0 and previous_direction != 8:
        # There is a previous direction: return all the next directions, sorted according to the distance from it,
        #                                except the direction, if any, that corresponds to the previous position
        next_positions.sort(key = lambda d: 4 - abs(abs(d - previous_direction) - 4))
        if next_positions[-1] == (previous_direction + 4) % 8: # the direction of the previous position is the opposite one
            next_positions = next_positions[:-1] # removes it
    return next_positions

def follow_ridge_and_compute_angle(x, y,nd_lut,xy_steps, d = 8):
    px, py = x, y
    length = 0.0
    while length < 20: # max length followed
        next_directions = nd_lut[cn_values[py,px]][d]
        if len(next_directions) == 0:
            break
        # Need to check ALL possible next directions
        if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
            break # another minutia found: we stop here
        # Only the first direction has to be followed
        d = next_directions[0]
        ox, oy, l = xy_steps[d]
        px += ox ; py += oy ; length += l
    # check if the minimum length for a valid direction has been reached
    return math.atan2(-py+y, px-x) if length >= 10 else None

def Gs(t_sqr):
    mcc_sigma_s = 7.0
    mcc_tau_psi = 400.0
    mcc_mu_psi = 1e-2
    """Gaussian function with zero mean and mcc_sigma_s standard deviation, see eq. (7) in MCC paper"""
    return np.exp(-0.5 * t_sqr / (mcc_sigma_s**2)) / (math.tau**0.5 * mcc_sigma_s)

def Psi(v):
    mcc_sigma_s = 7.0
    mcc_tau_psi = 400.0
    mcc_mu_psi = 1e-2
    """Sigmoid function that limits the contribution of dense minutiae clusters, see eq. (4)-(5) in MCC paper"""
    return 1. / (1. + np.exp(-mcc_tau_psi * (v - mcc_mu_psi)))

"""### Step 1: Fingerprint Segmentation"""

def segmentation(image):
  # Calculate the local gradient (using Sobel filters)
  gx, gy = cv.Sobel(image, cv.CV_32F, 1, 0), cv.Sobel(image, cv.CV_32F, 0, 1)

  # Calculate the magnitude of the gradient for each pixel
  gx2, gy2 = gx**2, gy**2
  gm = np.sqrt(gx2 + gy2)

  # Integral over a square window
  sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize = False)

  # Use a simple threshold for segmenting the fingerprint pattern
  thr = sum_gm.max() * 0.2
  mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
  return mask, gx2, gy2, gx, gy

"""### Step 2: Estimation of local ridge orientation"""

def local_ridge_orientation(gx2, gy2, gx, gy):
  W = (23, 23)
  gxx = cv.boxFilter(gx2, -1, W, normalize = False)
  gyy = cv.boxFilter(gy2, -1, W, normalize = False)
  gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
  gxx_gyy = gxx - gyy
  gxy2 = 2 * gxy

  orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction
  sum_gxx_gyy = gxx + gyy
  strengths = np.divide(cv.sqrt((gxx_gyy**2 + gxy2**2)), sum_gxx_gyy, out=np.zeros_like(gxx), where=sum_gxx_gyy!=0)
  return orientations

"""### Step 3: Estimation of local ridge frequency"""

def local_ridge_frequency(image):
  region = image[10:90,80:130]

  # before computing the x-signature, the region is smoothed to reduce noise
  smoothed = cv.blur(region, (5,5), -1)
  xs = np.sum(smoothed, 1) # the x-signature of the region  
  x = np.arange(region.shape[0])

  # Find the indices of the x-signature local maxima
  local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
  x = np.arange(region.shape[0])

  # Calculate all the distances between consecutive peaks
  distances = local_maxima[1:] - local_maxima[:-1]

  # Estimate the ridge line period as the average of the above distances
  ridge_period = np.average(distances)

  return ridge_period

"""### Step 4: Fingerprint Enhancement"""

def fingerprint_enhancement(ridge_period,image,orientations,mask):
  # Create the filter bank
  or_count = 8
  gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]

  # Filter the whole image with each filter
  # Note that the negative image is actually used, to have white ridges on a black background as a result
  nf = 255-fingerprint
  all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])
  
  y_coords, x_coords = np.indices(image.shape)
  # For each pixel, find the index of the closest orientation in the gabor bank
  orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
  # Take the corresponding convolution result for each pixel, to assemble the final result
  filtered = all_filtered[orientation_idx, y_coords, x_coords]
  # Convert to gray scale and apply the mask
  enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
  return enhanced

"""### Step 5: Detection of minutiae positions"""

def detect_minutiae(enhanced,mask):
  # Binarization
  _, ridge_lines = cv.threshold(enhanced, 32, 255, cv.THRESH_BINARY)

  # Thinning
  skeleton = cv.ximgproc.thinning(ridge_lines, thinningType = cv.ximgproc.THINNING_GUOHALL)

  # Create a filter that converts any 8-neighborhood into the corresponding byte value [0,255]
  cn_filter = np.array([[  1,  2,  4],
                        [128,  0,  8],
                        [ 64, 32, 16]
                      ])
  
  # Create a lookup table that maps each byte value to the corresponding crossing number
  all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
  cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)

  # Skeleton: from 0/255 to 0/1 values
  skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
  # Apply the filter to encode the 8-neighborhood of each pixel into a byte [0,255]
  cn_values = cv.filter2D(skeleton01, -1, cn_filter, borderType = cv.BORDER_CONSTANT)
  # Apply the lookup table to obtain the crossing number of each pixel
  cn = cv.LUT(cn_values, cn_lut)
  # Keep only crossing numbers on the skeleton
  cn[skeleton==0] = 0

  # crossing number == 1 --> Termination, crossing number == 3 --> Bifurcation
  minutiae = [(x,y,cn[y,x]==1) for y, x in zip(*np.where(np.isin(cn, [1,3])))]

  # A 1-pixel background border is added to the mask before computing the distance transform
  mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,1:-1]

  filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]]>10, minutiae))
  return all_8_neighborhoods, filtered_minutiae, cn_values, cn,

"""### Step 6: Detection of minutiae directions"""

def minutiae_directions(all_8_neighborhoods,filtered_minutiae,cn_values,cn):
  r2 = 2**0.5 # sqrt(2)

  # The eight possible (x, y) offsets with each corresponding Euclidean distance
  xy_steps = [(-1,-1,r2),( 0,-1,1),( 1,-1,r2),( 1, 0,1),( 1, 1,r2),( 0, 1,1),(-1, 1,r2),(-1, 0,1)]

  # LUT: for each 8-neighborhood and each previous direction [0,8], 
  #      where 8 means "none", provides the list of possible directions
  nd_lut = [[compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]

  valid_minutiae = []
  for x, y, term in filtered_minutiae:
      d = None
      if term: # termination: simply follow and compute the direction        
          d = follow_ridge_and_compute_angle(x, y,nd_lut,xy_steps)
      else: # bifurcation: follow each of the three branches
          dirs = nd_lut[cn_values[y,x]][8] # 8 means: no previous direction
          if len(dirs)==3: # only if there are exactly three branches
              angles = [follow_ridge_and_compute_angle(x+xy_steps[d][0], y+xy_steps[d][1],nd_lut,xy_steps, d) for d in dirs]
              if all(a is not None for a in angles):
                  a1, a2 = min(((angles[i], angles[(i+1)%3]) for i in range(3)), key=lambda t: angle_abs_difference(t[0], t[1]))
                  d = angle_mean(a1, a2)                
      if d is not None:
          valid_minutiae.append( (x, y, term, d) )
  return valid_minutiae

"""### Step 7: Creation of local structures"""

def local_structures_calc(valid_minutiae):
  # Compute the cell coordinates of a generic local structure
  mcc_radius = 70
  mcc_size = 16

  g = 2 * mcc_radius / mcc_size
  x = np.arange(mcc_size)*g - (mcc_size/2)*g + g/2
  y = x[..., np.newaxis]
  iy, ix = np.nonzero(x**2 + y**2 <= mcc_radius**2)
  ref_cell_coords = np.column_stack((x[ix], x[iy]))

  mcc_sigma_s = 7.0
  mcc_tau_psi = 400.0
  mcc_mu_psi = 1e-2

  # n: number of minutiae
  # c: number of cells in a local structure

  xyd = np.array([(x,y,d) for x,y,_,d in valid_minutiae]) # matrix with all minutiae coordinates and directions (n x 3)

  # rot: n x 2 x 2 (rotation matrix for each minutia)
  d_cos, d_sin = np.cos(xyd[:,2]).reshape((-1,1,1)), np.sin(xyd[:,2]).reshape((-1,1,1))
  rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])

  # rot@ref_cell_coords.T : n x 2 x c
  # xy : n x 2
  xy = xyd[:,:2]
  # cell_coords: n x c x 2 (cell coordinates for each local structure)
  cell_coords = np.transpose(rot@ref_cell_coords.T + xy[:,:,np.newaxis],[0,2,1])

  # cell_coords[:,:,np.newaxis,:]      :  n x c  x 1 x 2
  # xy                                 : (1 x 1) x n x 2
  # cell_coords[:,:,np.newaxis,:] - xy :  n x c  x n x 2
  # dists: n x c x n (for each cell of each local structure, the distance from all minutiae)
  dists = np.sum((cell_coords[:,:,np.newaxis,:] - xy)**2, -1)

  # cs : n x c x n (the spatial contribution of each minutia to each cell of each local structure)
  cs = Gs(dists)
  diag_indices = np.arange(cs.shape[0])
  cs[diag_indices,:,diag_indices] = 0 # remove the contribution of each minutia to its own cells

  # local_structures : n x c (cell values for each local structure)
  local_structures = Psi(np.sum(cs, -1))
  return local_structures,ref_cell_coords

"""### Step 8: Fingerprint comparison"""

def comparison(image,valid_minutiae,local_structures,test_image,ref_cell_coords):
  # print(f"""Fingerprint image: {image.shape[1]}x{image.shape[0]} pixels
  # Minutiae: {len(valid_minutiae)}
  # Local structures: {local_structures.shape}""")
  
  f1, m1, ls1 = fingerprint, valid_minutiae, local_structures
  ofn = test_image # Fingerprint of the same finger
  #ofn = 'samples/sample_2' # Fingerprint of a different finger
  f2, (m2, ls2) = cv.imread(f'{ofn}.BMP', cv.IMREAD_GRAYSCALE), np.load(f'{ofn}.npz', allow_pickle=True).values()

  # Compute all pairwise normalized Euclidean distances between local structures in v1 and v2
  # ls1                       : n1 x  c
  # ls1[:,np.newaxis,:]       : n1 x  1 x c
  # ls2                       : (1 x) n2 x c
  # ls1[:,np.newaxis,:] - ls2 : n1 x  n2 x c 
  # dists                     : n1 x  n2
  dists = np.sqrt(np.sum((ls1[:,np.newaxis,:] - ls2)**2, -1))
  dists /= (np.sqrt(np.sum(ls1**2, 1))[:,np.newaxis] + np.sqrt(np.sum(ls2**2, 1))) # Normalize as in eq. (17) of MCC paper
  # Select the num_p pairs with the smallest distances (LSS technique)
  num_p = 5 # For simplicity: a fixed number of pairs
  pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
  score = 1 - np.mean(dists[pairs[0], pairs[1]]) # See eq. (23) in MCC paper
  print(f'Comparison score: {score:.2f}')

  return score
  # @interact(i = (0,len(pairs[0])-1), show_local_structures = False)
  # def show_pairs(i=0, show_local_structures = False):
  #     show(draw_match_pairs(f1, m1, ls1, f2, m2, ls2, ref_cell_coords, pairs, i, show_local_structures))

"""### Main Function"""

def verification(test):
  skipped = 0
  data = dict()


  for fingerprint_img in os.listdir():
    temp = dict()
    fingerprint = cv.imread(fingerprint_img, cv.IMREAD_GRAYSCALE)

    try:
      mask, gx2, gy2, gx, gy = segmentation(fingerprint)
      orientations = local_ridge_orientation(gx2, gy2, gx, gy)
      ridge_period = local_ridge_frequency(fingerprint)
      enhanced = fingerprint_enhancement(ridge_period,fingerprint,orientations,mask)
      all_8_neighborhoods,filtered_minutiae,cn_values, cn = detect_minutiae(enhanced,mask)
      valid_minutiae = minutiae_directions(all_8_neighborhoods,filtered_minutiae,cn_values,cn)
      local_structures,ref_cell_coords = local_structures_calc(valid_minutiae)
      temp['valid_minutiae'] = valid_minutiae
      temp['local_structures'] = local_structures
      temp['ref_cell_coords'] = ref_cell_coords
      data[fingerprint_img] = temp
      temp = dict()
    except:
      skipped += 1

  match_score = dict()
  for i in data:
    image = cv.imread(i,cv.IMREAD_GRAYSCALE)
    match_score[i] = comparison(image,data[i]['valid_minutiae'],data[i]['local_structures'],test_image,data[i]['ref_cell_coords'])
  return max(match_score)

skipped = 0
data = dict()


for fingerprint_img in os.listdir():
  temp = dict()
  fingerprint = cv.imread(fingerprint_img, cv.IMREAD_GRAYSCALE)

  try:
    mask, gx2, gy2, gx, gy = segmentation(fingerprint)
    orientations = local_ridge_orientation(gx2, gy2, gx, gy)
    ridge_period = local_ridge_frequency(fingerprint)
    enhanced = fingerprint_enhancement(ridge_period,fingerprint,orientations,mask)
    all_8_neighborhoods,filtered_minutiae,cn_values, cn = detect_minutiae(enhanced,mask)
    valid_minutiae = minutiae_directions(all_8_neighborhoods,filtered_minutiae,cn_values,cn)
    local_structures,ref_cell_coords = local_structures_calc(valid_minutiae)
    temp['valid_minutiae'] = valid_minutiae
    temp['local_structures'] = local_structures
    temp['ref_cell_coords'] = ref_cell_coords
    data[fingerprint_img] = temp
    temp = dict()
  except:
    skipped += 1

"""### Finding a match for a sample fingerprint"""

# test_image = "/content/samples/sample_2"
# score = verification(test_image)

# test_image = "/content/samples/sample_2"
# match_score = dict()
# for i in data:
#   image = cv.imread(i,cv.IMREAD_GRAYSCALE)
#   match_score[i] = comparison(image,data[i]['valid_minutiae'],data[i]['local_structures'],test_image,data[i]['ref_cell_coords'])

"""### Plotting the sample and its match"""

# plt.figure(figsize=(10,10))

# plt.subplot(1,2,1)
# plt.title(f"Original Fingerprint")
# plt.imshow(plt.imread(f"{test_image}.png"))
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.title(f"Matched Fingerprint with score of {match_score[max(match_score)]}")
# plt.imshow(plt.imread(f"/content/socofing/SOCOFing/Real/{max(match_score)}"))
# plt.axis("off")
