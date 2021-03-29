from numba import jit
import numpy as np
import warnings
import cv2

# Ignore Numba warnings
warnings.filterwarnings('ignore')

def NStepPhaseShifting(imgs, N):
    delta = 2*np.pi*np.arange(1,N+1)/N

    sumIsin, sumIcos = 0., 0.
    for fname, deltak in zip(imgs, delta):
        I = cv2.imread(fname, 0)
        sumIsin += I*np.sin(deltak)
        sumIcos += I*np.cos(deltak)
    
    return -np.arctan2(sumIsin, sumIcos)

def seedPoint(fn_cl, mask, verticalLine=True):
    cl = cv2.imread(fn_cl, 0)
    cl = np.uint8(cl*mask)
    
    # Binarize image
    _, bw = cv2.threshold(cl,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Coordinates of the points on the line
    y, x = np.where(bw)
    
    if verticalLine:
        # Estimate seed point on the vertical line
        v = np.abs(y-y.mean())
        ind = np.where(v == v.min())[0]
    else:
        # Estimate seed point on the horizontal line
        v = np.abs(x-x.mean())
        ind = np.where(v == v.min())[0]
    
    return np.array([x[ind].mean(), y[ind].mean()], np.float32)

@jit('double[:,:](double[:,:], float32[:], int32[:,:])', cache=True)
def spatialUnwrap(phased, p0, mask):
    # Convert initial point of unwrapping to int32
    p0 = np.round(p0).astype(np.int32)
    
    # Original discontinuous phase shape
    h, w = phased.shape
    
    # Zero-pad array edges and update p0 to this change
    mask = np.pad(mask, ((1,1) , (1,1)), 'constant')
    phased = np.pad(phased, ((1,1) , (1,1)), 'constant')
    p0 += 1

    # Initialize (unwrapped) continuous phase with the wrapped
    phasec = phased.copy()
    
    # Initialize array with the position of unwrapped points
    XY = np.zeros([np.sum(mask),2], np.int32)
    XY[0] = p0 # The first point is p0
    # Remove p0 from the mask
    mask[p0[1],p0[0]] = 0

    # Estimate final continuous phase
    phasec = _8neighbors_unwrap(phased, phasec, mask, p0, XY)
    
    return phasec[1:h+1,1:w+1] # Array without zero-padded edges

@jit('double[:,:](double[:,:], double[:,:], int32[:,:], int32[:], int32[:,:])',
     nopython=True, cache=True)
def _8neighbors_unwrap(phased, phasec, mask, p0, XY):
    xo = [-1,-1,-1,0,0,1,1,1] # x offset
    yo = [-1,0,1,-1,1,-1,0,1] # y offset

    cont, cont1, opct = 0, 0, 1
    while opct:
        # Continuous and discontinuous phase values in p0
        PCI = phasec[p0[1],p0[0]]
        PDI = phased[p0[1],p0[0]]
        
        # Unwrap the 8-neighbors of p0
        for i in range(8):
            # Move p0 with xo and yo
            px = xo[i] + p0[0]
            py = yo[i] + p0[1]
            
            # Check if point is within the mask
            if mask[py,px]:
                # Wrapped phase at the point neighboring to p0
                PDC = phased[py,px]
                D = (PDC-PDI)/(2*np.pi)
                
                # Unwrapp (px, py)
                phasec[py,px] = PCI+2*np.pi*(D-round(D))

                # Save unwrapped point (px, py)
                XY[cont1+1,0] = px
                XY[cont1+1,1] = py
                cont1 += 1 # Count a new point
                
                # Remove (px, py) from the mask
                mask[py,px] = 0
        
        # Count a while loop completed
        cont += 1
        # Check stop criteria
        if cont > cont1:
            opct = 0
        else:
            # Update p0
            p0 = XY[cont,:]
    
    return phasec