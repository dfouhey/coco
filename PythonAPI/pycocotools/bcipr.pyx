# distutils: language = c

#**************************************************************************
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# BCI Code written by David Fouhey, 2017
# Licensed under the Simplified BSD License [see coco/license.txt]
#**************************************************************************

__author__ = 'dfouhey'

import sys
PYTHON_VERSION = sys.version_info[0]

# import both Python-level and C-level symbols of Numpy
# the API uses Numpy to interface C and Python
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t

# intialized Numpy. must do.
np.import_array()

# import numpy C function
# we use PyArray_ENABLEFLAGS to make Numpy ndarray responsible to memory management
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
def bcipr(  np.ndarray[DTYPE_t,ndim=2] bciCounts, \
            np.ndarray[long,ndim=1] imgIds, np.ndarray[long,ndim=1] npigImgIds, 
            np.ndarray[DTYPE_t,ndim=2] tps,np.ndarray[DTYPE_t,ndim=2] fps, 
            np.ndarray[DTYPE_t,ndim=1] recThrs):
    #this is rewritten using as many of the original variable names to make 
    #this easier to follow

    #sizes of the various arrays; they'll get called per-replicate otherwise
    cdef int TPS0 = tps.shape[0]
    cdef int TPS1 = tps.shape[1]
    cdef int bciCount = bciCounts.shape[0]
    cdef int recCount = recThrs.shape[0]
    cdef int npigImgIdsCount = len(npigImgIds)

    cdef int R = len(recThrs)
    cdef int i, ri, pi, t, sub, bci
    cdef DTYPE_t npigBCI, tp, fp, tpsum, fpsum

    #note: switching this to malloc and doubles doesn't makes it way less readable
    cdef np.ndarray[DTYPE_t,ndim=1] q = np.zeros((R,))
    cdef np.ndarray[DTYPE_t,ndim=1] rc = np.zeros((TPS1,))
    cdef np.ndarray[DTYPE_t,ndim=1] pr = np.zeros((TPS1,))
    cdef np.ndarray[long,ndim=1] inds  

    cdef DTYPE_t spacing = np.spacing(1)
   
    #return nans, since this 
    cdef np.ndarray[DTYPE_t,ndim=2] APBCI = np.empty((TPS0,bciCounts.shape[0]))
    cdef np.ndarray[DTYPE_t,ndim=2] RBCI = np.empty((TPS0,bciCounts.shape[0]))

    APBCI.fill(np.nan)
    RBCI.fill(np.nan)


    for bci in range(bciCount):

        #compute the denominator of the recall for this replicate
        npigBCI = 0
        for i in range(npigImgIdsCount):
            npigBCI = npigBCI+bciCounts[bci,npigImgIds[i]]

        #sometimes the entire set of images is valid but the replicate is not
        #AP = nan flags this
        if npigBCI == 0: continue

        #for each threshold
        for t in range(TPS0):

            #everything rolled into one:
            #   produce the tp/fp cumulative sum for the replicate
            #   then compute p/r
            tpsum = 0
            fpsum = 0
            for i in range(TPS1):
                tpsum = tpsum + tps[t,i]*bciCounts[bci,imgIds[i]]
                fpsum = fpsum + fps[t,i]*bciCounts[bci,imgIds[i]]
                rc[i] = tpsum / npigBCI
                pr[i] = tpsum / (fpsum+tpsum+spacing)

            RBCI[t,bci] = rc[TPS1-1]

            #q isn't auto-cleared 
            for i in range(R): q[i] = 0

            #smooth the curves; count so cython gets range with a single arg
            for sub in range(TPS1-1):
                i = TPS1-1-sub
                if pr[i] > pr[i-1]:
                    pr[i-1] = pr[i]
            
            inds = np.searchsorted(rc, recThrs, side='left')

            for ri in range(recCount):
                pi = inds[ri]
                if pi < TPS1:
                    q[ri] = pr[pi]
           
            #AP = mean over the results at the recalls
            APBCI[t,bci] = 0 
            for i in range(R):
                APBCI[t,bci] = APBCI[t,bci] + q[i] 
            APBCI[t,bci] = APBCI[t,bci] / R

    return APBCI, RBCI



