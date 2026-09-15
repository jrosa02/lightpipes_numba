# -*- coding: utf-8 -*-



import numpy as _np
from numpy import pi as _pi

from ._numba_compat import njit

@njit(inline='always')
def UNWRAP(ol, co, factor, hfactor):
    """Shift `ol` by whole turns until it is within half a turn of `co`.

    `val` must stay in the dtype of `ol` rather than being unified to float64:
    for a float32 phase upstream rounds at every subtraction, and after ~12
    shells the accumulated difference reaches a full 2*pi. The caller passes
    `factor`/`hfactor` already cast to the buffer dtype, which keeps every
    intermediate in that dtype and reproduces upstream bit-for-bit.
    """
    val = ol
    if val-co >= hfactor:
        while val-co >= hfactor:
            val = val - factor
    elif val-co <= -hfactor:
        while val-co <= -hfactor:
            val = val + factor
    ne = val
    return ne


def unwrap_phase(Phi):
    """
    Attempt to unwrap the phase of the input. If data is noisy/unsmooth,
    this may result in unphysical patterns. In this case, try a finer sampling.

    Parameters
    ----------
    Phi : ndarray (MxN), real
        Real phase of size M (in y) and N (in x).

    Returns
    -------
    ndarray of same shape as input, with phase values unwrapped

    """
    ysize, xsize = Phi.shape
    # The kernel walks the grid in expanding square shells; each shell depends
    # on the previous one and each pixel on its predecessor within the shell,
    # so this is compiled serially (njit) rather than parallelized.
    # Keep the caller's float dtype: upstream returns float32 for complex64
    # fields, and promoting would change the accumulated averages.
    return _unwrap_phase_kernel(_np.ascontiguousarray(Phi), ysize, xsize)


@njit
def _unwrap_phase_kernel(Phi, ysize, xsize):
    #Checked functionality, gives similar results to Cpp code -> OK
    ibuffer = Phi.flatten()

    obuffer = _np.zeros_like(ibuffer)
    p = 0 #pointer/array index in flat array inbuffer, point to first el
    q = 0 #pointer to outbuffer
    hxsize = xsize >> 1 #x/2
    hysize = ysize >> 1
    # Every constant is cast to the buffer dtype. numpy keeps (f32+f32)/2 in
    # float32, but numba would promote the int literal and unify the result to
    # float64; the averages and the wrap arithmetic would then round
    # differently from upstream, diverging by a full 2*pi after ~12 shells.
    _dt = ibuffer.dtype.type
    hfactor = _dt(_pi)
    factor = _dt(2*_pi)
    _two = _dt(2.0)
    _three = _dt(3.0)
    
    """
    /* position p in centre of image */
    /* central pixel */
    """
    p += hysize*xsize + hxsize
    q += hysize*xsize + hxsize
    obuffer[q] = ibuffer[p] #central pixel
    """
    /* first shell of pixels */
    /* north */
    """
    comval = obuffer[q]
    oldpix = ibuffer[p-xsize]
    obuffer[q-xsize] = UNWRAP(oldpix, comval, factor, hfactor)
    """ /* east */
    """
    comval = (obuffer[q-xsize]
            + obuffer[q])/_two
    oldpix = ibuffer[p+1]
    obuffer[q+1] = UNWRAP(oldpix, comval, factor, hfactor)
    """ /* south */
    """
    comval = (obuffer[q+1]
            + obuffer[q])/_two
    oldpix = ibuffer[p+xsize]
    obuffer[q+xsize] = UNWRAP(oldpix, comval, factor, hfactor)
    """ /* west */
    """
    comval = (obuffer[q+xsize]+obuffer[q])/_two
    oldpix = ibuffer[p-1]
    obuffer[q-1] = UNWRAP(oldpix, comval, factor, hfactor)
    
    """ /* north-west */
    """
    comval = (obuffer[q-xsize]+obuffer[q-1]+obuffer[q])/_three
    oldpix = ibuffer[p-xsize-1]
    obuffer[q-xsize-1] = UNWRAP(oldpix, comval, factor, hfactor)
    """ /* north-east */
    """
    comval = (obuffer[q-xsize]+obuffer[q+1]+obuffer[q])/_three
    oldpix = ibuffer[p-xsize+1]
    obuffer[q-xsize+1] = UNWRAP(oldpix, comval, factor, hfactor)
    """ /* south-east */
    """
    comval = (obuffer[q+xsize]+obuffer[q+1]+obuffer[q])/_three
    oldpix = ibuffer[p+xsize+1]
    obuffer[q+xsize+1] = UNWRAP(oldpix, comval, factor, hfactor)
    """ /* south-west */
    """
    comval = (obuffer[q+xsize]+obuffer[q-1]+obuffer[q])/_three
    oldpix = ibuffer[p+xsize-1]
    obuffer[q+xsize-1] = UNWRAP(oldpix, comval, factor, hfactor)
    
    """
    /************ next shells ***********/
    """
    i = 1
    j = 0
    i += 1
    while i < hxsize:
        """/* north */
        """
        comval = (obuffer[q - (i-1)*xsize - i + 1]
                + obuffer[q - (i-1)*xsize - i + 2])/_two
        oldpix = ibuffer[p - i*xsize - i + 1]
        obuffer[q - i*xsize - i + 1] = UNWRAP(oldpix, comval, factor, hfactor)
        
        j = i-1
        j -= 1
        while j > -i:
            comval = (obuffer[q - i*xsize - j - 1]
                    + obuffer[q - (i-1)*xsize - j]
                    + obuffer[q - (i-1)*xsize - j - 1])/_three
            oldpix = ibuffer[p - i*xsize - j]
            obuffer[q - i*xsize - j] = UNWRAP(oldpix, comval, factor, hfactor)
            j -= 1
        
        """/* south */
        """
        comval = (obuffer[q + (i-1)*xsize + i - 1]
                + obuffer[q + (i-1)*xsize + i - 2])/_two
        oldpix = ibuffer[p + i*xsize + i - 1]
        obuffer[q + i*xsize + i - 1] = UNWRAP(oldpix, comval, factor, hfactor)
        
        j = i-1
        j -= 1
        while j > -i:
            comval = (obuffer[q + i*xsize + j + 1]
                    + obuffer[q + (i-1)*xsize + j]
                    + obuffer[q + (i-1)*xsize + j + 1])/_three
            oldpix = ibuffer[p + i*xsize + j]
            obuffer[q + i*xsize + j] = UNWRAP(oldpix, comval, factor, hfactor)
            j -= 1
        
        """/* east */
        """
        comval = (obuffer[q + i - 1 - (i-1)*xsize]
                + obuffer[q + i - 1 - (i-2)*xsize])/_two
        oldpix = ibuffer[p + i - (i - 1)*xsize]
        obuffer[q + i - (i - 1)*xsize] = UNWRAP(
                oldpix, comval, factor, hfactor)
        
        j = i-1
        j -= 1
        while j > -i:
            comval = (obuffer[q + i - (j+1)*xsize]
                    + obuffer[q + i - 1 - (j+1)*xsize]
                    + obuffer[q + i - 1 - j*xsize])/_three
            oldpix = ibuffer[p + i - j*xsize]
            obuffer[q + i - j*xsize] = UNWRAP(oldpix, comval, factor, hfactor)
            j -= 1
        
        """/* west */
        """
        comval = (obuffer[q - i + 1 + (i-1)*xsize]
                + obuffer[q - i + 1 + (i-2)*xsize])/_two
        oldpix = ibuffer[p - i + (i - 1)*xsize]
        obuffer[q - i + (i - 1)*xsize] = UNWRAP(
                oldpix, comval, factor, hfactor)
        
        j = i-1
        j -= 1
        while j > -i:
            comval = (obuffer[q - i + (j+1)*xsize]
                    + obuffer[q - i + 1 + (j+1)*xsize]
                    + obuffer[q - i + 1 + j*xsize])/_three
            oldpix = ibuffer[p - i + j*xsize]
            obuffer[q - i + j*xsize] = UNWRAP(oldpix, comval, factor, hfactor)
            j -= 1
        
        """/* north-west */
        """
        comval = (obuffer[q - (i-1)*xsize - i]
                + obuffer[q - i*xsize - i + 1]
                + obuffer[q - (i-1)*xsize - i + 1])/_three
        oldpix = ibuffer[p - i*xsize - i]
        obuffer[q - i*xsize - i] = UNWRAP(oldpix, comval, factor, hfactor)
        """ /* north-east */
        """
        comval = (obuffer[q - (i-1)*xsize + i]
                + obuffer[q - i*xsize + i - 1]
                + obuffer[q - (i-1)*xsize + i - 1])/_three
        oldpix = ibuffer[p - i*xsize + i]
        obuffer[q - i*xsize + i] = UNWRAP(oldpix, comval, factor, hfactor)
        """ /* south-east */
        """
        comval = (obuffer[q + (i-1)*xsize + i]
                + obuffer[q + i*xsize + i - 1]
                + obuffer[q + (i-1)*xsize + i - 1])/_three
        oldpix = ibuffer[p + i*xsize + i]
        obuffer[q + i*xsize + i] = UNWRAP(oldpix, comval, factor, hfactor)
        """ /* south-west */
        """
        comval = (obuffer[q + (i-1)*xsize - i]
                + obuffer[q + i*xsize - i + 1]
                + obuffer[q + (i-1)*xsize - i + 1])/_three
        oldpix = ibuffer[p + i*xsize - i]
        obuffer[q + i*xsize - i] = UNWRAP(oldpix, comval, factor, hfactor)
        
        i += 1 #end of while
        
    """/* upper line and left column */
    """
    """
	/* upper line*/
    """
    comval = (obuffer[q - (hxsize-1)*xsize - hxsize + 1]
            + obuffer[q - (hxsize-1)*xsize - hxsize + 2])/_two
    oldpix = ibuffer[p - hxsize*xsize - hxsize + 1]
    obuffer[q - hxsize*xsize - hxsize + 1] = UNWRAP(
            oldpix, comval, factor, hfactor)
    
    j = hxsize-1
    j -= 1
    while j > -hxsize:
        comval = (obuffer[q - hxsize*xsize - j - 1]
                + obuffer[q - (hxsize-1)*xsize - j]
                + obuffer[q - (hxsize-1)*xsize - j - 1])/_three
        oldpix = ibuffer[p - hxsize*xsize - j]
        obuffer[q - hxsize*xsize - j] = UNWRAP(oldpix, comval, factor, hfactor)
        j -= 1
    """
	/* left line */
    """
    comval = (obuffer[q - hxsize + 1 + (hxsize-1)*xsize]
            + obuffer[q - hxsize + 1 + (hxsize-2)*xsize])/_two
    oldpix = ibuffer[p - hxsize + (hxsize - 1)*xsize]
    obuffer[q - hxsize + (hxsize - 1)*xsize] = UNWRAP(
            oldpix, comval, factor, hfactor)
    
    j = hxsize-1
    j -= 1
    while j > -hxsize:
        comval = (obuffer[q - hxsize + (j+1)*xsize]
                + obuffer[q - hxsize + 1 + (j+1)*xsize]
                + obuffer[q - hxsize + 1 + j*xsize])/_three
        oldpix = ibuffer[p - hxsize + j*xsize]
        obuffer[q - hxsize + j*xsize] = UNWRAP(oldpix, comval, factor, hfactor)
        j -= 1
    
    """ /* upper left corner */
    """
    comval = (obuffer[q - (hxsize-1)*xsize - hxsize]
            + obuffer[q - hxsize*xsize - hxsize + 1]
            + obuffer[q - (hxsize-1)*xsize - hxsize + 1])/_three
    oldpix = ibuffer[p - hxsize*xsize - hxsize]
    obuffer[q - hxsize*xsize - hxsize] = UNWRAP(
            oldpix, comval, factor, hfactor)
    
    return obuffer.reshape((ysize,xsize))



