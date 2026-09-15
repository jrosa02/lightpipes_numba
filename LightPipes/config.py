_USE_PYFFTW = False #Change to True for always using pyFFTW

# Allow LLVM to reassociate floating-point operations in the numba kernels.
# OFF by default: results then track upstream LightPipes as closely as
# possible, so any numerical difference points at a real bug rather than at
# floating-point reordering. Set to True (before importing LightPipes) to
# trade a little accuracy for speed.
_FASTMATH = False
