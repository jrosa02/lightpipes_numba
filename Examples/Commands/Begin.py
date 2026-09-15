from OptimLightPipes import *

GridSize = 30*mm
GridDimension = 5
lambda_ = 500*nm #lambda_ is used because lambda is a Python build-in function.

Field = Begin(GridSize, lambda_, GridDimension)
print("OptimLightPipes version = ", LPversion)
if LPversion < "2.0.0":
    print(Field)
else:
    print(Field.field)
LPtest()
LPdemo()

