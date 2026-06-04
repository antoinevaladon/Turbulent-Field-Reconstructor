# turbulent-fields-PINN
PINN to reconstruct turbulent plasma fields with sparse data

## Problem Statement:
While large turbulent plasma fields in space contain critical measurements, it's vastness combined with expensive spacecraft, make it very difficult to measure large areas directly. 
Largely, measurements are made at only very few points, with the upcoming 2028 NASA helioswam mission sending 9 spacecraft for continued measurements. In the large field, this 
creates a sparse dataset to determine the greater field. While many innovative methods an PINNs have been able to produce great reconstructions of these fields, a turbulent field 
introduces exrta complexity that assumed linearity does not apply to. The goal of this PINN is to accurately and efficiently recreate the plasma field using PDE constraints
