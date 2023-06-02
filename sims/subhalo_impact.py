import initial_stream
import subhalo_orbit
import stream_impact
import numpy as np

import multiprocessing
from multiprocessing import Pool

def chi_eval(r,phi,vphi,vz,M_sat,tmax,t_a,phi_a,rs_sat,pid):
    #==========================================================================[    0     ,     1      ,      2      ,     3     ,   4    ,   5    ,     6     ,      7      ,     8      , 9 ]
    #==========================================================================[impact_r_D,impact_phi_D,impact_vphi_D,impact_vz_D,masses_D,tmaxes_D,tapproach_D,phiapproach_D,scaleradii_D,pid]
    
    SH_x, SH_y, SH_z, SH_vx, SH_vy, SH_vz, dunno = initial_stream.chi2_eval(-0.38297458,   -0.87059476, -109.48359169,   21.8659734 , 0.70106313,15,t_a,tmax,int(pid))
    sat_x, sat_y, sat_z, sat_vx, sat_vy, sat_vz = subhalo_orbit.chi2_eval(SH_x, SH_y, SH_z, SH_vx, SH_vy, SH_vz,r,phi,vphi,vz,tmax,t_a,phi_a,int(pid))
    chi = stream_impact.chi2_eval(-0.38297458,   -0.87059476, -109.48359169,   21.8659734 , 0.70106313,15, sat_x, sat_y, sat_z, sat_vx, sat_vy, sat_vz, tmax,M_sat,rs_sat,int(pid))
    return chi


r = 0.1
phi = 250
vphi = 35 
vz = -10 
M_sat = 0.001
t_a = 0.25
phi_a = -4
rs_sat = 0.3
pid=0

tmax=4

print('EVALUATED LIKELIHOOD: ', chi_eval(r,phi,vphi,vz,M_sat,tmax,t_a,phi_a,rs_sat,pid))



