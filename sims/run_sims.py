# from subhalo_impact import chi_eval
from multiprocessing import Process
from multiprocessing import Pool
import itertools

import initial_stream
import subhalo_orbit
import stream_impact
import numpy as np
import os

# import sys

def run_sims(nsims=1000):
    pool = Pool()
    logM_array = np.random.uniform(-5,1,nsims)
    vz_array = np.random.uniform(-50,0,nsims)
    pool.map(run_sim, zip(logM_array, vz_array))        

def run_sim(params):
    logM_sat, vz = params
    M_sat = 10**logM_sat
    # M_sat = 10**float(sys.argv[1])
    # vz = 10**float(sys.argv[2])
    rs_sat = 1.05 * (M_sat*10*10)**0.5
    
    pid = calculate_pid(logM_sat, vz) 
    
    r = 0.2 # impact parameter in kpc (distance from stream to sat)
    phi = 250 # angle around stream in dec
    vphi = 35  # velocity around stream in km/s
    # vz = -10  # velocity along stream in km/s
    # M_sat = 0.001 # mass of subhalo in 1e10 Msun
    t_a = 0.2 # time since interaction in Gyr
    phi_a = -4 # interaction point along stream in deg, phi=0 is progenitor location (try -20 to 10)
    # rs_sat = 0.3 # scale radius of subhalo in kpc, can be adjusted along with M_sat using equation 15 in erkal et al. 2015
    # pid=0 # index included in saved filenames
    tmax=4 # how long stream disrupts in Gyr

    print()
    print()
    print('Running logM = %.3f, vz = %.3f' %(M_sat, vz))
    print()
    print()
    simulate_stream(r,phi,vphi,vz,M_sat,tmax,t_a,phi_a,rs_sat,pid)


def calculate_pid(logM_sat, vz):
    pid = '%i'%(logM_sat*1000) + '%i'%(-1*vz*1000)
    return int(pid)

def calculate_pid_old(log_Msat, vz):
    pid = 0
    log_Msat_round = round(log_Msat, 3)
    vz_round = round(vz, 3)
    if log_Msat_round >= 0:
        pid = log_Msat_round * 1e6
    else: 
        pid = abs(log_Msat_round) * 1e6 + 1e7
    if  vz_round >= 0: 
        pid += vz_round * 10
    else:
        pid += abs(vz_round) * 10 + 1e3
    return int(pid)

def simulate_stream(r, phi, vphi, vz, M_sat, tmax, t_a, phi_a, rs_sat, pid):
    SH_x, SH_y, SH_z, SH_vx, SH_vy, SH_vz, dunno = initial_stream.chi2_eval(-0.38297458,   -0.87059476, -109.48359169,   21.8659734 , 0.70106313,15,t_a,tmax,int(pid))
    sat_x, sat_y, sat_z, sat_vx, sat_vy, sat_vz = subhalo_orbit.chi2_eval(SH_x, SH_y, SH_z, SH_vx, SH_vy, SH_vz,r,phi,vphi,vz,tmax,t_a,phi_a,int(pid))
    chi = stream_impact.chi2_eval(-0.38297458,   -0.87059476, -109.48359169,   21.8659734 , 0.70106313,15, sat_x, sat_y, sat_z, sat_vx, sat_vy, sat_vz, tmax,M_sat,rs_sat,int(pid))
    os.remove('orbits/orbit_%i.txt' %pid)
    os.remove('pre_impact/pre_impact_%i.txt' %pid)
    os.remove('final_coords/final_coords_%i.txt' %pid)

if __name__ == '__main__':
    run_sims(nsims=10000)


