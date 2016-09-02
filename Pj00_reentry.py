"""
Pj00_reentry.py

Reentry vehicle tracking

Filters under investigation:
1. Continuous-Discrete Extended Kalman Filter
   1.1 Discretized Linearization method
   1.2 Linearized Discretization method
2. Continuous-Discrete Unscented Kalman Filter
   2.1 Discretized Linearization method
   2.2 Linearized Discretization method
3. Continuous-Discrete Cubature Kalman Filter

Author: Ashiv Dhondea, RRSG, UCT
Date: 01/09/16

Based on:
1. @article{julier2004unscented,
  title={Unscented filtering and nonlinear estimation},
  author={Julier, Simon J and Uhlmann, Jeffrey K},
  journal={Proceedings of the IEEE},
  volume={92},
  number={3},
  pages={401--422},
  year={2004},
  publisher={IEEE}
}

2. @article{sarkka2007unscented,
  title={On unscented Kalman filtering for state estimation of continuous-time nonlinear systems},
  author={Sarkka, Simo},
  journal={IEEE Transactions on automatic control},
  volume={52},
  number={9},
  pages={1631--1641},
  year={2007},
  publisher={IEEE}
}

Copyright 2016 AshivD <ashivdhondea5@gmail.com>
 """
## Include necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

## Ashiv's own stuff
import Num_Integ as Ni
import ReentryDynamics as Rd

## Define time variable
dt = 0.1; # in [s]
t_end = 200;

T = np.arange(0,t_end,dt,dtype=np.float64);

## Define the constants from ref [1]
b0 = -0.59783;
H0 = 13.406;
Gm0 = 3.9860e5;
R0 = 6374;

Qk = np.diag([2.4064e-5,2.4064e-5,1e-6]); 
Qc = Qk/dt;

# L : dispersion matrix
L = np.array([[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64);

true_P0 = np.diag([1e-6,1e-6,1,1e-6,1e-6,0.0]);
true_Qc = Qc;
true_Qc[2,2] = 1e-6;
true_m0 = np.array([6500.4,349.14,-1.8093,-6.7967,0.6932],dtype=np.float64);

m0 = np.array([6500.4,349.14,-1.8093,-6.7967,0.0],dtype=np.float64);
P0 = true_P0; P0[4] = 1.0;

xradar = np.array([R0,0],dtype=np.float64);
## Define state vector
# x[0],x[1] -> x & y position
# x[2],x[3] -> x & y velocity
# x[4] -> parameter of the vehicle's aerodynamic properties
    
x_state = Ni.fnEuler_Maruyama(true_m0,Rd.fnReentry,T,L,true_Qc*dt);
print 'Truth data generated.'
x_state_strong = Ni.fnSRK_Crouse(true_m0,Rd.fnReentry,T,L,true_Qc);

fig = plt.figure(1)
fig.suptitle('Reentry tracking problem')
plt.plot(x_state[0,:],x_state[1,:],'g',label = 'EM method');
plt.plot(x_state_strong[0,:],x_state_strong[1,:],'m',label='IT-1.5 method');
aa = 0.02*np.arange(-1,4,0.1,dtype=np.float64);
cx = R0*np.cos(aa);
cy = R0*np.sin(aa);
plt.plot(xradar[0],xradar[1],'k',marker='o',label='Radar');
plt.plot(cx,cy,'r',label='Earth');
plt.legend(loc='best');
plt.axis([6340,6520,-200,600])
plt.ylabel('y [km]')
plt.xlabel('x [km]')
ax = plt.gca()
ax.grid(True)
plt.show()
fig.savefig('Pj00_reentry.png',bbox_inches='tight',pad_inches=0.01);

