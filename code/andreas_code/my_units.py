
# coding: utf-8

# In[1]:


import math


# In[ ]:


# Notebook for user defined units. All dimensioal quantities are in GeV
# and GeV=1


# In[2]:


Kg = 5.6096*math.pow(10,26)
Gram = math.pow(10,-3)*Kg
Meter = 1/(0.1973*math.pow(10,-15))
CentiMeter = math.pow(10,-2)*Meter
Second = 299792458*Meter
Hz = math.pow(Second, -1)
Hour = 3600*Second
Year = 365*24*Hour
MPlanck = 1.2209*math.pow(10, 19)
GN = math.pow(MPlanck, -2)
mPlanck = MPlanck/math.sqrt(8*math.pi)
kpc = 3261*Year
Mpc = math.pow(10, 3)*kpc
pc = math.pow(10, -3)*kpc
Hubble0 = 67.8*math.pow(10, 3)*Meter/Second/Mpc
zeq = 3250
HubbleEq = Hubble0*math.pow(1+zeq, 3/2) 
aeq = 1/(1 + zeq)
RhoCrit = (3*math.pow(Hubble0,2))/(8*math.pi*GN)
hubble0 = Hubble0/(100*10000*Meter/Second/Mpc)
RhoDMU = 0.23*3/(8*math.pi)*math.pow(Hubble0, 2)*math.pow(MPlanck, 2)
RhoDMG = 0.4/math.pow(CentiMeter, 3)
v0DM = 235*1000*Meter/Second
SigmavDM = v0DM/math.sqrt(2)
MSolar = 1.98*math.pow(10, 30)*Kg


# In[10]:


arcmin = (2*math.pi)/360/60
arcsec = (2*math.pi)/360/3600
mas = math.pow(10, -3)*arcsec; 
muas = math.pow(10, -6)*arcsec;
masy = math.pow(10, -3)*arcsec/Year
muasy = math.pow(10, -6)*arcsec/Year
muasyy = math.pow(10, -6)*arcsec/math.pow(Year, 2)
degree = math.radians(1.0)


# In[12]:


rsMW = 18*kpc
rhosMW = 3*math.pow(10,-3)*MSolar*math.pow(pc,-3)
Rsun = 8.29*kpc
a0 = 192.85948*degree
d0 = 27.12825*degree
l0 = 122.93192*degree
vSun = 238*math.pow(10,3)*Meter/Second

