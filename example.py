import numpy as np
import matplotlib.pyplot as plt
import ADM


#Physical properties of the star
#-------------------------------------------------------------------------------
#Effective temperature, Teff, is in Kelvins
#Stellar mass, Mstar, is in solar mass
#Stellar radius, Rstar, is in solar radius
#Terminal velocity, Vinf, is in km/s
#Mass-loss rate, Mdot, is in solar mass per year
#Polar field strength, Bstar, is in Gauss
Teff = 35000.0 
Mstar = 30.0
Rstar = 15.
Vinf = 2500.0
Mdot = 10**(-6.0)
Bstar = 2500.0


#Geometric angles
#-------------------------------------------------------------------------------
#Inclination angle, inc, in degrees
#Magnetic obliquity, beta, in degrees
inc = 30.
beta = 60.
A = inc+beta
B = np.abs(inc-beta)


#Extra parameters
#-------------------------------------------------------------------------------
#Smoothing length, delta
#Macroturbulent velocity, vmac, in km/s
#FWHM of the line, FWHM_lda, in Angstroms
#Clumping factor, fcl
#Rest wavelength, lda0, in Angstroms
delta = 0.1 
vmac = 100.0
FWHM_lda = 0.0
fcl = 5.0
lda0 = 6562.8


#Calling ADM
#-------------------------------------------------------------------------------
phi = np.linspace(0.,1.,25) #rotational phase
lda = np.linspace(lda0-10,lda0+10,25) #wavelength grid
phot=np.ones(len(lda)) #photospheric line profile (ignored here)
Nx = Ny = Nz = 50 #grid size 
out = ADM.Halpha(phi, A, B, Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf, Mdot, Bstar, delta, fcl, lda0*10**(-8), lda*10**(-8), phot, vmac*100000., FWHM_lda**10**(-8))


#Plotting phased Halpha EW curves and line profiles
#-------------------------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.plot(phi,out[0],'k')
plt.plot(phi+1,out[0],'k')
plt.plot(phi-1,out[0],'k')
plt.xlabel('Rotational phase',fontsize=14)
plt.ylabel(r'Equivalent width [$\AA$]',fontsize=14)
plt.xlim([-0.5,1.5])
plt.show()

plt.figure(figsize=(6,6))
for i in range(len(phi)):
	plt.plot(lda,out[1][i,:])
plt.xlabel('Wavelength [$\AA$]',fontsize=14)
plt.ylabel('Normalized flux',fontsize=14)
plt.xlim([lda0-10,lda0+10])
plt.show()

