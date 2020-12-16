#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ADM-based Halpha EW curve synthesis 
# Melissa Munoz
# Updated Dec 2020
#
# See publication Munoz et al. 2020, in prep
# See also Owocki et al. 2006 for more details on the ADM formalism
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#-------------------------------------------------------------------------------
# Library import ---------------------------------------------------------------
#-------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import newton
from scipy import interpolate
from scipy.ndimage.interpolation import rotate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time 
#from scipy.integrate import simps
from scipy.integrate import trapz
from scipy.integrate import cumtrapz
from scipy.special import voigt_profile

#-------------------------------------------------------------------------------
# Some constants ---------------------------------------------------------------
#-------------------------------------------------------------------------------

G = 6.67408*10**(-8)
eV = 1.602*10**(-12)
c = 2.99792458*10**(10)	 
h= 6.6260755*10**(-27)	 
kb = 1.380658*10**(-16)
eV = 1.6021772*10**(-12)	
me = 9.1093897*10**(-28)
mp = 1.6726*10**(-24)
mH = 1.6733*10**(-24)
e = 4.8032068*10**(-10)
X=0.72
Y=0.28
alphae=(1.+X)/2.
alphap=(X+Y/4.)
sigmat = 6.6524*10**(-25)
sigma0=sigmat*3./(16.*np.pi)
Msol = 1.99*10**(33)
Rsol = 6.96*10**(10)
Lsol = 3.9*10**(33)



#-------------------------------------------------------------------------------
# Line-of-sight angle ----------------------------------------------------------
#-------------------------------------------------------------------------------

def csalpha(phi, beta, inc):
	#Note the degenerancy between the inclination and obliquity
	return np.sin(beta)*np.cos(phi)*np.sin(inc)+np.cos(beta)*np.cos(inc)

def csalpha2(phi, A, B):
	#Degenrancy avoided by reexpressong the above equation with 
	#A = inc + beta,
	#B = |inc - beta|
	return 0.5*(np.cos(B)*(1.+np.cos(phi)) + np.cos(A)*(1.-np.cos(phi)) )



#-------------------------------------------------------------------------------
# Energy levels ----------------------------------------------------------------
#-------------------------------------------------------------------------------

def gn(n):
	return 2.*n**2

def En(n): 
	return -13.6*eV/n**2

g2=gn(2)
f32=0.64108
const1=np.pi*e**2*h**3*f32*g2/(me*c*2*(2.*np.pi*me*kb)**(3./2.))



#-------------------------------------------------------------------------------
# Line profiles ----------------------------------------------------------------
#-------------------------------------------------------------------------------

def phi_L(x,nuN):
	#Lorentzian line profile
	#x=nu-nu0
	return (nuN)/(x**2 + (nuN)**2)/np.pi

def phi_G(x,nuD):
	#Gaussian line profile
	#x=nu-nu0
	return np.exp(-(x/nuD)**2)/(nuD*np.sqrt(np.pi))

def phi_LG(x,nuD,nuN):
	#Voigt line profile
	#x=nu-nu0
	return voigt_profile(x, nuD/np.sqrt(2.), nuN)



#-------------------------------------------------------------------------------
# Opacity ----------------------------------------------------------------------
#-------------------------------------------------------------------------------

def chi32(x,nuD,nuN,Np,Ne,T):
	b2=1. 
	b3=1.
	return const1*Np*Ne*T**(-1.5)*(b2*np.exp(39455.1/T) - b3*np.exp(17535.6/T))*phi_LG(x,nuD,nuN)
	 	 
def Bnu(nu,T):
	return 2.*h*nu**3/c**2/(np.exp(h*nu/(kb*T))-1.)



#-------------------------------------------------------------------------------
# ADM auxiliary equations ------------------------------------------------------
#-------------------------------------------------------------------------------

#Dipole magnetic field
def Bd(r,mu,mustar):
	return (1./r)**3*((1+3*mu**2)/(1+3*mustar**2))**0.5

# Wind upflow
#-------------------------------------------------------------------------------

def w(r):
	return 1.-1./r

def vw(r,vinf):
	return vinf*(1.-1./r)

def rhow(r,mu):
	return 2.*np.sqrt(r - 1. + mu**2)*np.sqrt(1.+3.*mu**2)/((r- 1.)*(4.*r - 3. + 3.*mu**2))*(1./r)**(3./2.)


# Hot post-shock 
#-------------------------------------------------------------------------------

def wh(r,rs,mu,mus,Tinf,Teff):
	ws=w(rs)
	Ts=Tinf*ws**2
	return ws/4.*(Th(rs,mu,mus,Tinf,Teff)/Ts)*(Bd(r,mu,mus))

def vh(r,rs,mu,mus,Tinf,Teff,vinf):
	ws=w(rs)
	Ts=Tinf*ws**2
	return ws*vinf/4.*Th(rs,mu,mus,Tinf,Teff)/Ts*np.sqrt((1.+3.*mu**2)/(1.+3.*mus**2))*(rs/r)**3

def g(mu):
	return np.abs(mu - mu**3 + 3.*mu**5/5. - mu**7/7.)

def TTh(rs,mu,mus,Tinf):
	ws=w(rs)
	Ts=Tinf*ws**2
	return Ts*(g(mu)/g(mus))**(1./3.)

def Th(rs,mu,mus,Tinf,Teff):
	return np.maximum(TTh(rs,mu,mus,Tinf),Teff)

def rhoh(r,rs,mu,mus,Tinf,Teff):
	ws=w(rs)
	Ts=Tinf*ws**2
	return 4.*rhow(rs,mus)*Ts/Th(rs,mu,mus,Tinf,Teff)

# Cooled downflow 
#-------------------------------------------------------------------------------

def wc(r,mu):
	return np.abs(mu)*np.sqrt(1./r)

def vc(r,mu,ve):
	return np.abs(mu)*np.sqrt(1./r)*ve

def rhoc(r,mu,delta):
	return 2.*np.sqrt(r - 1. + mu**2)*np.sqrt(1.+3.*mu**2)/(np.sqrt(mu**2+delta**2/r**2)*(4.*r - 3. + 3.*mu**2))*(1./r)**(2.)

def f(mus,mustar,chiinf):
	rm = 1./(1.-mustar**2)
	rs = rm*(1.-mus**2)
	ws = w(rs)
	ggmus = chiinf/(6.*mustar)*(1.+3.*mustar**2)/(1.+3.*mus**2)*(ws*rs/rm)**4*(rs)**2
	gmus = g(mus)
	return (gmus-ggmus) 



#-------------------------------------------------------------------------------
# ADM --------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def admCAL(Nx, Ny, Nz, RA, Rc, Teff, Tinf, chiinf, delta ):

	#Defining magnetosphere grid size	
	NNx=2*Nx
	NNy=2*Ny
	NNz=2*Nz

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)	
	X=XX[Nx:NNx]
	Y=YY[Ny:NNy]
	Z=ZZ[Nz:NNz]
	
	#Defining density, speed, and temperature grids of each component
	Rhoh=np.zeros([Nz,Nx,Ny])
	Rhow=np.zeros([Nz,Nx,Ny])
	Rhoc=np.zeros([Nz,Nx,Ny])
	Vh=np.zeros([Nz,Nx,Ny])
	Vw=np.zeros([Nz,Nx,Ny])
	Vc=np.zeros([Nz,Nx,Ny])
	tw = np.zeros([Nz,Nx,Ny])
	tc = np.zeros([Nz,Nx,Ny])
	th = np.zeros([Nz,Nx,Ny])

	#Last closed loop
	mustar_RA = np.sqrt(1.-1./RA)
	mustars_RA = np.linspace(0.01,mustar_RA,Nz)
	mus_RA = np.zeros(Nz)
	rs_RA = np.zeros(Nz)
	r_RA = (1.-mustars_RA**2)/(1-mustar_RA**2)
	for i in range(Nz):
		try:
			tempmus = newton(f, 0.3, args=(mustars_RA[i],chiinf,))
		except RuntimeError:	
			tempmus=0.
			#print 'error LC'
		mus_RA[i]=np.abs(tempmus)
		rs_RA[i]=(1.-mus_RA[i]**2)/(1.-mustars_RA[i]**2)
	fs=interp1d( mustars_RA,mus_RA,bounds_error=False, fill_value=0. )

	#Compute ADM in first octant of the magnetosphere
	for i in range(0,Nx):
		for j in range(0,Ny):
			p=np.sqrt(X[i]**2+Y[j]**2)
			for k in range(0,Nz):
				r=np.sqrt(p**2+Z[k]**2)
				mu=Z[k]/r
				rRA=(1.-mu**2)/(1-mustar_RA**2)
				if r > 1.05:
					mustar=np.sqrt(1.-(1.-mu**2)/r)
					rm = 1./(1.-mustar**2)
					mus=fs(mustar)
					'''
					try:
						tempmus = newton(f, 0.3, args=(mustar,chiinf,))
					except RuntimeError:	
						tempmus=0.
						#print 'error LC'
					mus = np.abs(tempmus)
					'''
					rs = rm*(1.-mus**2)
					Rhow[k,i,j]=rhow(r,mu)
					Vw[k,i,j]=w(r)
					tw[k,i,j]=Teff		
					if r < rRA:
						Rhoc[k,i,j]=rhoc(r,mu,delta)
						Vc[k,i,j]=wc(r,mu)
						tc[k,i,j]=Teff
						if r > rs and rs > 1.05 :
							Rhoh[k,i,j]=rhoh(r,rs,mu,mus,Tinf,Teff)
							Vh[k,i,j]=wh(r,rs,mu,mus,Tinf,Teff)
							th[k,i,j]=Th(rs,mu,mus,Tinf,Teff)
	
	#Transposing density in remaining octants (axial symmetry)
	Rhoh=np.concatenate([Rhoh[::-1,:,:],Rhoh],axis=0)
	Rhoh=np.concatenate([Rhoh[:,::-1,:],Rhoh],axis=1)
	Rhoh=np.concatenate([Rhoh[:,:,::-1],Rhoh],axis=2)
	Rhoc=np.concatenate([Rhoc[::-1,:,:],Rhoc],axis=0)
	Rhoc=np.concatenate([Rhoc[:,::-1,:],Rhoc],axis=1)
	Rhoc=np.concatenate([Rhoc[:,:,::-1],Rhoc],axis=2)
	Rhow=np.concatenate([Rhow[::-1,:,:],Rhow],axis=0)
	Rhow=np.concatenate([Rhow[:,::-1,:],Rhow],axis=1)
	Rhow=np.concatenate([Rhow[:,:,::-1],Rhow],axis=2)	
	
	#Transposing speed in remaining octants (axial symmetry)
	Vw=np.concatenate([Vw[::-1,:,:],Vw],axis=0)
	Vw=np.concatenate([Vw[:,::-1,:],Vw],axis=1)
	Vw=np.concatenate([Vw[:,:,::-1],Vw],axis=2)
	Vc=np.concatenate([Vc[::-1,:,:],Vc],axis=0)
	Vc=np.concatenate([Vc[:,::-1,:],Vc],axis=1)
	Vc=np.concatenate([Vc[:,:,::-1],Vc],axis=2)
	Vh=np.concatenate([Vh[::-1,:,:],Vh],axis=0)
	Vh=np.concatenate([Vh[:,::-1,:],Vh],axis=1)
	Vh=np.concatenate([Vh[:,:,::-1],Vh],axis=2)	

	#Transposing temperature in remaining octants (axial symmetry)	
	tw=np.concatenate([tw[::-1,:,:],tw],axis=0)
	tw=np.concatenate([tw[:,::-1,:],tw],axis=1)
	tw=np.concatenate([tw[:,:,::-1],tw],axis=2)
	tc=np.concatenate([tc[::-1,:,:],tc],axis=0)
	tc=np.concatenate([tc[:,::-1,:],tc],axis=1)
	tc=np.concatenate([tc[:,:,::-1],tc],axis=2)
	th=np.concatenate([th[::-1,:,:],th],axis=0)
	th=np.concatenate([th[:,::-1,:],th],axis=1)
	th=np.concatenate([th[:,:,::-1],th],axis=2)

	return [Rhow, Rhoc, Rhoh, Vw, Vc, Vh, tw, tc, th]	



#-------------------------------------------------------------------------------
# RT ---------------------------------------------------------------------------
#-------------------------------------------------------------------------------


def Halpha(phi, A, B, Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf, Mdot, Bd, delta, fcl, lda0, lda, phot, vmac, FWHM_lda):
	
	#Defining phase grid size
	PH=len(phi)

	#Defining magnetosphere grid size
	NNx=2*Nx
	NNy=2*Ny
	NNz=2*Nz

	#Conversion of stellar properties into cgs 
	mdot=Mdot*Msol/(365.*24*3600.)
	vinf=Vinf*100000.
	rstar=Rstar*Rsol
	mstar=Mstar*Msol
	ve = np.sqrt(2.*G*mstar/rstar)
	Ve = np.sqrt(2.*G*mstar/rstar)/100000.
	rhowstar = mdot/(4.*np.pi*rstar**2*vinf)
	rhocstar = rhowstar*vinf/ve

	#Some scalling relations (see Owocki 2016)
	chiinf = 0.034*(vinf/10.**8)**4*(rstar/10**12)/(Mdot/10**(-6))
	Tinf = 14*10**6*(vinf/10.**8)**2

	#Computing the Alfven radius and closure radius
	Beq = Bd/2.
	eta = (Beq)**2*rstar**2/(mdot*vinf)
	RA = 0.3+(eta+0.25)**(0.25)
	Rc = RA # This can be changed occording to the user

	#ADM output (the magnetosphere density components)
	admOUT=admCAL(Nx, Ny, Nz, RA, Rc, Teff, Tinf, chiinf, delta )
	rhow=admOUT[0]*rhowstar*np.sqrt(fcl)
	rhoc=admOUT[1]*rhocstar*np.sqrt(fcl)
	rhoh=admOUT[2]*rhowstar*np.sqrt(fcl)
	vw=admOUT[3]*vinf
	vc=admOUT[4]*ve
	vh=admOUT[5]*vinf
	tw=admOUT[6]
	tc=admOUT[7]
	th=admOUT[8]

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)
	dX=np.abs(XX[0]-XX[1])
	dY=np.abs(YY[0]-YY[1])
	dZ=np.abs(ZZ[0]-ZZ[1])
	dx=dX*Rstar*Rsol
	dy=dY*Rstar*Rsol
	dz=dZ*Rstar*Rsol

	#Creating 3D meshgrids
	X_grid, Y_grid = np.meshgrid( XX, YY, indexing='xy')
	P_grid = np.sqrt( X_grid**2 + Y_grid**2 )
	Z_grid, X_grid, Y_grid = np.meshgrid(ZZ, XX, YY, indexing='xy')
	R_grid = np.sqrt( Z_grid**2 + X_grid**2 + Y_grid**2 )
	MU_grid = X_grid / R_grid
	PHI_grid = np.arctan2(Y_grid,Z_grid)

	#Dipole cartesian unit vectors
	xhat = ( 3*MU_grid*np.sqrt(1.-MU_grid**2) )/np.sqrt(1.+3.*MU_grid**2)*np.cos(PHI_grid)
	yhat = ( 3*MU_grid*np.sqrt(1.-MU_grid**2) )/np.sqrt(1.+3.*MU_grid**2)*np.sin(PHI_grid)
	zhat = ( 3*MU_grid**2 - 1. )/np.sqrt(1.+3.*MU_grid**2) 

	#Rest frequency
	nu0 = c/lda0
	nu = c/lda

	#Setting electron and radiative temperatures
	T_e=0.75*Teff
	T_rad=0.77*Teff
	S=Bnu(nu0,T_e)/Bnu(nu0,T_rad) #This can be changed for a more sophisticated temperature structure

	#Variable setup
	NU=len(nu)
	Pem=np.zeros([PH,NU])
	Pabs=np.zeros([PH,NU])
	P=np.zeros([PH,NU])
	W=np.zeros(PH)
	I0=np.zeros([NNx,NNy])
	I0[ P_grid<1.0 ]=1.
	for ph in range(0,PH):	
		
		alpha=np.arccos(csalpha2(phi[ph]*2.*np.pi,np.radians(A),np.radians(B)))

		RHOw_rot=np.zeros([NNz,NNx,NNy])
		RHOc_rot=np.zeros([NNz,NNx,NNy])
		RHOh_rot=np.zeros([NNz,NNx,NNy])
		Vw_rot=np.zeros([NNz,NNx,NNy])
		Vc_rot=np.zeros([NNz,NNx,NNy])
		Vh_rot=np.zeros([NNz,NNx,NNy])
		Tw_rot=np.zeros([NNz,NNx,NNy])
		Tc_rot=np.zeros([NNz,NNx,NNy])
		Th_rot=np.zeros([NNz,NNx,NNy])
		Vzhat_d = np.zeros([NNz,NNx,NNy])

		#Rotating dipole unit vector according to rotational phase
		xhat_rot = xhat
		yhat_rot = yhat*np.cos(alpha) - zhat*np.sin(alpha)
		zhat_rot = yhat*np.sin(alpha) + zhat*np.cos(alpha)

		#Trick to make dipole filed lines of opposing colatitude negative in sign 
		if alpha < np.pi/4 or alpha > 3.*np.pi/4:
			Vzhat_d[ X_grid > -np.tan(alpha)*Y_grid ] = -1.
			Vzhat_d[ X_grid < -np.tan(alpha)*Y_grid ] = 1.
		if alpha > np.pi/4 and alpha < 3.*np.pi/4:
			Vzhat_d[ (Y_grid > -(1./np.tan(alpha))*X_grid )] = -1.			
			Vzhat_d[ (Y_grid < -(1./np.tan(alpha))*X_grid) ] = 1.	

		#Line-of-sight unit vector in the rotated frame
		Vzhat_d = Vzhat_d*zhat_rot

		#Rotation of density, speed and temperature cubes 
		for k in range(0,NNx):
			RHOw_rot[:,k,:] = rotate(rhow[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			RHOc_rot[:,k,:] = rotate(rhoc[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			RHOh_rot[:,k,:] = rotate(rhoh[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			Vw_rot[:,k,:]  = rotate(vw[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			Vc_rot[:,k,:] = rotate(vc[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			Vh_rot[:,k,:] = rotate(vh[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			Tw_rot[:,k,:] = rotate(tw[:,k,:],np.degrees(alpha),reshape=False,cval=Teff)
			Tc_rot[:,k,:] = rotate(tc[:,k,:],np.degrees(alpha),reshape=False,cval=Teff)
			Th_rot[:,k,:] = rotate(th[:,k,:],np.degrees(alpha),reshape=False,cval=Teff)

		#Line-of-sight component in the rotated frame
		Vzw_rot = -Vw_rot*Vzhat_d
		Vzc_rot = Vc_rot*Vzhat_d
		Vzh_rot = -Vh_rot*Vzhat_d

		#Removing occulted regions
		RHOw_rot[ R_grid<1.0 ]=0.
		RHOw_rot[ (np.sqrt(Z_grid**2+Y_grid**2)<1) & (X_grid<0) ] = 0
		RHOc_rot[ R_grid<1.0 ]=0.
		RHOc_rot[ (np.sqrt(Z_grid**2+Y_grid**2)<1) & (X_grid<0) ] = 0
		RHOh_rot[ R_grid<1.0 ]=0.
		RHOh_rot[ (np.sqrt(Z_grid**2+Y_grid**2)<1) & (X_grid<0) ] = 0

		#Electron density
		New_rot=RHOw_rot*alphae/mp
		Nec_rot=RHOc_rot*alphae/mp
		Neh_rot=RHOh_rot*alphae/mp

		#Proton density
		Npw_rot=RHOw_rot*alphap/mp
		Npc_rot=RHOc_rot*alphap/mp
		Nph_rot=RHOh_rot*alphap/mp	

		for ll in range(0,NU):

			#Convert wavelength to velocity space
			v = (lda[ll]-lda0)/lda0*c
			vth = (2.*kb*Teff/mp)**0.5

			#Gaussian variance
			vtot = np.sqrt(vth**2 + vmac**2)
			nuD = nu0*vtot/c

			#Lorentzian FWHM
			FWHM_nu = FWHM_lda*c/lda0**2
			nuN = FWHM_nu/(2.*np.pi)

			#Doppler shifts
			x = nu[ll] - nu0
			u_w = x - nu0*Vzw_rot/c
			u_c = x - nu0*Vzc_rot/c
			u_h = x - nu0*Vzh_rot/c

			#Incremental optical depth
			dtau_w = chi32(u_w,nuD,nuN,Npw_rot,New_rot,T_e)
			dtau_c = chi32(u_c,nuD,nuN,Npc_rot,Nec_rot,T_e)
			dtau_h = 0.#chi32(u_h0,nuD,nuN,Nph_rot,Neh_rot,T_e)
			dtau = dtau_w + dtau_c + dtau_h
				
			#Optical depth
			tauinf_w = trapz(dtau_w,ZZ*Rsol*Rstar,axis=0)
			tauinf_c = trapz(dtau_c,ZZ*Rsol*Rstar,axis=0)
			tauinf_h = 0.#trapz(dtau_h0,Z*Rsol*Rstar,axis=0)
			tauinf = trapz(dtau,ZZ*Rsol*Rstar,axis=0)
			
			#Emergent intensity and flux
			tau = cumtrapz(dtau,ZZ*Rsol*Rstar,axis=0,initial=0.)
			Iabs = I0*np.exp(-tauinf)*phot[ll]
			Iem = (1.-np.exp(-tauinf))*S #trapz(np.exp(-tau)*S,tau,axis=0)
			P[ph,ll] = trapz(trapz(Iabs+Iem,XX),YY)/np.pi

		#Calculation of equivalent width
		W[ph]=ew(lda*10**8,P[ph,:]/P[ph,0],lda[0]*10**8,lda[-1]*10**8,yerr='None')

	return [W,P]


def ew(x,y,a,b,yerr='None'):

	if yerr != 'None':
		integrand=1.-y
		integranderr=yerr

		index=np.arange(len(x))
		index=index[ (x>a) & (x<b)]

		EW=0.
		errEW=0.
		for k in (index-1):
			EW+=0.5*(x[k+1]-x[k])*(integrand[k+1]+integrand[k])
			errEW+=0.5*(x[k+1]-x[k])*(integranderr[k+1]+integranderr[k])
		
		out=[-EW,errEW]

	else:
		integrand=1.-y

		index=np.arange(len(x))
		index=index[ (x>a) & (x<b) ]

		EW=0.
		for k in (index-1):
			EW+=0.5*(x[k+1]-x[k])*(integrand[k+1]+integrand[k])
			
		out=-EW

	return out



