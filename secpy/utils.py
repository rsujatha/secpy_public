import numpy as np
import astropy.constants as const
from scipy.integrate import simps
from scipy.interpolate import interp1d

c_light = const.c.value
h_planck = const.h.value
k_B = const.k_B.value
jansky = 1.0e-23            # erg/s/cm/cm/Hz
arcmin2rad = np.pi / 180. / 60. 



def btheta (fwhm_arcmin, theta):
	""" 
	Returns the real-space symmetric Gaussian beam.

	Parameters
	----------
	fwhm_arcmin : float
		Beam full-width-at-half-maximum (fwhm) in arcmin.

	theta : float
		Angle.

	Returns
	-------
	bl : array
		Gaussian beam function

	"""
	sigmab = fwhm_arcmin/(np.sqrt(8.*np.log(2.)))
	# thetas = np.linspace(0, theta, 1000)
	return np.exp(-theta**2/(2.*sigmab**2))/(2.*np.pi*sigmab**2)

def btheta2D (fwhm_arcmin, reso=0.2, theta_max=5):
	""" 
	Returns the real-space 2D symmetric Gaussian beam.

	Parameters
	----------
	fwhm_arcmin : float
		Beam full-width-at-half-maximum (fwhm) in arcmin.

	reso : float
		Resolution of the 2d array [arcmin/pix]
	
	theta_max : float
		Angle.

	Returns
	-------
	bl : 2d array
		Gaussian beam function

	"""
	sigmab = fwhm_arcmin/(np.sqrt(8.*np.log(2.)))
	theta_x,theta_y = np.meshgrid(np.arange(-theta_max,theta_max+reso,reso), np.arange(-theta_max,theta_max+reso,reso)) 
	thetas = np.sqrt(theta_x**2+theta_y**2) # arcmin
	return np.exp(-thetas**2/(2.*sigmab**2))/(2.*np.pi*sigmab**2)


def bl(fwhm_arcmin, lmax=3000):
	""" 
	Returns the map-level transfer function for a symmetric Gaussian beam.

	Parameters
	----------
	fwhm_arcmin : float
		Beam full-width-at-half-maximum (fwhm) in arcmin.

	lmax : int
		Maximum multipole.

	Returns
	-------
	bl : array
		Gaussian beam function

	"""
	ls = np.arange(0, lmax+1)
	return np.exp( -ls*(ls+1.) * (fwhm_arcmin * np.pi/180./60.)**2 / (16.*np.log(2.)) )

def nl_cmb(noise_uK_arcmin, fwhm_arcmin, lmax=3000, lknee=None, alpha=None):
	""" 
	Returns the beam-deconvolved noise power spectrum in units of uK^2 for

	Parameters
	----------
	noise_uK_arcmin : float or list  
		Map noise level in uK-arcmin 

	fwhm_arcmin : float or list
		Beam full-width-at-half-maximum (fwhm) in arcmin, must be same size as noise_uK_arcmin

	lmax : int
		Maximum multipole.
	"""
	ls = np.arange(0, lmax+1)
	if np.isscalar(noise_uK_arcmin) or (np.size(noise_uK_arcmin) == 1):
		if (lknee is not None) and (alpha is not None):
			return  ((noise_uK_arcmin * np.pi/180./60.)**2 / bl(fwhm_arcmin, lmax=lmax)**2) * (1. + (lknee/ls)**alpha)
		else:   
			return  ((noise_uK_arcmin * np.pi/180./60.)**2 / bl(fwhm_arcmin, lmax=lmax)**2)
	else:
		return 1./np.sum([1./nl_cmb(noise_uK_arcmin[i], fwhm_arcmin[i], lmax=lmax, lknee=lknee, alpha=alpha) for i in range(len(noise_uK_arcmin))], axis=0)

def B_nu(nu, T_cmb=2.725, MJy_sr=False):
	""" 
	Returns the planck blackbody function (in W sr^{-1} Hz^{-1} or MJy/sr)
	at frequency \nu (in GHz) for a blackbody with temperature T (in K). 

	Parameters
	----------
	nu : float
		Frequency in GHz
	"""
	x = h_planck*(nu*1e9)/(k_B*T_cmb)
	if MJy_sr:
		return 2*h_planck*(nu*1e9)**3 / c_light**2 / (np.exp(x) - 1.) / (1e6*jansky)
	else:
		return 2*h_planck*(nu*1e9)**3 / c_light**2 / (np.exp(x) - 1.)

def dB_nu_dT(nu, T_cmb=2.725):
	""" 
	Returns the derivative of the planck blackbody function (in W sr^{-1} Hz^{-1})
	at frequency \nu (in GHz) for a blackbody with temperature T (in K). 

	Parameters
	----------
	nu : float
		Frequency in GHz
	"""
	x = h_planck*(nu*1e9)/(k_B*T_cmb)
	return 2.*k_B/c_light**2 * x**2*np.exp(x)/(np.exp(x)-1)**2

def Kcmb2MJysr(nu, DeltaT, T_cmb=2.725):
	""" 
	Returns the planck blackbody function (in W sr^{-1} Hz^{-1} or MJy/sr)
	at frequency \nu (in GHz) for a blackbody with temperature T (in K). 

	Parameters
	----------
	nu : float
		Frequency in GHz
	"""
	x = h_planck*(nu*1e9)/(k_B*T_cmb)
	bb = B_nu(nu, T_cmb=T_cmb, MJy_sr=True)
	return bb * x * np.exp(x)/(np.exp(x)-1.) * DeltaT

def RJ_law(nu, T_cmb=2.725, MJy_sr=False):
	"""
	The Rayleigh Jeans limit of Planck's law for h*nu << kT
	
	Parameters
	----------
	nu : float or array
		Frequency in GHz
		
	Returns
	-------
	B_nu : float or numpy.ndarray
		specific intensity in MJy/sr
	"""
	B_nu = 2. * (1e9*nu)**2 * k_B * T_cmb / c_light**2
	if MJy_sr:
		return B_nu / (1e6*jansky)
	else:
		return B_nu

def dT_dB(nu, T_cmb=2.725):
	"""
	The inverse of the derivative of Planck's law with respect to temperature.
	
	Parameters
	----------
	nu : float or numpy.ndarray 
		Frequency in GHz
	"""

	return dB_nu_dT(nu, T_cmb)**(-1.)

def j2k(nu, T_cmb=2.725):
	""" 
	Returns the conversion factor between Jansky units and CMB Kelvin. 

	Parameters
	----------
	nu : float
		Frequency in [GHz]
	"""
	x = h_planck*(nu*1e9)/(k_B*T_cmb)
	g = (np.exp(x) - 1.)**2 / x**2 / np.exp(x)
	return c_light**2 / (2. * (nu*1e9)**2 * k_B) * g * 1.e-26

def k2j(nu, T_cmb=2.725):
	""" 
	Returns the conversion factor between CMB Kelvin and Jansky units. 

	Parameters
	----------
	nu : float
		Frequency in [GHz]
	"""
	return 1.0 / j2k(nu, T_cmb=T_cmb)

def f_sz(nu, T_cmb=2.725):
	"""
	The frequency dependence of the thermal SZ effect
	
	Parameters
	----------
	nu : float or array
		the frequency in GHz
	"""
	x = h_planck*1e9*nu / k_B / T_cmb
	return x*(np.exp(x) + 1.) / (np.exp(x) - 1.) - 4.0

def g_sz(nu, T_cmb=2.725):
	"""
	The frequency dependence of the thermal SZ effect
	
	Parameters
	----------
	nu : float or array
		the frequency in GHz
	"""
	x = h_planck*1e9*nu / k_B / T_cmb
	return x**4 * np.exp(x) / (np.exp(x)-1)**2 * f_sz(nu, T_cmb=T_cmb)

def GHz_to_lambda(ghz):
	"""
	Converts from GHz to wavelenght (in micron)

	Parameters
	----------
	nu : float
		Frequency in [GHz]
	"""
	lam = c_light/ghz * 1e-3
	return lam

def lambda_to_GHz(lam):
	"""
	Converts from wavelenght (in micron) to GHz

	Parameters
	----------
	lam : float
		Wavelenght in [micron]
	"""
	hz  = c_light/(lam*1e-6)
	ghz = 1e-9*hz
	return ghz

def W_k_tophat(k):
	""" 
	Returns the Fourier Transform of a tophat window function. 
	"""
	return 3./k**3*(np.sin(k) - k*np.cos(k))

def dW_k_tophat(k):
	""" 
	Returns the derivative w.r.t. k of the fourier transform of a tophat window function. 
	"""
	return -9./k**4*(np.sin(k) - k*np.cos(k)) + 3./k**2 * np.sin(k)

def dW_lnk_tophat(k):
	""" 
	Returns the derivative w.r.t. \ln{k} of the fourier transform of a tophat window function. 
	"""
	return np.where(k > 1e-3, (9 * k * np.cos(k) + 3 * (k ** 2 - 3) * np.sin(k)) / k**3, 0)

def W_tilde(x):
	""" 
	Eq. 16 of Mueller+14 (astro-ph:1408.6248)
	"""
	return (2.*np.cos(x) + x*np.sin(x))/x**3

def W_Delta(k, Rmin, Rmax):
	""" 
	Eq. 15 of Mueller+14 (astro-ph:1408.6248)
	"""
	return 3. * (Rmin**3 * W_tilde(k*Rmin) - Rmax**3 * W_tilde(k*Rmax)) / (Rmax**3 - Rmin**3)

def V_bin(r, dr):
	return np.pi/3. * (dr * (dr**2 + 12*r**2))

def V_sphere(r):
	return 4.*np.pi/3.*r**3

def Diameter2Theta(D, nu):
	""" 
	Returns the beam [arcmin] corresponding to a given telescope diameter D [m] operating at a frequency nu [GHz].

	Parameters
	----------
	D : float
		Diameter of the telescope in meters.

	nu : int
		Frequency in GHz.

	Returns
	-------
	theta : array
		Beam in arcmin
	"""
	return np.degrees(1.22 * (c_light/(nu*1e9))/D) * 60.

def GetLxLy(nx, dx, ny=None, dy=None, shift=False):
    """ 
    Returns two grids with the (lx, ly) pair associated with each Fourier mode in the map. 
    If shift=True , \ell = 0 is centered in the grid
    ~ Note: already multiplied by 2\pi 
    """
    if ny is None: ny = nx
    if dy is None: dy = dx
    
    dx *= arcmin2rad
    dy *= arcmin2rad
    
    if shift:
        return np.meshgrid( np.fft.fftshift(np.fft.fftfreq(nx, dx))*2.*np.pi, np.fft.fftshift(np.fft.fftfreq(ny, dy))*2.*np.pi )
    else:
        return np.meshgrid( np.fft.fftfreq(nx, dx)*2.*np.pi, np.fft.fftfreq(ny, dy)*2.*np.pi )

def GetL(nx, dx, ny=None, dy=None, shift=False):
    """ 
    Returns a grid with the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode in the map. 
    If shift=True, \ell = 0 is centered in the grid
    """
    lx, ly = GetLxLy(nx, dx, ny=ny, dy=dy, shift=shift)
    return np.sqrt(lx**2 + ly**2)

def Interpolate2D(nx, dx, l, cl, dy=None, ny=None, shift=False):
    """ 
    Returns a function cl interpolated on the 2D L plane.
    If shift=True (default), \ell = 0 is centered in the grid
    """
    if ny is None: ny = nx
    if dy is None: dy = dx
 
    L = GetL(nx, dx, ny=ny, dy=dy, shift=shift)
    CL = np.interp(L, l, cl)

    return CL


def n_distribution(z, alpha=2, zstar=0.5, beta=1):
	""" 
	Galaxy number density distribution for LSST like experiments. 
	Eq 13.10 of https://arxiv.org/pdf/0912.0201.pdf  (page: 472)
    """
	pz=(z**alpha) * np.exp(-(z/zstar)**beta)
	return pz

def get_norm_gal(Ntotal):
	""" 
	The normalization factor for the total number of galaxies. 
	Ntotal: total number of galaxies per unit arc-min^2
    """
	zs=np.linspace(0.0, 10, num=50)
	pz=n_distribution(zs)
	res=simps(pz, zs)
	A=Ntotal/ res
	return A

def get_nbar(z, Ntotal=50):
	"""
	The normalized distribution of galaxies.
	"""
	norm=get_norm_gal(Ntotal)
	return n_distribution(z) /(Ntotal* arcmin2rad**2 )

def galaxy_shot_noise(zmin, zmax):
	"""
	Shot noise
	"""
	zsn=np.linspace(zmin, zmax, num=100)
	SN_g=1.0/get_nbar(zsn, Ntotal=50)
	SN=simps(SN_g, zsn)
	return SN
	
def SNR_xy(lmin_sn, lmax_sn, ells,clxy,clxx,nlxx,clyy,nlyy,fsky=None):
    larray=np.linspace(lmin_sn, lmax_sn, num=int(lmax_sn-lmin_sn+1))
    clxy_int=interp1d(ells,clxy)
    clxx_int=interp1d(ells,clxx)
    clyy_int=interp1d(ells,clyy)
    nlxx_int=interp1d(ells,nlxx)
    nlyy_int=interp1d(ells,nlyy)
    
    if fsky:
        res=np.sqrt(np.sum(clxy_int(larray)**2/((clxx_int(larray)+nlxx_int(larray))*(clyy_int(larray)+nlyy_int(larray)))*(2.*larray + 1)*fsky))
    else:
        res=np.sqrt(np.sum(clxy_int(larray)**2/((clxx_int(larray)+nlxx_int(larray))*(clyy_int(larray)+nlyy_int(larray)))*(2.*larray + 1)))
    return res

