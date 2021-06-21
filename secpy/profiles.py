#!/usr/bin/env python2 -tt
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:30:39 2020

@author: anirbanroy
"""

import numpy as np
from scipy import integrate
from scipy.special import erf, sici
from scipy.optimize import newton
from scipy import interpolate

from secpy.utils import btheta2D, Interpolate2D
from secpy.mass_func import MassFunction
from default_params import *

from astropy.convolution import convolve
import astropy.constants as const
import astropy.units as u

#import matplotlib.pyplot as plt
#from numba import jit
import matplotlib as pl
pl.rcParams['figure.figsize'] = 7,7
pl.rcParams['ytick.minor.visible'] =True
pl.rcParams['xtick.minor.visible'] = True
pl.rcParams['xtick.top'] = True
pl.rcParams['ytick.right'] = True
pl.rcParams['font.size'] = '22'
pl.rcParams['legend.fontsize'] = '16'
pl.rcParams['legend.borderaxespad'] = 1.0
#pl.rcParams['legend.numpoints'] = '1'

pl.rcParams['lines.dashed_pattern']= 3.0, 1.9
pl.rcParams['lines.dotted_pattern']= 3.0, 1.9

def update_params(params_old, params_to_update):
    for i in params_old:
        for j in params_to_update:
            if i==j:
                params_old[i]=params_to_update[j]
    return params_old

def Si(x):
	return sici(x)[0]
	
def Ci(x):
	return sici(x)[1]
	

# DEFAULT HOD PARAMS
M_min_default = 10**11.57
sigma_logm_default = 0.15
M_cut_default = 10**12.23
M1_default = 10**12.75
alpha_g_default = 0.99


def m_int_nfw(x):
	return np.log(1.+x) - x/(1.+x)

def nfw_dimensionless_mass(x):
    """
    The dimensionless function f(x) giving the mass enclosed within a
    dimensionless radius for an NFW profile.

    Notes
    -----
    As given in Eq. C3 of Hu and Kravtsov 2002. Given by:

    ..math: f(x) = x^3 * [ ln(1 + 1/x) - 1/(1+x) ]

    and the mass enclosed at radius r_h is:

    ..math: M_h(r_h) = 4*\pi*rho_s*r_h^3 * f(r_s/r_h)
    """
    return x**3 * ( np.log(1. + 1./x) - 1./(1+x) )
    

def rho_fit_Battaglia16(x, M, z, x_c=0.5, gamma=-0.2, sim='AGN'): # unitless
    params_scale=lambda A_0, alpha_m, alpha_z: A_0 * (M/1e14)**alpha_m * (1.+z)**alpha_z

    if sim == 'AGN' or sim == 'agn':
        az_rho= 4e3
        alpham_rho = 0.029
        alphaz_rho= -0.66

        az_beta= 3.83
        alpham_beta= 0.04
        alphaz_beta= -0.025

        az_alpha= 0.88
        alpham_alpha= 0.03
        alphaz_alpha= 0.19

    if sim == 'shock' or sim == 'SHOCK':
        az_rho= 1.9e4
        alpham_rho= 0.09
        alphaz_rho= -0.95

        az_beta= 4.43
        alpham_beta= 0.005
        alphaz_beta= 0.037

        az_alpha= 0.7
        alpham_alpha= -0.017
        alphaz_alpha= 0.27



    rho_0=params_scale(az_rho,alpham_rho,alphaz_rho)
    alpha=params_scale(az_alpha,alpham_alpha,alphaz_alpha)
    beta=params_scale(az_beta,alpham_beta,alphaz_beta)

    x_over_xc = x / x_c
    rho_fit = rho_0 * x_over_xc**gamma * (1.+(x_over_xc)**alpha)**((-beta+gamma)/alpha)
    return rho_fit



def pressure_fit_Battaglia12(x, M, z, alpha=1.0, gamma=-0.3):

    """
	Average pressure profile as function of halo mass and redshift from Battaglia's model, see Eq.(10) of arXiv:1109.3711
	! x = r / r_200
	! M_Delta in M_sun
	"""
    params_scale=lambda A_0, alpha_m, alpha_z: A_0 * (M/1e14)**alpha_m * (1.+z)**alpha_z

    az_p= 18.1
    alpham_p= 0.154
    alphaz_p= -0.758

    az_xc= 0.497
    alpham_xc= -0.00865
    alphaz_xc= 0.731

    az_betap= 4.350
    alpham_betap= 0.0393
    alphaz_betap= 0.415

    P_0=params_scale(az_p,alpham_p, alphaz_p)
    x_c=params_scale(az_xc,alpham_xc, alphaz_xc)
    beta=params_scale(az_betap,alpham_betap, alphaz_betap)


    x_over_xc = x / x_c

    return P_0 * x_over_xc**gamma * (1.+(x_over_xc)**alpha)**(-beta)




def c_Duffy(M, z, h, kind='200'):
	"""
	Concentration from c(M) relation published in Duffy et al. (2008).
	"""

	M_pivot = 2.e12/h # [M_sun]

	if kind == '200':
		A = 5.71
		B = -0.084
		C = -0.47
	elif kind == 'vir':
		A = 7.85
		B = -0.081
		C = -0.71

	concentration = A * ((M / M_pivot)**B) * (1+z)**C

	return concentration

def c_SCP(M, z, h, kind='200'):
	"""
	Concentration from c(M) relation published in Sanchez-Conde & Prada (2014).
	To be tested
	"""

	M_pivot = 1./h # [M_sun]

	if kind == '200':
		c_i=np.array([37.5153,-1.5093,1.636e-02,3.66e-4,-2.89237e-05,5.32e-7])
		M_i=np.array([np.log(M)**i for i in range(6)])
		c_200=np.dot(c_i,M_i)
		C = -1
	else :
		raise ValueError ("Not implemented kind!=200 for c_SCP")

	concentration = c_200 * (1+z)**C

	return concentration

def convert_halo_mass(mass, z, input_definition, output_definition, output_delta, cosmo):
    """
    Convert between definitions of halo mass, assuming a NFW profile and the
    concentration relation from Duffy et al. 2008.

    Notes
    -----
    Input mass must be defined either wrt 200x crit density, 200x
    mean density, or virial radius.

    Parameters
    ----------
    mass : {float, array_like}
        The input mass to convert [units :math: `M_\odot h^{-1}`]
    z : float
        The redshift of the halo
    input_definition : {`critical`, `mean`, `virial`}
        The input mass definition
    output_definition : {`critical`, `mean`, `virial`}
        The output mass definition
    output_delta : float
        The desired overdensity for the output mass.
    params : {str, dict, Cosmology}
        The cosmological parameters to use. Default is set by the value
        of ``parameters.default_params``
    """
    # mass = tools.vectorize(mass)

    # if not isinstance(params, Cosmology):
    #     params = Cosmology(params)

    # check the input defnitionas
    choices = ['critical', 'virial']
    if input_definition not in choices:
        raise ValueError("Input mass definition '%s' must be one of: %s" %(input_definition, choices))
    if output_definition not in choices:
        raise ValueError("Output mass definition '%s' must be one of: %s" %(output_definition, choices))

    alpha_out = 1.
    if output_definition == 'critical':
        alpha_out = 1.
    elif output_definition == 'virial':
        alpha_out = 1.
        output_delta = cosmo.Dv_cz(z)

    input_delta = 200.
    alpha_in = 1.
    if input_definition == 'critical':
        alpha_in = 1.
    elif input_definition == 'virial':
        alpha_in = 1.
        input_delta = cosmo.Dv_cz(z)

    # get the concentration for the input mass
    if input_definition == 'critical':
	    concen = c_Duffy(mass, z, cosmo.h, kind='200')
    elif input_definition == 'virial':
	    concen = c_Duffy(mass, z, cosmo.h, kind='vir')

    f_in = nfw_dimensionless_mass(1./concen)

    def objective(x):
        return nfw_dimensionless_mass(x) - f_in*output_delta*alpha_out/(input_delta*alpha_in)

    x_final = newton(objective, 1.)
    return (x_final*concen)**(-3.) * (alpha_out*output_delta) / (alpha_in*input_delta) * mass

#end convert_halo_mass



#-------------------------------------------------------------------------------
def convert_halo_radius(radius, z, input_definition, output_definition, output_delta, cosmo):
    """
    Convert between definitions of halo radius, assuming a NFW profile and the
    concentration relation from Duffy et al. 2008.

    Notes
    ------
    Input radius must be defined either wrt 200x crit density, 200x
    mean density, or virial radius.

    Parameters
    ----------
    radius : {float, array_like}
        The input radius to convert [units :math: `Mpc h^{-1}`]
    z : float
        The redshift of the halo
    input_definition : {`critical`, `mean`, `virial`}
        The input mass definition
    output_definition : {`critical`, `mean`, `virial`}
        The output mass definition
    output_delta : float
        The desired overdensity for the output mass.
    params : {str, dict, Cosmology}
        The cosmological parameters to use. Default is set by the value
        of ``parameters.default_params``
	output_radius: The output radius in [units :math: `Mpc h^{-1}`]
    """
    # radius = tools.vectorize(radius)

    # if not isinstance(params, Cosmology):
    #     params = Cosmology(params)

    rho_crit = cosmo.rho_c(z) # in kg/m^3
    om_m = cosmo.rho_bar(z)*u.M_sun.to('kg') / u.Mpc.to('m')**3 # in kg/m^3
    conversion_factor = 4./3*np.pi*rho_crit

    # compute the input mass scale
    if input_definition == 'critical':	input_delta = 200.
    if input_definition == 'virial':	input_delta = cosmo.Dv_cz(z) # to be checked output units
    alpha = 1.
    if input_definition == 'mean':	alpha = om_m

    input_mass = conversion_factor * alpha * (radius*u.Mpc.to('m'))**3 * input_delta*u.kg.to('M_sun') # in Msun/h

    # now convert to an output mass
    output_mass = convert_halo_mass(input_mass, z, input_definition, output_definition, output_delta, cosmo) # in Msun/h

    # now convert back to a radius
    if output_delta == 'virial':
        output_delta = cosmo.Dv_cz(z)
    if output_definition=='critical':
        output_delta=output_delta
    alpha = 1.
    if output_definition == 'mean':
        alpha = om_m
    output_radius = u.m.to('Mpc')*(output_mass*u.M_sun.to('kg') / (conversion_factor * alpha * output_delta) )**(1./3)

    return output_radius
#end convert_halo_radius

class ClusterOpticalDepth():

	def __init__(self, cosmo, mass_def=200.):

		self.epsrel = 1.49e-6
		self.epsabs = 0.
		self.cosmo = cosmo
		self.mass_def = mass_def
		# self.concentration = concentration

	def GetTauProfile(self, M, z, x_c=0.5, gamma=-0.2, reso=0.2, theta_max=10, chi=1., fwhm_arcmin=0., profile='battaglia', lowpass=False):
		"""
		Returns the 2D profile of the optical depth for a cluster of mass M (M_sun) at redshift z.
		"""
		theta_x,theta_y = np.meshgrid(np.arange(-theta_max,theta_max+reso,reso), np.arange(-theta_max,theta_max+reso,reso))
		theta = np.sqrt(theta_x**2+theta_y**2) # arcmin

		d_A = self.cosmo.d_A(z) # [Mpc]
		rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]
		f_gas = self.cosmo.omegab/self.cosmo.omegam # we assume f_gas ~ f_baryon
		BAD_TWEAK = 1.4 # Battaglia profile seems lower in amplitude w.r.t. the plots in the paper
		# print f_gas

		R = np.radians(theta/60.) * d_A # [Mpc]
		R = R.flatten()
		rho_fit = np.zeros_like(R)

		# Radius such as M(< R_Delta) = Delta * rho_crit n stuff
		r_v = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.))*(u.m).to(u.Mpc) # [Mpc]

		if profile == 'battaglia' or profile == 'Battaglia':

			for ii in range(len(R)):
				def integrand(r):
					return 2. * rho_fit_Battaglia16(r/r_v, M, z, x_c=x_c, gamma=gamma) * ( r / np.sqrt(r**2. - R[ii]**2.) )
				rho_fit[ii] =  integrate.quad(integrand, R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0]

			norm_for_int = BAD_TWEAK * rho_c_z
			# tau_theta = f_gas * rho_fit * u.Mpc.to('m') * rho_c_z * const.sigma_T.value * chi / (1.14*const.m_p.value)

		elif profile == 'NFW' or profile == 'nfw':
			concentration = c_Duffy(M,z,self.cosmo.h)
			# print concentration

			#scale radius
			r_s = r_v/concentration # [Mpc]

			#NFW profile properties
			delta_c = (self.mass_def/3.)*(concentration**3.)/(np.log(1.+concentration)-concentration/(1.+concentration))

			#make sure that we get same delta_c using num. integration as well
			t1 = (self.mass_def/3.) * r_v**3. # [Mpc]
			t2 = 1./ r_s**3. / integrate.quad(lambda r : r / (r + r_s)**2., 0., r_v, epsabs=self.epsabs, epsrel=self.epsrel)[0]
			delta_c_2 = t1 * t2 #should be same as delta_c and it is same!
			assert round(delta_c,1) == round(delta_c_2,1)

			norm_for_int = 2. * delta_c * rho_c_z * r_s**3. #[kg/m^3 Mpc^3]

			for ii in range(len(R)):
				def integrand(r):
					return  1 / ( r * (r_s + r )**2. ) * ( r / np.sqrt(r**2. - R[ii]**2.))
				rho_fit[ii] =  integrate.quad(integrand, R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0]

		elif profile == 'gNFW_Moore':
			concentration = c_Duffy(M,z,self.cosmo.h)
			# print concentration

			#scale radius
			r_s = r_v/concentration
			alpha, beta, gamma = 1.5, 3., 1.5 #arXiv:1005.0411 (Eq. 27); https://arxiv.org/abs/astro-ph/9903164

			t1 = (self.mass_def/3.) * r_v**3. # [Mpc^3]
			t2 = 1. / integrate.quad(lambda r : r**2. * (2. / ( (r/r_s)**gamma * ( 1 + (r/r_s)**(3-gamma) ) ) ), 0., r_v, epsabs=self.epsabs, epsrel=self.epsrel)[0] # [Mpc^3]
			delta_c = t1 * t2 # [Mpc^6]

			norm_for_int = 2. * delta_c * rho_c_z #[Mpc^6 * kg/m^3]

			for ii in range(len(R)):
				rho_fit[ii] =  integrate.quad(lambda r : (2. / ( (r/r_s)**gamma * ( 1 + (r/r_s)**(3-gamma) ) ) ) * ( r / np.sqrt(r**2. - R[ii]**2.) ), R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0]

		# rho -> tau
		tau_theta = f_gas * rho_fit * norm_for_int * u.Mpc.to('m') * const.sigma_T.value * chi / (1.14*const.m_p.value)

		# 1D -> 2D
		tau_theta = tau_theta.reshape(theta.shape)

		# theta_R = R_Delta/d_A
		# print np.degrees(theta_R)*60.
		# tau_theta = (1.+theta**2/(np.degrees(theta_R)*60.)**2)**(-1.)

		# Apply beam smoothing
		if fwhm_arcmin != 0.:
			tau_theta = convolve(tau_theta, btheta2D(fwhm_arcmin, reso=reso, theta_max=theta_max), normalize_kernel=True)

		# Smoothly filter out scales below the beam
		if lowpass:
			l = np.arange(0,5e4)
			f_l =  np.exp(-(l/(np.pi/np.radians(fwhm_arcmin/60.)))**4)
			filt = Interpolate2D(tau_theta.shape[1], reso, l, f_l)
			tau_theta = np.fft.ifft2(np.fft.fft2(tau_theta)*filt).real
			# plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(tau_theta))))
		return tau_theta

	def GetApertureTau(self, M, z, theta_R, x_c=0.5, gamma=-0.2, reso=0.2, theta_max=10., chi=1., fwhm_arcmin=0., profile='Battaglia', lowpass=False):
		# if theta_R > theta_max:
		# 	theta_max = theta_R * 2.

		tau_profile = self.GetTauProfile(M, z, x_c=x_c, gamma=gamma, reso=reso, theta_max=theta_max, chi=chi, fwhm_arcmin=fwhm_arcmin, profile=profile, lowpass=lowpass)

		theta_R_pix = theta_R/reso

		apertures = CircularAperture([(tau_profile.shape[1]/2., tau_profile.shape[0]/2.)], r=theta_R_pix)
		annulus_apertures = CircularAnnulus([(tau_profile.shape[1]/2., tau_profile.shape[0]/2.)], r_in=theta_R_pix, r_out=theta_R_pix*np.sqrt(2))
		apers = [apertures, annulus_apertures]

		phot_table = aperture_photometry(tau_profile, apers)
		bkg_mean = phot_table['aperture_sum_1'] / annulus_apertures.area()
		bkg_sum = bkg_mean * apertures.area()
		final_sum = phot_table['aperture_sum_0'] - bkg_sum
		phot_table['residual_aperture_sum'] = final_sum

		return phot_table['residual_aperture_sum'][0]/apertures.area()

		# apertures = CircularAperture([(tau_profile.shape[1]/2., tau_profile.shape[0]/2.)], r=theta_R_pix)
		# phot_table = aperture_photometry(tau_profile, apertures)

		# return phot_table['aperture_sum'][0]/(np.pi*theta_R_pix**2)



class ClusterElectronTau():
    def __init__(self, cosmo, mass_def=200.):
        self.epsrel = 1.49e-6
        self.epsabs = 0.
        self.cosmo = cosmo
        self.mass_def = mass_def


    def GetProfile1D(self, M, z, alpha=1.0, gamma=-0.3, X=0.76):
        Pth2Pe = (2*X+2.)/(5*X+3)
        rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]
        f_gas = self.cosmo.omegab/self.cosmo.omegam # we assume f_gas ~ f_baryon

		# Radius such as M(< R_Delta) = Delta * rho_crit n stuff
        r_v = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
        P_Delta = self.mass_def * rho_c_z * f_gas * const.G.value * M / 2. / r_v #[M_sun/Mpc/s^2]
        P_th = P_Delta * pressure_fit_Battaglia12(x, M, z, alpha=alpha, gamma=gamma)

        return P_th * Pth2Pe #[M_sun/Mpc/s^2]


    def tau_ell(self, ell, M, z, profile='battaglia', npts=100, xmax=100):
        if profile =='battaglia':
            tau_ell=self.tau_ell_battaglia(ell, M, z, x_c=0.5, gamma=-0.2, X=0.76, npts=npts, xmax=xmax)
        #else:
        #	raise ValueError("Unknown profile type")
        return tau_ell


    def tau_ell_battaglia(self, ell, M, z, x_c=0.5, gamma=-0.2, X=0.76, npts=100, xmax=100):

        R200 = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/self.cosmo.rho_c(z))**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
        c200 = c_Duffy(M,z,self.cosmo.h)
        R_s  = R200/c200 # [Mpc]
        ells = self.cosmo.d_A(z)/R_s
        xarr = np.logspace(-5,np.log10(xmax),npts)
        rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]
        f_gas = self.cosmo.omegab/self.cosmo.omegam # we assume f_gas ~ f_baryo
        #r_v = R200 # everything is done in R200


        rho_profile =  f_gas*rho_c_z * rho_fit_Battaglia16(xarr/c200, M, z, x_c=0.5, gamma=-0.2, sim='AGN')#[M_sun/Mpc/s^2]
        arg = ((ell+0.5)*xarr/ells)
        tauell = integrate.simps(xarr**2 * np.sin(arg)/arg * rho_profile, xarr)
        tauell *= 4*np.pi*(R_s) / ells**2 #[Mpc]
        tauell *= u.Mpc.to('m') * const.sigma_T.value / (1.14*const.m_p.value)
        return tauell



class ClusterElectronicPressure():

	def __init__(self, cosmo, mass_def=200.):

		self.epsrel = 1.49e-6
		self.epsabs = 0.
		self.cosmo = cosmo
		self.mass_def = mass_def
		# self.concentration = concentration

	def GetYProfile(self, M, z, alpha=1.0, gamma=-0.3, X=0.76, reso=0.2, theta_max=10, chi=1., fwhm_arcmin=0.):
		"""
		Returns the 2D profile of the compton Y for a cluster of mass M (M_sun) at redshift z.
		"""
		theta_x,theta_y = np.meshgrid(np.arange(-theta_max,theta_max+reso,reso), np.arange(-theta_max,theta_max+reso,reso))
		theta = np.sqrt(theta_x**2+theta_y**2) # arcmin

		d_A = self.cosmo.d_A(z) # [Mpc]
		rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]
		f_gas = self.cosmo.omegab/self.cosmo.omegam # we assume f_gas ~ f_baryon
		Pth2Pe = (2*X+2.)/(5*X+3)
		r_v = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
		P_Delta = self.mass_def * rho_c_z * f_gas * const.G.value * M / 2. / r_v #[M_sun/Mpc/s^2]
		c200 = c_Duffy(M,z,self.cosmo.h)

		R = np.radians(theta/60.) * d_A # [Mpc]
		R = R.flatten()
		y_fit = np.zeros_like(R)


		for ii in xrange(len(R)):
			def integrand(r):
				return 2. * pressure_fit_Battaglia12(r/r_v, M, z, alpha=alpha, gamma=gamma) * ( r / np.sqrt(r**2. - R[ii]**2.) )
			y_fit[ii] =  integrate.quad(integrand, R[ii], np.inf, epsabs=self.epsabs, epsrel=self.epsrel)[0]

		y_fit *= P_Delta * Pth2Pe * u.M_sun.to(u.kg) * const.sigma_T.value / const.m_e.value / const.c.value**2

		y_fit = y_fit.reshape(theta.shape)

		return y_fit

	def GetProfile1D(self, M, z,  alpha=1.0, gamma=-0.3, X=0.76):
		"""
		Returns the 2D profile of the optical depth for a cluster of mass M (M_sun) at redshift z.
		"""
		Pth2Pe = (2*X+2.)/(5*X+3)

		rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]
		f_gas = self.cosmo.omegab/self.cosmo.omegam # we assume f_gas ~ f_baryon

		# Radius such as M(< R_Delta) = Delta * rho_crit n stuff
		r_v = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.))*(u.m).to(u.Mpc) # [Mpc]

		P_Delta = self.mass_def * rho_c_z * f_gas * const.G.value * M / 2. / r_v #[M_sun/Mpc/s^2]
		P_th = P_Delta * pressure_fit_Battaglia12(x, M, z, alpha=alpha, gamma=gamma)

		return P_th * Pth2Pe #[M_sun/Mpc/s^2]
	def y_ell(self, ell, M, z, profile='battaglia', npts=100, xmax=100):
		'''
		Fourier transform of the Compton-y profile using differente electron pressure profiles
		output: y_ell
		'''
		if profile is 'arnaud':
			y_ell=self.y_ell_arnaud(ell, M, z, npts=npts, xmax=xmax)
		elif profile is 'battaglia':
			y_ell=self.y_ell_battaglia(ell, M, z, alpha=1.0, gamma=-0.3, X=0.76, npts=npts, xmax=xmax)
		else: raise ValueError("Unknown profile type")
		return y_ell

	# @jit(nopython=True)
	def y_ell_battaglia(self, ell, M, z, alpha=1.0, gamma=-0.3, X=0.76, npts=100, xmax=100):
		'''
		Fourier transform of the Compton-y profile using Battaglia et al 2012 recipes.
		output: y_ell
		'''
		R200 = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/self.cosmo.rho_c(z))**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
		c200 = c_Duffy(M,z,self.cosmo.h)
		R_s  = R200/c200 # [Mpc]
		ells = self.cosmo.d_A(z)/R_s

		xarr = np.logspace(-5,np.log10(xmax),npts)

		Pth2Pe = (2*X+2.)/(5*X+3)
		rho_c_z = self.cosmo.rho_c(z) # [kg/m^3]
		f_gas = self.cosmo.omegab/self.cosmo.omegam # we assume f_gas ~ f_baryon
		#r_v = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
		r_v = R200 # everything is done in R200
		P_Delta = self.mass_def * rho_c_z * f_gas * const.G.value * M / 2. / r_v #[M_sun/Mpc/s^2]
		# P_th = Pth2Pe * P_Delta * pressure_fit_Battaglia12(x, M, z, alpha=alpha, gamma=gamma)
		pressure_profile = Pth2Pe * P_Delta * pressure_fit_Battaglia12(xarr/c200, M, z, alpha=alpha, gamma=gamma) #[M_sun/Mpc/s^2]
		arg = ((ell+0.5)*xarr/ells)
		yell = integrate.simps(xarr**2 * np.sin(arg)/arg * pressure_profile, xarr)
		yell *= 4*np.pi*(R_s) / ells**2 #[Mpc]
		yell *= u.M_sun.to(u.kg) * const.sigma_T.value / const.m_e.value / const.c.value**2

		return yell

	def y_ell_arnaud(self, ell, M, z, alpha_p=0, P_0=6.41, c500=1.81, alpha=1.0, beta=4.13, gamma=-0.31, npts=100, xmax=100):
		'''
		Fourier transform of the Compton-y using Arnaud et al. 2010 recipe.
		output using: y_ell
		'''
		if self.mass_def!=500: # Arnaud profile is computed in terms of M_500.
			M_500= convert_halo_mass(M,z,'critical','critical',500,self.cosmo)
		else: M_500=M
		R500 = (((M_500*(u.Msun).to(u.kg)/(500*4.*np.pi/3.))/self.cosmo.rho_c(z))**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
		R_s  = R500 # [Mpc] # Arnaud profile works in r/r500
		ells = self.cosmo.d_A(z)/R_s
		m2Mpc= 3.2408e-23 # conoverts m to Mpc
		pression_conversion = 1.60218e-19*1e06/(m2Mpc**2) #convert eV/cm^3 -> J/m^3 = N/m^2 - > N/Mpc^2
		xarr = np.logspace(-5,np.log10(xmax),npts)
		h_norm = 0.7/self.cosmo.h
		p_x = P_0*(h_norm**1.5)/((c500*xarr)**gamma)/((1+(c500*xarr)**alpha)**((beta-gamma)/alpha))
		pressure_profile = 1.65*pression_conversion*(h_norm**-2)*(self.cosmo.E_z(z)**(8./3))*((M_500/3.e14/h_norm)**(2./3. + alpha_p))*p_x
		arg = ((ell+0.5)*xarr/ells)
		yell = integrate.simps(xarr**2 * np.sin(arg)/arg * pressure_profile, xarr)
		yell *= 4*np.pi*(R_s) / ells**2 #[Mpc]
		yell *= m2Mpc*const.sigma_T.value / const.m_e.value / const.c.value**2 # extra factor m2Mpc convertss N=Kg * m / s^2 -> kg*Mpc/s^2 so that y is dimensionless
		return yell


class ClusterDensity():
	def __init__(self, cosmo, hmf=None, zgal=None, ngal=None, gal_dist_name="HOD",Mmin=default_limits['halo_Mmin'],
		Mmax=default_limits['halo_Mmax'], mass_def=200.):
	
		self.epsrel = 1.49e-6
		self.epsabs = 0.
		self.cosmo = cosmo
		self.zgal=zgal
		self.ngal = ngal
		self.mass_def = mass_def
		self.Mmin= Mmin
		self.Mmax=Mmax
		self.gal_dist_name=gal_dist_name
		if hmf is None:
			self.hmf = MassFunction(self.cosmo)
		else:
			self.hmf = hmf

		zs = np.logspace(-3,1,1000)
		#self.nbar_spline = interpolate.interp1d(zs, [self.nbar(z, npts=50) for z in zs], bounds_error=False, fill_value=0.)
		self.nbar_spline = interpolate.interp1d(zs, [self.nbar(z,  npts=50) for z in zs], bounds_error=False, fill_value="extrapolate") #GF add extrapolation option
		
		self.dn_dz_spline= interpolate.interp1d(self.zgal, self.ngal)
		
	def u_g_ell(self, ell, M, z, npts=100):
		'''
		Fourier transform of the galaxy profile
		output: u_g_ell
		'''
		R200 = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/self.cosmo.rho_c(z))**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
		c200 = c_Duffy(M,z,self.cosmo.h)
		R_s  = R200/c200 # [Mpc]
		ells = self.cosmo.d_A(z)/R_s

		xarr = np.logspace(-5,1,100)

		delta_c = (self.mass_def/3.)*(c200**3.)/(np.log(1.+c200)-c200/(1.+c200))

		norm = delta_c * self.cosmo.rho_c(z) #[kg/m^3]
		nfw = lambda x:  1. / ( x*(1+x)**2. ) # [unitless]
		density_profile = nfw(xarr) * norm # [kg/m^3]

		# norm = delta_c * self.cosmo.rho_c(z) * R_s**3. #[kg/m^3 Mpc^3]
		# nfw = lambda r:  1. / ( r * (R_s + r )**2. ) # [1/Mpc^3]
		# density_profile = nfw(xarr/c200) * norm # [kg/m^3]
		arg = ((ell+0.5)*xarr/ells)
		uell = integrate.simps(xarr**2 * np.sin(arg)/arg * density_profile, xarr) # [kg/m^3]
		uell *= 4*np.pi*(R_s) / ells**2 # [kg/m^3 * Mpc]
		uell *= u.Mpc.to(u.m) # [kg/m^2]

		return uell
	
	
	def u_s_k(self, k, M, z, npts=100):
		'''
		Fourier transform of the galaxy profile hosting central and satellite galaxies
		output: u_g_ell 
		'''
		R200 = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/self.cosmo.rho_c(z))**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
		c200 = c_Duffy(M,z,self.cosmo.h)
		#R_s  = R200/c200 # [Mpc]
		
		q=k*R200/c200
		cp=1+c200
		
		us_k= 1.0/(np.log(cp)-c200/cp)
		us_k*=(np.cos(q)*(Ci(cp*q)-Ci(q)) + np.sin(q)*(Si(cp*q)-Si(q)) - np.sin(c200*q)/(cp*q))
		return us_k
		

	def N_M_simple(self, M, M_min=1e11, alpha=0.8, amp=0.7, M_pivot=4.2e12):
		if np.isscalar(M):
			if M < M_min:
				N_Ms = 0.
			elif M >= M_pivot:
				N_Ms = amp * (M/M_pivot)**alpha
			else:
				N_Ms = amp
		else:
			below_M_min = np.where(M < M_min)[0]
			between_M = np.where((M >= M_min) & (M < M_pivot))[0]
			above_M_pivot = np.where(M >= M_pivot)[0]

			N_Ms = np.zeros(len(M))
			N_Ms[below_M_min] = 0.
			N_Ms[between_M] = amp
			N_Ms[above_M_pivot] = amp * (M[above_M_pivot]/M_pivot)**alpha

		return N_Ms


	def f_cen(self, M):	
		res=0.5*(1+erf((np.log10(M)-np.log10(M_min_default))/sigma_logm_default))
		return res

	def N_sat(self, M):
		
		if np.isscalar(M):
			if M <= M_cut_default:
				res=0.0
			else:
				res= ((M-M_cut_default)/M1_default)**alpha_g_default

		else:
			below_M_cut = np.where(M < M_cut_default)[0]
			above_M_cut = np.where(M >= M_cut_default)[0]

			res = np.zeros(len(M))
			res[below_M_cut] = 0.
			res[above_M_cut]= ((M[above_M_cut]-M_cut_default)/M1_default)**alpha_g_default

		return res


	def N_M_hod(self, M):
		return  self.f_cen(M)*(1 + (self.N_sat(M)))

	def N_M(self, M):
		if (self.gal_dist_name=="HOD"):
			return self.N_M_hod(M)
		if (self.gal_dist_name=="simple"):
			return self.N_M_simple(M)

	def nbar(self, z, npts=50):
		Ms = np.logspace(np.log10(self.Mmin), np.log10(self.Mmax), npts)
		return integrate.simps(self.hmf.dndm(Ms,z) * self.N_M(Ms), x=Ms)


	def u_ell_2h(self, ell, M, z, npts=100):
		wg=self.cosmo.H_z(z)/const.c.value
		wg*=(u.km).to(u.Mpc)/(u.m).to(u.Mpc) #change the units
		wg *=self.dn_dz_spline(z)
		
		da=self.cosmo.d_A(z)
		k = ell/da
		u_sat=self.u_s_k(k, M, z)
		one_over_nbar_spline_z = 0 if self.nbar_spline(z)<=0 else self.nbar_spline(z)
		res= wg *  self.N_M(M) * u_sat * one_over_nbar_spline_z / da**2
		return res
		
		
	def u_ell_1h(self, ell, M, z, npts=100):
		wg=self.cosmo.H_z(z)/const.c.value
		wg*=(u.km).to(u.Mpc)/(u.m).to(u.Mpc)
		wg *=self.dn_dz_spline(z)
		da=self.cosmo.d_A(z)
		k = ell/da
		
		u_sat=self.u_s_k(k, M, z)
		f_g=  2 * self.f_cen(M) * self.N_sat(M)* u_sat + self.f_cen(M)**2 * self.N_sat(M)**2 * u_sat**2
		one_over_nbar_spline_z = 0 if self.nbar_spline(z)<=0 else self.nbar_spline(z)
		res= wg * np.sqrt(f_g) * one_over_nbar_spline_z/ da**2 #/ self.nbar_spline(z)
		return res
		
		
'''
class ClusterDensity():

	def __init__(self, cosmo, hmf=None, mass_def=200.):

		self.epsrel = 1.49e-6
		self.epsabs = 0.
		self.cosmo = cosmo
		self.mass_def = mass_def
		if hmf is None:
			self.hmf = MassFunction(self.cosmo)
		else:
			self.hmf = hmf

		zs = np.logspace(-3,1,1000)
		self.nbar_spline = interpolate.interp1d(zs, [self.nbar(z, npts=50) for z in zs], bounds_error=False, fill_value=0.)
		#self.nbar_spline = interpolate.interp1d(zs, [self.nbar(z, npts=50) for z in zs], bounds_error=False, fill_value="extrapolate") #GF add extrapolation option

	def u_g_ell(self, ell, M, z, npts=100):
		"""
		Fourier transform of the galaxy profile
		output: u_g_ell
		"""
		R200 = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/self.cosmo.rho_c(z))**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
		c200 = c_Duffy(M,z,self.cosmo.h)
		R_s  = R200/c200 # [Mpc]
		ells = self.cosmo.d_A(z)/R_s

		xarr = np.logspace(-5,1,npts)

		delta_c = (self.mass_def/3.)*(c200**3.)/(np.log(1.+c200)-c200/(1.+c200))

		norm = delta_c * self.cosmo.rho_c(z) #[kg/m^3]
		nfw = lambda x:  1. / ( x*(1+x)**2. ) # [unitless]
		density_profile = nfw(xarr) * norm # [kg/m^3]

		# norm = delta_c * self.cosmo.rho_c(z) * R_s**3. #[kg/m^3 Mpc^3]
		# nfw = lambda r:  1. / ( r * (R_s + r )**2. ) # [1/Mpc^3]
		# density_profile = nfw(xarr/c200) * norm # [kg/m^3]
		arg = ((ell+0.5)*xarr/ells)
		uell = integrate.simps(xarr**2 * np.sin(arg)/arg * density_profile, xarr) # [kg/m^3]
		uell *= 4*np.pi*(R_s) / ells**2 # [kg/m^3 * Mpc]
		uell *= u.Mpc.to(u.m) # [kg/m^2]

		return uell

	def N_M(self, M, M_min=1e11, alpha=0.8, amp=0.7, M_pivot=4.2e12):
		if np.isscalar(M):
			if M < M_min:
				N_Ms = 0.
			elif M >= M_pivot:
				N_Ms = amp * (M/M_pivot)**alpha
			else:
				N_Ms = amp
		else:
			below_M_min = np.where(M < M_min)[0]
			between_M = np.where((M >= M_min) & (M < M_pivot))[0]
			above_M_pivot = np.where(M >= M_pivot)[0]

			N_Ms = np.zeros(len(M))
			N_Ms[below_M_min] = 0.
			N_Ms[between_M] = amp
			N_Ms[above_M_pivot] = amp * (M[above_M_pivot]/M_pivot)**alpha

		return N_Ms

	def nbar(self, z, M_min=1e11, M_max=5e15, npts=50):
		Ms = np.logspace(np.log10(M_min), np.log10(M_max), npts)
		return integrate.simps(self.hmf.dndm(Ms,z) * self.N_M(Ms), x=Ms)

	def u_ell(self, ell, M, z, npts=100):
		one_over_nbar_spline_z = 0 if self.nbar_spline(z)<=0 else self.nbar_spline(z) #GF correction
		#return self.u_g_ell(ell,M,z,npts=npts) * self.N_M(M) / self.nbar_spline(z)
		return self.u_g_ell(ell,M,z,npts=npts) * self.N_M(M) * one_over_nbar_spline_z #GF  correction



class ClusterDensity4():

	def __init__(self, cosmo, hmf=None, gal_dist_name="HOD", mass_def=200.):
	
		self.epsrel = 1.49e-6
		self.epsabs = 0.
		self.cosmo = cosmo
		self.mass_def = mass_def
		self.gal_dist_name=gal_dist_name
		if hmf is None:
			self.hmf = MassFunction(self.cosmo)
		else:
			self.hmf = hmf

		zs = np.logspace(-3,1,1000)
		self.nbar_spline = interpolate.interp1d(zs, [self.nbar(z, npts=50) for z in zs], bounds_error=False, fill_value=0.)
		#self.nbar_spline = interpolate.interp1d(zs, [self.nbar(z,  npts=50) for z in zs], bounds_error=False, fill_value="extrapolate") #GF add extrapolation option

	def u_g_ell(self, ell, M, z, npts=100):
		"""
		Fourier transform of the galaxy profile
		output: u_g_ell
		"""
		R200 = (((M*(u.Msun).to(u.kg)/(self.mass_def*4.*np.pi/3.))/self.cosmo.rho_c(z))**(1./3.))*(u.m).to(u.Mpc) # [Mpc]
		c200 = c_Duffy(M,z,self.cosmo.h)
		R_s  = R200/c200 # [Mpc]
		ells = self.cosmo.d_A(z)/R_s

		xarr = np.logspace(-5,1,100)

		delta_c = (self.mass_def/3.)*(c200**3.)/(np.log(1.+c200)-c200/(1.+c200))

		norm = delta_c * self.cosmo.rho_c(z) #[kg/m^3]
		nfw = lambda x:  1. / ( x*(1+x)**2. ) # [unitless]
		density_profile = nfw(xarr) * norm # [kg/m^3]

		# norm = delta_c * self.cosmo.rho_c(z) * R_s**3. #[kg/m^3 Mpc^3]
		# nfw = lambda r:  1. / ( r * (R_s + r )**2. ) # [1/Mpc^3]
		# density_profile = nfw(xarr/c200) * norm # [kg/m^3]
		arg = ((ell+0.5)*xarr/ells)
		uell = integrate.simps(xarr**2 * np.sin(arg)/arg * density_profile, xarr) # [kg/m^3]
		uell *= 4*np.pi*(R_s) / ells**2 # [kg/m^3 * Mpc]
		uell *= u.Mpc.to(u.m) # [kg/m^2]

		return uell

	def N_M_simple(self, M, M_min=1e11, alpha=0.8, amp=0.7, M_pivot=4.2e12):
		if np.isscalar(M):
			if M < M_min:
				N_Ms = 0.
			elif M >= M_pivot:
				N_Ms = amp * (M/M_pivot)**alpha
			else:
				N_Ms = amp
		else:
			below_M_min = np.where(M < M_min)[0]
			between_M = np.where((M >= M_min) & (M < M_pivot))[0]
			above_M_pivot = np.where(M >= M_pivot)[0]

			N_Ms = np.zeros(len(M))
			N_Ms[below_M_min] = 0.
			N_Ms[between_M] = amp
			N_Ms[above_M_pivot] = amp * (M[above_M_pivot]/M_pivot)**alpha

		return N_Ms


	def N_cen(self, M, M_min=M_min_default, sigma_logm=sigma_logm_default):	
		res=0.5*(1+erf((np.log10(M)-np.log10(M_min))/sigma_logm))
		return res

	def N_sat(self, M, M_cut=M_cut_default, M1=M1_default, alpha_g=alpha_g_default):
		
		if np.isscalar(M):
			if M <= M_cut:
				res=0.0
			else:
				res= ((M-M_cut)/M1)**alpha_g

		else:
			below_M_cut = np.where(M < M_cut)[0]
			above_M_cut = np.where(M >= M_cut)[0]

			res = np.zeros(len(M))
			res[below_M_cut] = 0.
			res[above_M_cut]= self.N_cen(M[above_M_cut])*((M[above_M_cut]-M_cut)/M1)**alpha_g

		return res


	def N_M_hod(self, M ):
		 Nsat = self.N_sat(M)
		 Ncen = self.N_cen(M)
		 return Ncen + Ncen * Nsat
		 
	def N_M(self, M):
		if (self.gal_dist_name=="HOD"):
			return self.N_M_hod(M)
		if (self.gal_dist_name=="simple"):
			return self.N_M_simple(M)

		
	def nbar(self, z, M_min=1e11, M_max=5e15, npts=50):
		Ms = np.logspace(np.log10(M_min), np.log10(M_max), npts)
		return integrate.simps(self.hmf.dndm(Ms,z) * self.N_M(Ms), x=Ms)

	def u_ell(self, ell, M, z, npts=100):
		one_over_nbar_spline_z = 0 if self.nbar_spline(z)<=0 else self.nbar_spline(z) #GF correction
		#return self.u_g_ell(ell,M,z,npts=npts) * self.N_M(M) / self.nbar_spline(z)
		return self.u_g_ell(ell,M,z,npts=npts) * self.N_M(M) * one_over_nbar_spline_z #GF  correction


'''
