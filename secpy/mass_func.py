import numpy as np
from scipy import interpolate
from scipy import integrate
from secpy.default_params import *
from IPython import embed
import scipy.special as special


class MassFunction(object):
    """
    Object representing an halo mass function (HMF) for a given input cosmology.

    A MassFunction object can return several quantities such as:
    - properly normalized halo abundance, dn/dM(M,z)
    - halo bias as a function of halo mass or as a function of nu, b(M,z) or b(\nu,z)
    - ...

    Currently the HMF implemented is from Tinker+10 (see arXiv:1001.3162)

    Attributes
    ----------
    cosmo : Cosmo object (from universe.py)
        Cosmology object

    delta_v : float
        Virial overdensity

    nz : int
        Number of points to interpolate functions
    """
    def __init__(self, cosmo, delta_v=200, nz=50, verbose=False, do_interpolation=True, **kws):
        self.cosmo = cosmo
        self.delta_c = self.cosmo.delta_c()
        self.nz = nz

        # if halo_dict is None:
        #     halo_dict = default_halo_dict
        # self.halo_dict = halo_dict

        self.delta_v = delta_v

        delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
        if not do_interpolation: # use only value of Tinker mass function without interpolation
            if self.delta_v in delta_virs: # could be commented
                index = np.where(delta_virs == self.delta_v)[0]
            else:
                raise ValueError("delta_halo = %s is invalid; must be one of %s" %(self.delta_v, delta_virs))

        alpha_array = np.array([ 0.368,
                                 0.363,
                                 0.385,
                                 0.389,
                                 0.393,
                                 0.365,
                                 0.379,
                                 0.355,
                                 0.327])

        beta_array = np.array([ 0.589,
                                0.585,
                                0.544,
                                0.543,
                                0.564,
                                0.623,
                                0.637,
                                0.673,
                                0.702])

        gamma_array = np.array([ 0.864,
                                 0.922,
                                 0.987,
                                 1.09,
                                 1.20,
                                 1.34,
                                 1.50,
                                 1.68,
                                 1.81])

        phi_array = np.array([ -0.729,
                               -0.789,
                               -0.910,
                               -1.050,
                               -1.200,
                               -1.260,
                               -1.450,
                               -1.500,
                               -1.490])

        eta_array = np.array([ -0.243,
                               -0.261,
                               -0.261,
                               -0.273,
                               -0.278,
                               -0.301,
                               -0.301,
                               -0.319,
                               -0.336 ])

        # interpolate between the delta_virs to get the correct value
        self.alpha_func = interpolate.InterpolatedUnivariateSpline(delta_virs, alpha_array)
        self.beta_func  = interpolate.InterpolatedUnivariateSpline(delta_virs, beta_array)
        self.gamma_func = interpolate.InterpolatedUnivariateSpline(delta_virs, gamma_array)
        self.phi_func   = interpolate.InterpolatedUnivariateSpline(delta_virs, phi_array)
        self.eta_func   = interpolate.InterpolatedUnivariateSpline(delta_virs, eta_array)

        self.alpha_0 = self.alpha_func(self.delta_v)
        self.beta_0  = self.beta_func(self.delta_v)
        self.gamma_0 = self.gamma_func(self.delta_v)
        self.phi_0   = self.phi_func(self.delta_v)
        self.eta_0   = self.eta_func(self.delta_v)

        self._set_mass_and_z_limits()
        if verbose:
            print('\tmass and z limits set')
        self._initialize_splines()
        if verbose:
            print('\tspline initialized')
          
         ## fits for halo assembly bias
		self.m1 = np.load("../fits/p_m1_alpha.npz")
		self.s1 = np.load("../fits/p_S1_alpha.npz")
		self.haloprop = {}
		self.haloprop['c200b'] = np.load("../fits/")
		self.haloprop['spin_peebles'] = np.load("../fits/")
		self.haloprop['velocity_anisotropy'] = np.load("../fits/")
		self.haloprop['velocity_asphericity'] = np.load("../fits/")
		self.haloprop['shape_asphericity'] = np.load("../fits/")
        # self._normalize()
        # print('\tnormalizations evaluated')

    def _set_mass_and_z_limits(self):
        # Mass limits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mass_min = 1.0e8 # To match Tinker's code
        mass_max = 10**17.7 # To match Tinker's code
        if (default_limits["mass_min"] > 0 and default_limits["mass_max"] > 0):
            self.ln_mass_min = np.log(default_limits["mass_min"])
            self.ln_mass_max = np.log(default_limits["mass_max"])
            self._ln_mass_array = np.linspace( self.ln_mass_min, self.ln_mass_max, default_precision["mass_npoints"])
            return None

        # mass_limit_not_set = True
        # while mass_limit_not_set:
        #     if 0.1*(1.0+0.05) < self.cosmo.nu_M(mass_min, self.z)**2:
        #         #print "Min mass", mass_min,"too high..."
        #         mass_min = mass_min/1.05
        #         #print "\tSetting to",mass_min,"..."
        #         continue
        #     elif 0.1*(1.0-0.05) > self.cosmo.nu_M(mass_min, self.z)**2:
        #         #print "Min mass", mass_min,"too low..."
        #         mass_min = mass_min*1.05
        #         #print "\tSetting to",mass_min,"..."
        #         continue
        #     if  50.0*(1.0-0.05) > self.cosmo.nu_M(mass_max, self.z)**2:
        #         #print "Max mass", mass_max,"too low..."
        #         mass_max = mass_max*1.05
        #         #print "\tSetting to",mass_max,"..."
        #         continue
        #     elif 50.0*(1.0+0.05) < self.cosmo.nu_M(mass_max, self.z)**2:
        #         #print "Max mass", mass_max,"too high..."
        #         mass_max = mass_max/1.05
        #         #print "\tSetting to",mass_max,"..."
        #         continue
        #     mass_limit_not_set = False

        # print "Mass Limits: [%e, %e]" %(mass_min*(0.95),mass_max*(1.05))

        self.ln_mass_min = np.log(mass_min)
        self.ln_mass_max = np.log(mass_max)

        self._ln_mass_array = np.linspace( self.ln_mass_min, self.ln_mass_max, default_precision["mass_npoints"])

        # Redshift limits !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.z_min = 0. # HARDCODED
        self.z_max = 3. # HARDCODED
        self._z_array = np.linspace( self.z_min, self.z_max, self.nz)

        # \nu limits # HARDCODED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.nu_min = 1e-8 # FIXME: to integrate over all masses and get the right normalization
        self.nu_max = 25.

    def _initialize_splines(self):
        # Nu spline
        self._nu_array = np.zeros((len(self._ln_mass_array),len(self._z_array)))
        for idz, z in enumerate(self._z_array):
            for idx in range(self._ln_mass_array.size):
                mass = np.exp(self._ln_mass_array[idx])
                self._nu_array[idx,idz] = self.cosmo.nu_M(mass, z=z)

        # \nu(M,z) - logNU <=> logM at a given z
        self._nu_spline = interpolate.RectBivariateSpline(self._ln_mass_array, self._z_array, self._nu_array)

        # # M(\nu,z) - logM <=> logNU at a given z
        # self._ln_mass_spline = interpolate.RectBivariateSpline(self._nu_array, self._ln_mass_array)

        # HMF normalization spline
        self._norm = np.zeros_like(self._z_array)
        for idz, z in enumerate(self._z_array):
            self._norm[idz] = self._normalize(z)

        self._norm_spline = interpolate.InterpolatedUnivariateSpline(self._z_array, self._norm)

    def _normalize(self, z):
        norm = integrate.quad(lambda x: self.f_nu_unnorm(x,z)*self.bias_nu(x),
                              self.nu_min, self.nu_max, epsabs=0.0, epsrel=default_precision["mass_precision"], limit=100)[0]
        return 1./norm

    def f_nu_unnorm(self, nu, z):
        """
        Un-normalized halo mass function as a function of normalized mass over-density nu and redshift z.

        Parameters
        ----------
        nu: float
            Array normalized mass over-density nu

        Returns
        -------
        fnu: float array
            Number of halos
        """
        # Tinker+10 HMF is valid up to z = 3
        if z > 3:
            z = 3

        beta  = self.beta_0  * (1+z)**0.20
        phi   = self.phi_0   * (1+z)**(-0.08)
        eta   = self.eta_0   * (1+z)**0.27
        gamma = self.gamma_0 * (1+z)**(-0.01)

        fnu = (1 + (beta*nu)**(-2*phi)) * nu**(2*eta) * np.exp(-gamma*(nu**2)/2)
        # fnu = self.alpha_0 * (1 + (beta*nu)**(-2*phi)) * nu**(2*eta) * np.exp(-gamma*(nu**2)/2)

        return fnu

    def f_nu(self, nu, z):
        """
        Halo mass function as a function of normalized mass over-density nu and redshift z.

        Parameters
        ----------
        nu: float array
            Normalized mass over-density nu

        Returns
        -------
        fnu: float array
            Number of halos
        """
        return self._norm_spline(z) * self.f_nu_unnorm(nu, z)

    def f_sigma(self, M, z):
        """
        """
        return self.f_nu(self.nu(M,z),z) * self.nu(M,z)

    def f_m(self, mass, z):
        """
        Halo mass function as a function of halo mass

        Parameters
        ----------
        mass: float array
            Array halo mass [M_sun]

        z: float array
            Redshift

        Returns
        -------
        fnu: float array
            Number of halos
        """
        return self.f_nu(self.nu(mass, z), z)

    def dndm(self, M, z):
        """
        Convenience function for computing the number of halos per mass at a given redshift.

        Parameters
        ----------
        mass: float value or array
            Halo mass [M_sun]

        z : float
            Redshift

        Returns
        -------
        dndm : float or array
            Number of halos per mass per (Mpc)^3

        """
        try:
            _dndm_ = np.empty(len(M))
            for idx, m in enumerate(M):
                _dndm_[idx] = (self.cosmo.rho_bar()/(m*m) * self.f_m(m,z)* self._nu_spline.ev(np.log(m),z,dx=1))
            return _dndm_
        except TypeError:
            return (self.cosmo.rho_bar()/(M*M) * self.f_m(M,z) * self._nu_spline.ev(np.log(M),z,dx=1))

    def dndlnm(self, M, z):
        """
        The differential mass function in terms of natural log of `m`
        """
        return M * self.dndm(M, z)

    def dndmdz(self, M, z):
        """
        Total number of dark matter haloes at a given redshift.
        Product of dndm * dVdz
        """
        return self.dndm(M, z) * self.cosmo.dVdz(z)

    def dndz(self, z, M_min=None, M_max=None):
        """
        """
        if np.isscalar(z) or (np.size(z) == 1):
            return self.n_cl(z=z, M_min=M_min, M_max=M_max) * self.cosmo.dVdz(z)
        else:
            return np.asarray([ self.dndz(tz, M_min=M_min, M_max=M_max) for tz in z ])

    def dndlmdz(self, M, z):
        """
        Total number of dark matter haloes at a given redshift.
        Product of dndm * dVdz
        """
        return self.dndlnm(M, z) * self.cosmo.dVdz(z)

    def bias_M(self, M, z):
        """
        Halo bias as a function of mass.

        Parameters
        ----------
        mass: float array
            Array halo mass [M_sun]

        z: float array
            Redshift

        Returns
        -------
        b : float array
            Halo bias
        """
        return self.bias_nu(self.nu(M,z))

    def bias_nu(self, nu, delta_v=200.):
        """
        Halo bias as a function of normalized mass overdensity \nu (Eq.6 from 1001.3162)

        Parameters
        ----------
        nu: float array
            Array of normalized mass overdensity \nu

        Returns
        -------
        b : float array
            Halo bias
        """
        y = np.log10(delta_v)
        A = 1.0 + 0.24*y*np.exp(-(4./y)**4.)
        a = 0.44*y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4.)
        c = 2.4
        # return self.bias_norm * (1. - A*nu**a/(nu**a + self.delta_c**a) + B*nu**b + C*nu**c)
        return (1. - A*nu**a/(nu**a + self.delta_c**a) + B*nu**b + C*nu**c)

    def nu(self, mass, z):
        """
        Normalized mass overdensity as a function of halo mass and redshift

        Parameters
        ----------
        mass : float or array
            halo mass [M_sun]

        Returns
        --------
        nu : float array
            Normalized mass overdensity
        """
        return self._nu_spline.ev(np.log(mass), z)

    # def ln_mass(self, nu):
    #     """
    #     Natural log of halo mass as a function of nu.

    #     Args:
    #         nu: float array normalized mass over-density
    #     Returns:
    #         float array natural log mass [M_sun]
    #     """
    #     return self._ln_mass_spline(nu)

    # def mass(self, nu):
    #     """
    #     Halo mass as a function of nu.

    #     Args:
    #         nu: float array normalized mass over-density
    #     Returns:
    #         float array halo mass [M_sun]
    #     """
    #     return np.exp(self.ln_mass(nu))

    def bias_avg(self, z=0., M_min=None, M_max=None):
        """
        Mass averaged halo bias at a given redshift

        Args
        ----
        z: float or array
            Redshift

        M_min: float
            Lower limit of mass integration [M_sun]

        M_mass: float
            Upper limit of mass integration [M_sun]

        Returns
        -------
        b_avg : float or array
            Mass averaged halo bias

        """
        if M_min is None:
            M_min = np.exp(self.ln_mass_min)
        else:
            assert(M_min >= np.exp(self.ln_mass_min))
        if M_max is None:
            M_max = np.exp(self.ln_mass_max)
        else:
            assert(M_max <= np.exp(self.ln_mass_max))

        def norm_int(M):
            return M * self.dndm(M,z)

        def avg_bias_int(M):
            return M * self.dndm(M,z) * self.bias_M(M,z)

        norm = integrate.quad(norm_int, M_min, M_max, epsabs=0.0, epsrel=1e-5, limit=100)[0]
        bias = integrate.quad(avg_bias_int, M_min, M_max, epsabs=0.0, epsrel=1e-5, limit=100)[0]

        return bias / norm

    def n_cl(self, z=0., M_min=None, M_max=None):
        """
        Total halo densities within a given mass bin at fixed redshift.

        Args
        ----
        z: float or array
            Redshift

        M_min: float
            Lower limit of mass integration [M_sun]

        M_mass: float
            Upper limit of mass integration [M_sun]

        Returns
        -------
        n_cl : float or array
            Halo density [Mpc^(-3)}]
        """
        if M_min is None:
            M_min = np.exp(self.ln_mass_min)
        else:
            assert(M_min >= np.exp(self.ln_mass_min))
        if M_max is None:
            M_max = np.exp(self.ln_mass_max)
        else:
            assert(M_max <= np.exp(self.ln_mass_max))

        return integrate.quad(self.dndm, M_min, M_max, args=(z), epsabs=0.0, epsrel=1e-5, limit=200)[0]

    def N_cl(self, z_min, z_max, M_min=None, M_max=None):
        """
        Total number of dark matter haloes expected within a given mass and redshift bin.
        """
        if M_min is None:
            M_min = np.exp(self.ln_mass_min)
        else:
            assert(M_min >= np.exp(self.ln_mass_min))
        if M_max is None:
            M_max = np.exp(self.ln_mass_max)
        else:
            assert(M_max <= np.exp(self.ln_mass_max))

        return integrate.dblquad(self.dndmdz, z_min, z_max, lambda m: M_min, lambda m: M_max)[0]
        
	def secondary_bias(self,secondaryproperty,m,z,fromval,toval=None):
		"""
		secondaryproperty = 'tidal_anisotropy' , 'c200b' , 'spin_peebles' , 'velocity_anisotropy', 'velocity_asphericity','shape_asphericity'
		m = mass if z is a number or peakheight if z is "NA"
		z  = redshift or 'NA'
		fromval = value of the secondary property 
		toval = default value is None ,  otherwise we are intrested to obtain average secondary bias for a range of secondary property from fromval to toval
		"""
		h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)
		if np.exp(-toval**2/2)==0.0:
			h2avg = (fromval*np.exp(-fromval**2/2))/np.sqrt(2*np.pi)
		elif np.exp(-fromval**2/2)==0.0:
			h2avg = (-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)
		else:
			h2avg = (fromval*np.exp(-fromval**2/2)-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)
		_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		if z=='NA':
			v = m
		else:
			v = self.PeakHeight(m,z)
		mu1 = self.fit(v,self.m1)
		s1 = self.fit(v,self.s1)
		# ~ rhofit = self.ro_c200balpha['name1']
		rho = self.correlation_with_tidalenv(secondaryproperty,v)
		b1avg = self.bias_nu(v ,delta_v=200.) + rho*mu1*h1avg/_avg + 1/2.*rho**2*s1*h2avg/_avg
		##################### for generating error bar  ################################################################
		sampling=100
		rhosamp = self.correlation_with_tidalenv(secondaryproperty,v,sample_cov=1,sampling=sampling)
		mu1samp = self.fit(v,self.m1,sample_cov=1,sampling=sampling)
		s1samp = self.fit(v,self.s1,sample_cov=1,sampling=sampling)
		b1_fr_err= rhosamp*mu1samp*h1avg/_avg + 1/2.*rhosamp**2*s1samp*h2avg/_avg
		err_in_b1 = np.std(b1_fr_err,axis=0)
		return b1avg,err_in_b1 	

	def correlation_with_tidalenv(self,secondaryproperty,v,sample_cov=0,sampling=0):
		if sample_cov==0:
			rhofit = self.haloprop[secondaryproperty]['name1']
			rho = np.polyval(rhofit[::-1],np.log(v))
		elif sample_cov==1:
			rhofit = np.random.multivariate_normal(self.haloprop[secondaryproperty]['name1'],self.haloprop[secondaryproperty]['name2'],sampling)
			rho= 0	
			for i in range(len(self.haloprop[secondaryproperty]['name1'])):
				rho +=np.log(v.reshape([1,len(v)]))**i*(rhofit[:,i]).reshape([len(rhofit[:,i]),1])
		return rho
		
	def fit(self,v,p,sample_cov=0,sampling=0):
		"""
		Note that g(z) has already been divided out
		"""
		if sample_cov==0:
			fitval = (np.log10(v/1.5)**2*p['name1'][0] + np.log10(v/1.5)*p['name1'][1] + p['name1'][2]).flatten()
		elif sample_cov==1:
			fits = np.random.multivariate_normal(p['name1'],p['name2'],sampling)
			fitval= 0	
			v = v.reshape([1,len(v)])
			fits = fits.reshape([sampling,3,1])
			fitval = (np.log10(v/1.5)**2*fits[:,0,:] + np.log10(v/1.5)*fits[:,1,:] + fits[:,2,:])
		return fitval
		
    # def N_cl_log(self, z_min, z_max, M_min=None, M_max=None):
    #     """
    #     Total number of dark matter haloes expected within a given mass, redshift bin.
    #     """
    #     if M_min is None:
    #         lnM_min = self.ln_mass_min
    #     else:
    #         assert(np.log(M_min) >= self.ln_mass_min)
    #         lnM_min = np.log(M_min)
    #     if M_max is None:
    #         lnM_max = self.ln_mass_max
    #         M_max = np.exp(self.ln_mass_max)
    #     else:
    #         assert(np.log(M_max) <= self.ln_mass_max)
    #         lnM_max = np.log(M_max)

    #     return integrate.dblquad(self.dndlmdz, z_min, z_max, lambda m: lnM_min, lambda m: lnM_max)[0]
