import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import (
    CartesianRepresentation,
    CylindricalRepresentation,
    UnitSphericalRepresentation,
    EarthLocation,
    SkyCoord,
    SkyOffsetFrame,
    get_body_barycentric_posvel
)

from astropy.time import Time
from screenpkg.screens.screen import Source, Screen1D, Telescope
from screenpkg.screens.fields import phasor
from screenpkg.screens.visualization import axis_extent

def get_vearth(psrname, mjds):
    
    """
    Function to get the Earth velocity vector in RA, DEC coords for a given name pulsar as a string,
    and MJD as a float. Function returns 2-element vector in km/s units of the form
    Velocity in [RA, DEC] coords.
    
    """
    
    pulsar = SkyCoord.from_name(psrname)
    
    rarad = pulsar.ra.value * np.pi/180
    decrad = pulsar.dec.value * np.pi/180
    
    vtel_ra = np.zeros_like(mjds)
    vtel_dec = np.zeros_like(mjds)
    
    time = Time(mjds, format='mjd')
    pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
    
    vx = vel_xyz.x.to(u.m/u.s).value
    vy = vel_xyz.y.to(u.m/u.s).value
    vz = vel_xyz.z.to(u.m/u.s).value
    
    vtel_ra = - vx * np.sin(rarad) + vy * np.cos(rarad)
    vtel_dec = - vx * np.sin(decrad) * np.cos(rarad) - vy * np.sin(decrad) * np.sin(rarad) + vz * np.cos(decrad)
    
    vtel_vec = np.array([vtel_ra,vtel_dec])
    
    return vtel_vec / 1e3 * u.km / u.s


def screen_1d_curvature(d_p, vpsr, dp_angle, vearth, e_angle, ds1, xi1, v1, freq, extra_info = None):
    
    """
    Function to get a curvature for a given frequency freq (float) given the parameters dp, vspr and dp_angle as the pulsar's
    distance, velocity and angle in the sky. The parameters of the earth are given from velocity and angle as vearth and e_angle. 
    Finally, the distance, orientation and velocity of the scintillation screen d_s1, xi1, and v1.
    """
    
    #get Deff
    deff1 = d_p * ds1  / (d_p - ds1 )
    
    #get Veff
    veff1 = (ds1  / (d_p - ds1 ) * vpsr  * ( np.cos(xi1 ) * np.sin(dp_angle ) + np.sin(xi1 ) * np.cos(dp_angle )) 
             - v1  * d_p / (d_p - ds1 )
             + vearth * ( np.cos(xi1 ) * np.sin(e_angle) + np.sin(xi1 ) * np.cos(e_angle))
            )
    
    #get curvature for given frequency
    eta1 = ( deff1 * const.c / 2 / (veff1)**2 / freq**2 ).to( u.s**3 )
    
    if extra_info :
        return eta1, deff1, veff1
    else:
        return eta1
    
    
    
def interaction_arcs(d_p, vpsr, dp_angle, vearth, e_angle, ds1, xi1, v1, ds2, xi2, v2, freq):
    
    """
    Function to calculate interaction arcs curvatures, effectice distance and velocity given initial two screen parameters
    """
    
    #some useful relations
    Vex = vearth * np.cos(e_angle) 
    Vey = vearth * np.sin(e_angle)
    vpx = vpsr * np.cos(dp_angle) 
    vpy = vpsr * np.sin(dp_angle) 
    e1 = -(xi1 -90.*u.deg)
    e2 = -(xi2 -90.*u.deg)
    vs1 = -v1
    vs2 = -v2
    s12 = 1 - ds1 / ds2
    s1p = 1 - ds1 / d_p
    d = e2 - e1
    
    
    #computing the parameters for CASE I interaction arc
    sig2 = np.sin(d) * (s12 - s1p) / (s1p + (s12 - s1p)*np.cos(d)**2 ) 
    sig1 = np.cos(d) * sig2

    deff_int_1 = ( 
                + ds1  
                + ds1 * sig1**2 
                + ds1**2 / (ds2 - ds1) 
                + (ds2 * sig2)**2 / (d_p - ds2)
                + 2 * ( ds1 * ds2 * sig2 * np.sin(d) ) / (ds2 - ds1)
                + ( (ds1 * sig1)**2 + (ds2 * sig2)**2 - 2 * ds1 * ds2 * sig1 * sig2 * np.cos(d)) / (ds2 - ds1) 
                 ).to(u.pc)
    
    veff_int_1 = ( 
              Vex * (-np.cos(e1 + np.arctan(sig1) )) * np.sqrt(1 + sig1**2)
            + Vey * (-np.sin(e1 + np.arctan(sig1) )) * np.sqrt(1 + sig1**2)
            + vpx * ds2 / (d_p - ds2) * (-np.cos(e2 + np.pi/2 * u.rad)) * sig2
            + vpy * ds2 / (d_p - ds2) * (-np.sin(e2 + np.pi/2 * u.rad)) * sig2
            + ( -ds1 * vs1 / (ds2 - ds1) 
               + ds1 * vs2 * np.cos(d) / (ds2 - ds1) 
               - vs1 + np.sin(d) / (ds2 - ds1) * (ds1 * sig1 * vs2 - ds2 * sig2 * vs1) )
           ).to(u.km / u.s)
    
    eta_int_1 = (deff_int_1 / 2 * const.c / veff_int_1**2 / freq**2).to(u.s**3)
    
    
    #computing the parameters for CASE II interaction arc
    gam2 = -np.sin(d) * np.cos(d) * (s12 - s1p) / (s1p + (s12 - s1p)*np.cos(d)**2 ) 
    gam1 = np.sin(d) * s1p / (s1p + (s12 - s1p) * np.cos(d)**2 ) 
    
    deff_int_2 = (
                ds1 * gam1**2 
                + ds2**2 / (d_p - ds2)
                + ds2**2 * gam2**2 / (d_p - ds2)
                + ds2**2 / (ds2 - ds1)
                - 2 * ds1 * ds2 * gam1 * np.sin(d) / (ds2 - ds1)
                + (ds1**2 * gam1**2 + ds2**2 * gam2**2 - 2 * ds1 * ds2 * gam1 * gam2 * np.cos(d)) / (ds2 - ds1)
                ).to(u.pc)

    veff_int_2 = ( 
              Vex * (-np.cos(e1 + np.pi/2 * u.rad )) * gam1
            + Vey * (-np.sin(e1 + np.pi/2 * u.rad )) * gam1
            + vpx * ds2 / (d_p - ds2) * (-np.cos(e2 + np.arctan(gam2) )) * np.sqrt( 1 + gam2**2)
            + vpy * ds2 / (d_p - ds2) * (-np.sin(e2 + np.arctan(gam2) )) * np.sqrt( 1 + gam2**2)
            + ( ds2 * vs1 * np.cos(d) / (ds2 - ds1)
               - ds2 * vs2 / (ds2 - ds1)
               - ds2 * vs2 / (d_p - ds2)
               + np.sin(d) * (ds1 * gam1 * vs2 - ds2 * gam2 * vs1) / (ds2 - ds1)
                )
           ).to(u.km / u.s)
    
    eta_int_2 = (deff_int_2 / 2 * const.c / veff_int_2**2 / freq**2).to(u.s**3)
    
    
    return eta_int_1, eta_int_2, deff_int_1, deff_int_2, veff_int_1, veff_int_2


def interaction_geometric_factors(d_p, vpsr, dp_angle, vearth, e_angle, ds1, xi1, v1, ds2, xi2, v2, freq):
    
    #some useful relations
    Vex = vearth * np.cos(e_angle) 
    Vey = vearth * np.sin(e_angle)
    vpx = vpsr * np.cos(dp_angle) 
    vpy = vpsr * np.sin(dp_angle) 
    e1 = -(xi1 -90.*u.deg)
    e2 = -(xi2 -90.*u.deg)
    vs1 = -v1
    vs2 = -v2
    s12 = 1 - ds1 / ds2
    s1p = 1 - ds1 / d_p
    d = e2 - e1
    
    
    #computing the parameters for CASE I interaction arc
    sig2 = np.sin(d) * (s12 - s1p) / (s1p + (s12 - s1p)*np.cos(d)**2 ) 
    sig1 = np.cos(d) * sig2

    deff_int_1 = ( 
                + ds1  
                + ds1 * sig1**2 
                + ds1**2 / (ds2 - ds1) 
                + (ds2 * sig2)**2 / (d_p - ds2)
                + 2 * ( ds1 * ds2 * sig2 * np.sin(d) ) / (ds2 - ds1)
                + ( (ds1 * sig1)**2 + (ds2 * sig2)**2 - 2 * ds1 * ds2 * sig1 * sig2 * np.cos(d)) / (ds2 - ds1) 
                 ).to(u.pc)
    
    veff_int_1 = ( 
              Vex * (-np.cos(e1 + np.arctan(sig1) )) * np.sqrt(1 + sig1**2)
            + Vey * (-np.sin(e1 + np.arctan(sig1) )) * np.sqrt(1 + sig1**2)
            + vpx * ds2 / (d_p - ds2) * (-np.cos(e2 + np.pi/2 * u.rad)) * sig2
            + vpy * ds2 / (d_p - ds2) * (-np.sin(e2 + np.pi/2 * u.rad)) * sig2
            + ( -ds1 * vs1 / (ds2 - ds1) 
               + ds1 * vs2 * np.cos(d) / (ds2 - ds1) 
               - vs1 + np.sin(d) / (ds2 - ds1) * (ds1 * sig1 * vs2 - ds2 * sig2 * vs1) )
           ).to(u.km / u.s)
    
    
    #computing the parameters for CASE II interaction arc
    gam2 = -np.sin(d) * np.cos(d) * (s12 - s1p) / (s1p + (s12 - s1p)*np.cos(d)**2 ) 
    gam1 = np.sin(d) * s1p / (s1p + (s12 - s1p) * np.cos(d)**2 ) 
    
    deff_int_2 = (
                ds1 * gam1**2 
                + ds2**2 / (d_p - ds2)
                + ds2**2 * gam2**2 / (d_p - ds2)
                + ds2**2 / (ds2 - ds1)
                - 2 * ds1 * ds2 * gam1 * np.sin(d) / (ds2 - ds1)
                + (ds1**2 * gam1**2 + ds2**2 * gam2**2 - 2 * ds1 * ds2 * gam1 * gam2 * np.cos(d)) / (ds2 - ds1)
                ).to(u.pc)

    veff_int_2 = ( 
              Vex * (-np.cos(e1 + np.pi/2 * u.rad )) * gam1
            + Vey * (-np.sin(e1 + np.pi/2 * u.rad )) * gam1
            + vpx * ds2 / (d_p - ds2) * (-np.cos(e2 + np.arctan(gam2) )) * np.sqrt( 1 + gam2**2)
            + vpy * ds2 / (d_p - ds2) * (-np.sin(e2 + np.arctan(gam2) )) * np.sqrt( 1 + gam2**2)
            + ( ds2 * vs1 * np.cos(d) / (ds2 - ds1)
               - ds2 * vs2 / (ds2 - ds1)
               - ds2 * vs2 / (d_p - ds2)
               + np.sin(d) * (ds1 * gam1 * vs2 - ds2 * gam2 * vs1) / (ds2 - ds1)
                )
           ).to(u.km / u.s)
    
    
    
    
    
    
    return deff_int_1, deff_int_2, veff_int_1, veff_int_2, sig1, sig2, gam1, gam2

    
def observations(d_s1, xi1, v1,
                 d_s2, xi2, v2,
                 d_p, vpsr, dp_angle,
                 vearth, e_angle,
                 N1, N2,
                 Amp1 , Amp2 ,
                 sig1 , sig2 ,
                 scr1_scale = 1., scr2_scale = 1.,
                 shift1 = 0., shift2 = 0. ):
    
    """
    Simulate a wavefield through two thin scattering screens.

    Parameters
    ----------
    d_s1 : `~astropy.units.Quantity`
        Distance from Earth to the first (nearest) scattering screen.
    xi1 : `~astropy.units.Quantity`
        Orientaiton angle of screen 1, measured East of North on the sky
    v1 : `~astropy.units.Quantity`
        Transverse velocity of screen 1.

    d_s2 : `~astropy.units.Quantity`
        Distance from Earth to the second (farther) scattering screen.
    xi2 : `~astropy.units.Quantity`
        Orientaiton angle of screen 2, measured East of North on the sky
    v2 : `~astropy.units.Quantity`
        Transverse velocity of screen 2.

    d_p : `~astropy.units.Quantity`
        Distance from Earth to the pulsar.

    vpsr : `~astropy.units.Quantity`
        Transverse velocity amplitude of the pulsar.
    dp_angle : `~astropy.units.Quantity`
        Direction angle of the pulsar velocity in the plane of the sky East of North.

    vearth : `~astropy.units.Quantity`
        Transverse velocity amplitude of the Earth.
    e_angle : `~astropy.units.Quantity`
        Direction angle of the Earth’s velocity in the plane of the sky East of North.

    N1, N2 : int
        Number of images used to uniformly sample the scattering screen 1 and 2.

    Amp1, Amp2 : float
        Amplitude scaling factors for the Gaussian magnification profile
        of screen 1 and 2, respectively.

    sig1, sig2 : float
        Gaussian widths (in units of AU) controlling the scattering strength
        of screen 1 and 2.

    scr1_scale, scr2_scale : float, optional
        Scaling factors applied to control the extent of screen 1 and 2 (in units of AU). 
        Default is 1.0.

    shift1, shift2 : float, optional
        Positional offsets applied to the screen coordinates (in units of AU).
        Useful for displacing the images of each screen relative to the
        line of sight. Default is 0.0.

    Returns
    -------
    obs0, obs1, obs2, obs12: screens 'obs' object
        Contains wavefield information about the geometry with respect to 
        0: pulsar to the Earth
        1: pulsar to screen 1 to the Earth
        2: pulsar to screen 2 to the Earth
        12: pulsar to screen 2 to screen 1 to the Earth

    sum_los : float
        Fractional power transmitted directly along the line of sight without
        encountering either screen (i.e., unscattered component).
        
    """
    
    p1 = ( scr1_scale * np.linspace(-1., 1., N1) + shift1 ) << u.AU
    #p1 = ( scr1_scale * np.linspace(-0.99, 0.99, N1) + shift1) << u.AU
    m1 = np.exp(-0.5*(p1/(sig1*u.AU))**2)
    m1 *= Amp1
    
    
    p2 = ( scr2_scale * np.linspace(-1., 1., N2) + shift2 ) << u.AU
    m2 = np.exp(-0.5*(p2/(sig2*u.AU))**2)
    m2 *= Amp2
    
    
    #power that will pass through scr2 and can reach scr1 (closest to the Earth)
    Sum2 = np.sqrt( 1. - np.sum( np.abs(m1)**2  ) )
    #power that will pass through the screen closest to the pulsar 
    #that is not scattered by the screen closest to the earth
    Sum1 = np.sqrt( 1. - np.sum( np.abs(m2)**2  ) )
    
    #los power that can make it to Earth without being scattered by scr1 and scr2
    sum_los = Sum1 * Sum2
    
    vel_psr = CylindricalRepresentation(vpsr, dp_angle, 0.*u.km/u.s).to_cartesian()
    vel_ear = CylindricalRepresentation(vearth, e_angle, 0.*u.km/u.s).to_cartesian()
    telescope = Telescope(vel = vel_ear)
    
    #source
    pulsar0 = Source(vel=vel_psr, magnification = np.array([1.]))
    
    normal1 = CylindricalRepresentation(1., 90.*u.deg - xi1, 0.).to_cartesian()
    screen1 = Screen1D(normal=normal1, p=p1, v=v1, magnification= m1 / Sum1)
    normal2 = CylindricalRepresentation(1., 90.*u.deg - xi2, 0.).to_cartesian()
    screen2 = Screen1D(normal=normal2, p=p2, v=v2, magnification= m2 / Sum2)
    
    #source
    obs0 = telescope.observe(source=pulsar0, distance = d_p)
    
    #source -> screen1 -> obs
    obs_scr1_pulsar = screen1.observe(source=pulsar0, distance=d_p-d_s1)
    obs1 = telescope.observe(source=obs_scr1_pulsar, distance=d_s1)
    
    #source -> screen2 -> obs
    obs_scr2_pulsar2 = screen2.observe(source=pulsar0, distance=d_p-d_s2)
    obs2 = telescope.observe(source=obs_scr2_pulsar2, distance=d_s2)
    
#     bool_on_lens2 = obs2.source.pos.x.ravel() < 7. * u.au

    #setting observation with screen2
    #source -> screen2 -> screen1 -> obs
    obs_scr2_pulsar = screen2.observe(source=pulsar0, distance=d_p-d_s2)
    obs_scr2_pulsar2 = screen1.observe(source=obs_scr2_pulsar, distance=d_s2-d_s1)
    obs12 = telescope.observe(source=obs_scr2_pulsar2, distance=d_s1)
    
    
    return obs0, obs1, obs2, obs12, sum_los



def observations_plot(d_s1, xi1, v1,
                 d_s2, xi2, v2,
                 d_p, vpsr, dp_angle,
                 vearth, e_angle,
                 p11, p22,
                 ):
    
    
    """
    Same function as observations(), except you give it a custom p11, and p22, and it also outputs a compatible set
    of obs for the 3D plotter script
    """
    
    t11 = 0.5 * u.one
    m11 = np.exp(-0.5*(p11/(0.5*u.AU))**2)
    m11 *= np.sqrt((1-t11**2) / np.sum(np.abs(m11)**2))

    t22 = 0.5 * u.one
    m22 = np.exp(-0.5*(p22/(0.5*u.AU))**2)
    m22 *= np.sqrt((1-t22**2) / np.sum(np.abs(m22)**2))
    
    t1122 = t11 * t22

    vel_psr = CylindricalRepresentation(vpsr, 
                                        dp_angle, 0.*u.km/u.s).to_cartesian()
    pulsar0 = Source(vel=vel_psr, magnification=t1122)
    pulsar1 = Source(vel=vel_psr, magnification=t22)
    pulsar2 = Source(vel=vel_psr, magnification=t11)
    pulsar12 = Source(vel=vel_psr)
    
    telescope = Telescope()
    
    normal1 = CylindricalRepresentation(1., 90 * u.deg - xi1, 0.).to_cartesian()
    screen1 = Screen1D(normal=normal1, p=p11, v=v1, magnification=m11)
    normal2 = CylindricalRepresentation(1., 90 * u.deg - xi2, 0.).to_cartesian()
    screen2 = Screen1D(normal=normal2, p=p22, v=v2, magnification=m22)

    obs0 = telescope.observe(source=pulsar0, distance=d_p)

    obs1 = telescope.observe(
        source=screen1.observe(source=pulsar1, distance=d_p-d_s1),
        distance=d_s1)

    obs2 = telescope.observe(
        source=screen2.observe(source=pulsar2, distance=d_p- d_s2),
        distance=d_s2)

    obs12 = telescope.observe(
        source=screen1.observe(
            source=screen2.observe(source=pulsar12, distance=d_p-d_s2 ),
            distance=d_s2 - d_s1),
        distance=d_s1)

    return obs0, obs1, obs2, obs12



def plot_screen(ax, s, d, pos_sel=(), color='black', mult = 1., point_sizes = 50, sl = 1., xy_lim = 1, **kwargs):
    
    d_unit = u.pc
    tau_unit = u.us
    taudot_unit = u.us/u.day
    
    ZHAT = CartesianRepresentation(0., 0., 1., unit=u.one)
    d = d.to_value(u.pc)
    x = np.array(ax.get_xlim3d())
    y = np.array(ax.get_ylim3d())[:, np.newaxis]
    ax.plot_surface([[-sl, sl]]*2, [[-sl]*2, [sl]*2], d*np.ones((2, 2)),
                    alpha=0.1, color=color)
    x = ax.get_xticks()
    y = ax.get_yticks()[:, np.newaxis]
    spos = (s.normal * s.p if isinstance(s, Screen1D) else s.pos)[pos_sel]
    ax.scatter(spos.x.to_value(u.AU), spos.y.to_value(u.AU), d,
               c=color, s = point_sizes)
    if spos.shape:
        zo = np.arange(2)
        for pos in spos:
            ax.plot(pos.x.to_value(u.AU)*zo * mult, pos.y.to_value(u.AU)*zo * mult,
                    np.ones(2) * d, c=color, linestyle=':')
            upos = pos + (ZHAT.cross(s.normal) * ([-xy_lim, xy_lim] * u.AU))
            ax.plot(upos.x.to_value(u.AU) * mult, upos.y.to_value(u.AU) * mult,
                    np.ones(2) * d, c=color, linestyle='-')
    elif s.vel.norm() != 0:
        dp = s.vel * 15 * u.day
        ax.quiver(spos.x.to_value(u.AU), spos.y.to_value(u.AU), d,
                dp.x.to_value(u.AU), dp.y.to_value(u.AU), np.zeros(1),
                color='k',
                linewidth = 3,
                arrow_length_ratio=0.2,
                length = 1.,
                normalize = False)
        
def plot_screen2(ax, s, d, pos_sel=(), color='black', mult = 1., 
                 point_sizes = 50, sl = 1., xy_lim = 1, termination = None, **kwargs):
    
    d_unit = u.pc
    tau_unit = u.us
    taudot_unit = u.us/u.day
    
    ZHAT = CartesianRepresentation(0., 0., 1., unit=u.one)
    d = d.to_value(u.pc)
    x = np.array(ax.get_xlim3d())
    y = np.array(ax.get_ylim3d())[:, np.newaxis]
    ax.plot_surface([[-sl, sl]]*2, [[-sl]*2, [sl]*2], d*np.ones((2, 2)),
                    alpha=0.1, color=color)
    x = ax.get_xticks()
    y = ax.get_yticks()[:, np.newaxis]
    spos = (s.normal * s.p if isinstance(s, Screen1D) else s.pos)[pos_sel]

    ax.scatter(spos.x.to_value(u.AU), spos.y.to_value(u.AU), d,
               c=color, s = point_sizes)
    if spos.shape:
        zo = np.arange(2)
        for pos in spos:
            ax.plot(pos.x.to_value(u.AU)*zo * mult, pos.y.to_value(u.AU)*zo * mult,
                    np.ones(2) * d, c=color, linestyle=':')
            upos = pos + (ZHAT.cross(s.normal) * ([-xy_lim, xy_lim] * u.AU))
            
            print(upos.x.shape)
            ax.plot(upos.x.to_value(u.AU) * mult, upos.y.to_value(u.AU) * mult,
                    np.ones(2) * d, c=color, linestyle='-')
            
            if termination != None:
                
                comp, bmin, bmax = termination
                
                theta = np.arctan2( spos.x, spos.y) 
                r = np.sqrt( spos.x**2 + spos.y**2).to(u.AU)
                
                
                if comp == '>':
                
                    beta = (bmin * u.AU / u.kpc * d * u.pc).to(u.AU)
                    L = np.sqrt( beta**2 + r**2).to(u.AU)
                    phi = (beta / L * u.rad)

                    p1 = np.array([ (L * np.sin( theta - phi)).value[0] ,  
                           (L * np.cos( theta - phi)).value[0] ])

                    beta = (bmax * u.AU / u.kpc * d * u.pc).to(u.AU)
                    p2 = np.array([ (-beta *np.cos(theta))[0].value + spos.x.value, 
                                    (beta * np.sin(theta))[0].value + spos.y.value])

                    linex = np.array([ p1[0], p2[0][0]])
                    liney = np.array([ p1[1], p2[1][0]])


                    ax.plot(linex, liney,
                        np.ones(2) * d, c=color, linestyle='-', lw = 5)
                    
                elif comp == '<':
                    
                    beta = (bmin * u.AU / u.kpc * d * u.pc).to(u.AU)
                    L = np.sqrt( beta**2 + r**2).to(u.AU)
                    phi = (beta / L * u.rad)

                    p1 = np.array([ (L * np.sin( theta - phi)).value[0] ,  
                           (L * np.cos( theta - phi)).value[0] ])

                    beta = (bmax * u.AU / u.kpc * d * u.pc).to(u.AU)
                    p2 = np.array([ (-beta *np.cos(theta))[0].value + spos.x.value, 
                                    (beta * np.sin(theta))[0].value + spos.y.value])

                    linex = np.array([ p1[0], p2[0][0]])
                    liney = np.array([ p1[1], p2[1][0]])


                    ax.plot(linex, liney,
                        np.ones(2) * d, c=color, linestyle='-', lw = 5)
        
def plot_trajectories(  obs1, obs2, obs12, p1_sel, p2_sel, p12_sel, ax_3d, xi_given = None, A = 1, scr_mult1 = 1, scr_mult2 = 1):
    
    d_unit = u.pc
    tau_unit = u.us
    taudot_unit = u.us/u.day

    for obs, color, path_sel in (
            (obs2, "red", p2_sel),
            (obs1, "blue", p1_sel),
            (obs12, "black", p12_sel )

    ):
        if color == 'black':
            smult = scr_mult1 
        else:
            smult = 1.
            
            
        if xi_given != None: 
            
            z_given = obs1.distance.value  # Example: z-plane in pc

            # Starting point (e.g., origin or adjust if needed)
            x0, y0 = 0.0, 0.0

            # Compute direction
            dx = A * np.sin(xi_given)
            dy = A * np.cos(xi_given)

            # End point
            x1 = x0 + dx
            y1 = y0 + dy

            # Plot on ax_3d (assuming it's your 3D plot)
            ax_3d.plot(
                [x0- dx, x1],  # x
                [y0- dy, y1],  # y
                [z_given, z_given],  # constant z
                color='green',
                linewidth=4,
                label='Custom XY Line'
            )
            
            
        pshape = obs.tau.shape
        ps = [obs.pos]
        source = obs
        ds = [0*u.pc]
        for i in range(len(pshape)+1):
            ds.append(ds[-1] + source.distance)
            source = source.source
            pos = source.pos
            ps.append(source.pos)

        xs = [p.x.to_value(u.AU) * smult for p in ps]
        ys = [p.y.to_value(u.AU) * smult for p in ps]
        zs = [d.to_value(d_unit) for d in ds]
        xyz = np.stack(
            [
                np.stack([np.broadcast_to(c, pshape) for c in cs])
                for cs in (xs, ys, zs)
            ]
        ).transpose(tuple(range(2, len(pshape)+2)) + (0, 1))
        for (_x, _y, _z) in xyz[path_sel].reshape((-1,)+xyz.shape[-2:]):
            ax_3d.plot(_x, _y, _z, color=color, linestyle=':', alpha = 0.1, lw = 3)