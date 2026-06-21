#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace constants {

    // Mathematical constants
    constexpr double pi = 3.141592653589793238462643383279502884;

    // Physical constants in cgs
    constexpr double c_cgs   = 2.99792458e10;        // cm / s
    constexpr double G_cgs   = 6.67430e-8;           // cm^3 g^-1 s^-2
    constexpr double Msun_g  = 1.98847e33;           // g
    constexpr double m_u_g   = 1.66053906660e-24;    // g

    // Gravitational radius:
    // r_g = GM / c^2
    inline double rg_cm(double M_msun)
    {
        return G_cgs * M_msun * Msun_g / (c_cgs * c_cgs);
    }

}

#endif