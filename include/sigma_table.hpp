#ifndef SIGMA_TABLE_HPP
#define SIGMA_TABLE_HPP

#include <string>
#include <vector>

class SigmaTable {
public:
    explicit SigmaTable(const std::string& filename);

    double sigma_cm2(double E_GeV) const;
    double sigma_GeV_minus2(double E_GeV) const;

    double Emin() const;
    double Emax() const;
    std::size_t size() const;

private:
    std::vector<double> E_;
    std::vector<double> sigma_GeV2_;
    std::vector<double> sigma_cm2_;

    static double log_interp(
        double x,
        const std::vector<double>& xs,
        const std::vector<double>& ys
    );
};

#endif