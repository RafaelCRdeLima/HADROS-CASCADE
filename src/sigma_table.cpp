#include "sigma_table.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

SigmaTable::SigmaTable(const std::string& filename)
{
    std::ifstream in(filename);

    if (!in) {
        throw std::runtime_error("Could not open sigma table: " + filename);
    }

    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream iss(line);

        double E, sig_GeV2, sig_cm2;

        if (!(iss >> E >> sig_GeV2 >> sig_cm2)) {
            continue;
        }

        if (E <= 0.0 || sig_GeV2 <= 0.0 || sig_cm2 <= 0.0) {
            continue;
        }

        E_.push_back(E);
        sigma_GeV2_.push_back(sig_GeV2);
        sigma_cm2_.push_back(sig_cm2);
    }

    if (E_.size() < 2) {
        throw std::runtime_error(
            "Sigma table must contain at least two valid data lines."
        );
    }

    for (std::size_t i = 1; i < E_.size(); ++i) {
        if (E_[i] <= E_[i - 1]) {
            throw std::runtime_error(
                "Energy grid in sigma table must be strictly increasing."
            );
        }
    }
}

double SigmaTable::sigma_cm2(double E_GeV) const
{
    return log_interp(E_GeV, E_, sigma_cm2_);
}

double SigmaTable::sigma_GeV_minus2(double E_GeV) const
{
    return log_interp(E_GeV, E_, sigma_GeV2_);
}

double SigmaTable::Emin() const
{
    return E_.front();
}

double SigmaTable::Emax() const
{
    return E_.back();
}

std::size_t SigmaTable::size() const
{
    return E_.size();
}

double SigmaTable::log_interp(
    double x,
    const std::vector<double>& xs,
    const std::vector<double>& ys
)
{
    if (x <= 0.0) {
        throw std::runtime_error("Interpolation energy must be positive.");
    }

    if (x < xs.front() || x > xs.back()) {
        throw std::runtime_error(
            "Requested energy outside sigma table range."
        );
    }

    auto it = std::lower_bound(xs.begin(), xs.end(), x);

    if (it == xs.begin()) {
        return ys.front();
    }

    if (it == xs.end()) {
        return ys.back();
    }

    std::size_t i = static_cast<std::size_t>(it - xs.begin());

    double x1 = xs[i - 1];
    double x2 = xs[i];

    double y1 = ys[i - 1];
    double y2 = ys[i];

    double lx  = std::log(x);
    double lx1 = std::log(x1);
    double lx2 = std::log(x2);

    double ly1 = std::log(y1);
    double ly2 = std::log(y2);

    double t = (lx - lx1) / (lx2 - lx1);

    return std::exp(ly1 + t * (ly2 - ly1));
}