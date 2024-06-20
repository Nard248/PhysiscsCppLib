#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <fftw3.h>

// Define M_PI if it is not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;
using Tensor = std::vector<Matrix>;

void precompute_terms(std::vector<Matrix>& exponent, Matrix& step1, Matrix& step2, int Nx, int Ny, int Nt, double dt, double dx, double dy) {
    auto fftfreq = [](int N, double d) {
        std::vector<double> freqs(N);
        double val = 1.0 / (N * d);
        for (int i = 0; i < N / 2; ++i) {
            freqs[i] = i * val;
        }
        for (int i = -N / 2; i < 0; ++i) {
            freqs[N + i] = i * val;
        }
        return freqs;
        };

    auto kx = fftfreq(Nx, dx);
    auto ky = fftfreq(Ny, dy);
    Matrix Kx(Nx, std::vector<Complex>(Ny));
    Matrix Ky(Nx, std::vector<Complex>(Ny));

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            Kx[i][j] = kx[i];
            Ky[i][j] = ky[j];
        }
    }

    Matrix q(Nx, std::vector<Complex>(Ny));
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            q[i][j] = 1e-6 - 4.0 * M_PI * M_PI * (Kx[i][j].real() * Kx[i][j].real() + Ky[i][j].real() * Ky[i][j].real());
        }
    }

    exponent.resize(Nt, Matrix(Nx, std::vector<Complex>(Ny)));
    step1.resize(Nx, std::vector<Complex>(Ny));
    step2.resize(Nx, std::vector<Complex>(Ny));

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            Complex exp_val = exp(q[i][j] * dt);
            exponent[0][i][j] = exp_val;
            step1[i][j] = (exp_val - 1.0) / q[i][j];
            step2[i][j] = ((exp_val - 1.0) - dt * q[i][j]) / (dt * q[i][j] * q[i][j]);
        }
    }
}

Complex non_linear_function(Complex xx, Complex yy) {
    return xx * (yy - std::abs(xx) * std::abs(xx));
}

Matrix next_state(const Matrix& A, const Matrix& myu, const std::vector<Matrix>& exponent, const Matrix& step1, const Matrix& step2, Matrix& N_hat_prev, int Nx, int Ny, int order = 2) {
    Matrix A_hat(Nx, std::vector<Complex>(Ny));
    Matrix N_hat(Nx, std::vector<Complex>(Ny));

    // Perform FFT on A
    fftw_complex* in = reinterpret_cast<fftw_complex*>(const_cast<Complex*>(A[0].data()));
    fftw_complex* out = reinterpret_cast<fftw_complex*>(A_hat[0].data());
    fftw_plan plan_forward = fftw_plan_dft_2d(Nx, Ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_forward);
    fftw_destroy_plan(plan_forward);

    // Apply non-linear function
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            N_hat[i][j] = non_linear_function(A[i][j], myu[i][j]);
        }
    }

    // Perform FFT on N_hat
    in = reinterpret_cast<fftw_complex*>(N_hat[0].data());
    out = reinterpret_cast<fftw_complex*>(N_hat[0].data());
    plan_forward = fftw_plan_dft_2d(Nx, Ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_forward);
    fftw_destroy_plan(plan_forward);

    // Compute the next state
    Matrix R(Nx, std::vector<Complex>(Ny));
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            R[i][j] = A_hat[i][j] * exponent[0][i][j] + N_hat[i][j] * step1[i][j];
        }
    }

    // Perform inverse FFT on R
    fftw_complex* out_inverse = reinterpret_cast<fftw_complex*>(R[0].data());
    fftw_plan plan_inverse = fftw_plan_dft_2d(Nx, Ny, out_inverse, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_inverse);
    fftw_destroy_plan(plan_inverse);

    // Normalize the result after inverse FFT
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            R[i][j] /= (Nx * Ny);
        }
    }

    return R;
}


Matrix compute_myu(int Nx, int Ny, int Nt, std::tuple<int, int, int> myu_size, std::tuple<double, double> myu_mstd) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(std::get<0>(myu_mstd), std::get<1>(myu_mstd));

    std::vector<std::vector<std::vector<Complex>>> myu_small(std::get<0>(myu_size), std::vector<std::vector<Complex>>(std::get<1>(myu_size), std::vector<Complex>(std::get<2>(myu_size))));

    for (auto& matrix : myu_small) {
        for (auto& row : matrix) {
            for (auto& val : row) {
                val = std::abs(d(gen));
            }
        }
    }

    int scale_x = Nx / std::get<1>(myu_size);
    int scale_y = Ny / std::get<2>(myu_size);
    Matrix myu(Nx, std::vector<Complex>(Ny));

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int small_x = i / scale_x;
            int small_y = j / scale_y;
            myu[i][j] = myu_small[0][small_x][small_y];
        }
    }

    return myu;
}

Tensor compute_state(const Matrix& myu, const std::vector<Matrix>& exponent, const Matrix& step1, const Matrix& step2, int Nx, int Ny, int Nt) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 0.01);

    Matrix A_0(Nx, std::vector<Complex>(Ny));
    for (auto& row : A_0) {
        for (auto& val : row) {
            val = d(gen) + std::complex<double>(0.0, d(gen));
        }
    }

    Tensor A(Nt, Matrix(Nx, std::vector<Complex>(Ny)));
    A[0] = A_0;

    Matrix N_hat_prev(Nx, std::vector<Complex>(Ny));

    for (int i = 1; i < Nt; ++i) {
        A[i] = next_state(A[i - 1], myu, exponent, step1, step2, N_hat_prev, Nx, Ny);
    }

    return A;
}

void check_properties(const Tensor& A, const Matrix& myu) {
    std::cout << "Max value of myu: " << *std::max_element(myu[0].begin(), myu[0].end(), [](const Complex& a, const Complex& b) { return std::abs(a) < std::abs(b); }) << std::endl;
    std::cout << "Min value of myu: " << *std::min_element(myu[0].begin(), myu[0].end(), [](const Complex& a, const Complex& b) { return std::abs(a) < std::abs(b); }) << std::endl;
    std::cout << "A.shape = (" << A.size() << ", " << A[0].size() << ", " << A[0][0].size() << ")\n";
    std::cout << "Myu.shape = (" << myu.size() << ", " << myu[0].size() << ")\n";
}



int main() {
    int Nx = 600, Ny = 600, Nt = 200;
    double dt = 0.03, dx = 90.0 / 600, dy = 90.0 / 600;
    std::tuple<int, int, int> myu_size = std::make_tuple(5, 8, 8);
    std::tuple<double, double> myu_mstd = std::make_tuple(5.4, 0.8);

    // Precompute terms
    std::vector<Matrix> exponent;
    Matrix step1, step2;
    precompute_terms(exponent, step1, step2, Nx, Ny, Nt, dt, dx, dy);
    std::cout << "Precomputed terms successfully." << std::endl;

    // Compute myu
    Matrix myu = compute_myu(Nx, Ny, Nt, myu_size, myu_mstd);
    std::cout << "Computed myu successfully." << std::endl;

    // Compute the state
    Tensor A = compute_state(myu, exponent, step1, step2, Nx, Ny, Nt);
    std::cout << "Computed state successfully." << std::endl;

    // Check properties
    check_properties(A, myu);
    std::cout << "Checked properties successfully." << std::endl;

    return 0;
}
