#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <cufft.h>
#include <Eigen/Dense>
#include <unordered_set>

using namespace std;
using namespace Eigen;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Function to initialize parameters
struct Params {
    double dt, dx, dy;
    int Nt, Nx, Ny;
    array<int, 3> myu_size;
    array<double, 2> myu_mstd;
    VectorXd kx, ky;
    MatrixXd Kx, Ky, q, exponent, expm1, step_1, step_2;
    cufftHandle plan_forward;
    cufftHandle plan_backward;
};

__global__ void initialize_parameters(float* kx, float* ky, float* Kx, float* Ky, float* q, float* exponent, float* expm1, float* step_1, float* step_2, int Nx, int Ny, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j < Ny) {
        kx[i] = (i < Nx / 2) ? i / (dx * Nx) : (i - Nx) / (dx * Nx);
        ky[j] = (j < Ny / 2) ? j / (dy * Ny) : (j - Ny) / (dy * Ny);
        Kx[i * Ny + j] = kx[i];
        Ky[i * Ny + j] = ky[j];
        q[i * Ny + j] = 1e-6 - 4.0 * M_PI * M_PI * (Kx[i * Ny + j] * Kx[i * Ny + j] + Ky[i * Ny + j] * Ky[i * Ny + j]);
        exponent[i * Ny + j] = exp(q[i * Ny + j] * dt);
        expm1[i * Ny + j] = exp(q[i * Ny + j] * dt) - 1.0;
        step_1[i * Ny + j] = expm1[i * Ny + j] / q[i * Ny + j];
        step_2[i * Ny + j] = (expm1[i * Ny + j] - dt * q[i * Ny + j]) / (dt * q[i * Ny + j] * q[i * Ny + j]);
    }
}

Params initialize_parameters_host(array<double, 3> d, array<int, 3> N, array<int, 3> myu_size, array<double, 2> myu_mstd) {
    Params params;
    params.dt = d[0];
    params.dx = d[1];
    params.dy = d[2];
    params.Nt = N[0];
    params.Nx = N[1];
    params.Ny = N[2];
    params.myu_size = myu_size;
    params.myu_mstd = myu_mstd;

    params.kx = VectorXd::LinSpaced(params.Nx, -0.5 / params.dx, 0.5 / params.dx);
    params.ky = VectorXd::LinSpaced(params.Ny, -0.5 / params.dy, 0.5 / params.dy);

    params.Kx = params.kx.replicate(1, params.Ny);
    params.Ky = params.ky.transpose().replicate(params.Nx, 1);

    params.q = 1e-6 - 4.0 * M_PI * M_PI * (params.Kx.array().square() + params.Ky.array().square());
    params.exponent = params.q.array().exp() * params.dt;
    params.expm1 = params.q.array().exp() * params.dt - 1.0;
    params.step_1 = params.expm1.array() / params.q.array();
    params.step_2 = (params.expm1.array() - params.dt * params.q.array()) / (params.dt * params.q.array().square());

    // Allocate GPU memory for parameters
    float* d_kx, * d_ky, * d_Kx, * d_Ky, * d_q, * d_exponent, * d_expm1, * d_step_1, * d_step_2;
    cudaMalloc((void**)&d_kx, params.Nx * sizeof(float));
    cudaMalloc((void**)&d_ky, params.Ny * sizeof(float));
    cudaMalloc((void**)&d_Kx, params.Nx * params.Ny * sizeof(float));
    cudaMalloc((void**)&d_Ky, params.Nx * params.Ny * sizeof(float));
    cudaMalloc((void**)&d_q, params.Nx * params.Ny * sizeof(float));
    cudaMalloc((void**)&d_exponent, params.Nx * params.Ny * sizeof(float));
    cudaMalloc((void**)&d_expm1, params.Nx * params.Ny * sizeof(float));
    cudaMalloc((void**)&d_step_1, params.Nx * params.Ny * sizeof(float));
    cudaMalloc((void**)&d_step_2, params.Nx * params.Ny * sizeof(float));

    // Launch kernel to initialize parameters on GPU
    dim3 blockSize(16, 16);
    dim3 gridSize((params.Nx + blockSize.x - 1) / blockSize.x, (params.Ny + blockSize.y - 1) / blockSize.y);
    initialize_parameters << <gridSize, blockSize >> > (d_kx, d_ky, d_Kx, d_Ky, d_q, d_exponent, d_expm1, d_step_1, d_step_2, params.Nx, params.Ny, params.dx, params.dy, params.dt);

    // Create FFT plans
    cufftPlan2d(&params.plan_forward, params.Nx, params.Ny, CUFFT_C2C);
    cufftPlan2d(&params.plan_backward, params.Nx, params.Ny, CUFFT_C2C);

    return params;
}

// Kernel for non-linear function
__global__ void non_linear_function(cufftComplex* xx, float* yy, cufftComplex* result, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j < Ny) {
        int index = i * Ny + j;
        float xx_abs2 = xx[index].x * xx[index].x + xx[index].y * xx[index].y;
        result[index].x = xx[index].x * (yy[index] - xx_abs2);
        result[index].y = xx[index].y * (yy[index] - xx_abs2);
    }
}

// Perform FFT using cuFFT
void fft_forward(cufftHandle plan, cufftComplex* d_data) {
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
}

void fft_inverse(cufftHandle plan, cufftComplex* d_data) {
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
}

// Kernel for computing the next state
__global__ void compute_next_state(cufftComplex* A_hat, cufftComplex* N_hat, cufftComplex* R, float* exponent, float* step_1, float* step_2, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j < Ny) {
        int index = i * Ny + j;
        R[index].x = A_hat[index].x * exponent[index] + N_hat[index].x * step_1[index] - (N_hat[index].x - A_hat[index].x) * step_2[index] * 0.01;
        R[index].y = A_hat[index].y * exponent[index] + N_hat[index].y * step_1[index] - (N_hat[index].y - A_hat[index].y) * step_2[index] * 0.01;
    }
}

// Function to compute myu
MatrixXd compute_myu(Params& params) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(params.myu_mstd[0], params.myu_mstd[1]);

    // Correct dimensions for myu
    MatrixXd myu_small = MatrixXd::NullaryExpr(params.myu_size[0], params.myu_size[1], [&]() { return abs(d(gen)); });
    MatrixXd myu = myu_small.replicate(params.Nt / params.myu_size[0], (params.Nx * params.Ny) / params.myu_size[1]);

    return myu;
}

// Function to compute the state over time on the GPU
MatrixXcd compute_state(MatrixXd& myu, Params& params) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0.0, 0.01);

    cufftComplex* d_A;
    cufftComplex* d_N_hat;
    cufftComplex* d_R;
    float* d_myu;

    cudaMalloc((void**)&d_A, params.Nx * params.Ny * sizeof(cufftComplex));
    cudaMalloc((void**)&d_N_hat, params.Nx * params.Ny * sizeof(cufftComplex));
    cudaMalloc((void**)&d_R, params.Nx * params.Ny * sizeof(cufftComplex));
    cudaMalloc((void**)&d_myu, params.Nx * params.Ny * sizeof(float));

    // Initialize A_0 on the GPU
    cufftComplex* A_0 = (cufftComplex*)malloc(params.Nx * params.Ny * sizeof(cufftComplex));
    for (int i = 0; i < params.Nx; ++i) {
        for (int j = 0; j < params.Ny; ++j) {
            A_0[i * params.Ny + j].x = d(gen);
            A_0[i * params.Ny + j].y = d(gen);
        }
    }
    cudaMemcpy(d_A, A_0, params.Nx * params.Ny * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    for (int i = 1; i < params.Nt; ++i) {
        // Copy myu to GPU
        cudaMemcpy(d_myu, myu.data() + i * params.Nx * params.Ny, params.Nx * params.Ny * sizeof(float), cudaMemcpyHostToDevice);

        // Non-linear function on GPU
        dim3 blockSize(16, 16);
        dim3 gridSize((params.Nx + blockSize.x - 1) / blockSize.x, (params.Ny + blockSize.y - 1) / blockSize.y);
        non_linear_function << <gridSize, blockSize >> > (d_A, d_myu, d_N_hat, params.Nx, params.Ny);

        // FFT forward
        fft_forward(params.plan_forward, d_N_hat);

        // Compute next state on GPU
        compute_next_state << <gridSize, blockSize >> > (d_A, d_N_hat, d_R, params.exponent.data(), params.step_1.data(), params.step_2.data(), params.Nx, params.Ny);

        // FFT inverse
        fft_inverse(params.plan_backward, d_R);

        // Copy result back to d_A for next iteration
        cudaMemcpy(d_A, d_R, params.Nx * params.Ny * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    }

    MatrixXcd result(params.Nx, params.Ny);
    cudaMemcpy(result.data(), d_R, params.Nx * params.Ny * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_N_hat);
    cudaFree(d_R);
    cudaFree(d_myu);
    free(A_0);

    return result;
}

// Function to get unique values from an Eigen matrix
template<typename T>
unordered_set<T> unique_values(const Matrix<T, Dynamic, Dynamic>& matrix) {
    unordered_set<T> unique;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            unique.insert(matrix(i, j));
        }
    }
    return unique;
}

int main() {
    array<double, 3> d = { 0.03, 90.0 / 600, 90.0 / 600 };
    array<int, 3> N = { 200, 600, 600 };
    array<int, 3> myu_size = { 5, 8, 8 };
    array<double, 2> myu_mstd = { 5.4, 0.8 };

    Params params = initialize_parameters_host(d, N, myu_size, myu_mstd);
    MatrixXd myu = compute_myu(params);
    MatrixXcd state = compute_state(myu, params);

    // Checking properties
    auto unique_myu = unique_values(myu);
    cout << "Unique Myus count\t" << unique_myu.size() << endl;
    cout << "Max value of myu:\t" << myu.maxCoeff() << endl;
    cout << "Min value of myu:\t" << myu.minCoeff() << endl;
    cout << "Any NaN values in Myu\t\t" << myu.hasNaN() << endl;
    cout << "Any NaN values in A\t\t" << state.hasNaN() << endl;

    // Destroy FFT plans
    cufftDestroy(params.plan_forward);
    cufftDestroy(params.plan_backward);

    return 0;
}
