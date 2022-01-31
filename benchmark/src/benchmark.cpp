#include<iostream>
#include<chrono>
#include<numeric>
#include<iomanip>
#include<vector>
#include<map>
#include<exception>
#include<Eigen/Dense>

#include <non_lin_optim/solvelin.h>
#include <non_lin_optim/version.h>

template<typename TimeT = std::chrono::milliseconds>
struct measure {
    template<typename F, typename ...Args>
    static typename TimeT::rep run(F&& func, Args&&... args) {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

template<size_t N>
struct mean_stddev {
    template<typename F, typename ...Args>
    static auto run(F&& func, Args&&... args){
        std::array<double, N> buffer;
        for(auto& buf:buffer)
            buf = std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto sum = std::accumulate(std::begin(buffer), std::end(buffer), 0.0);
        auto mean = sum/buffer.size();
        std::array<double, N> diff;
        std::transform(std::begin(buffer), std::end(buffer), std::begin(diff), [mean](auto x) { return x - mean; });
        auto sq_sum = std::inner_product(std::begin(diff), std::end(diff), std::begin(diff), 0.0);
        auto stddev = std::sqrt(sq_sum/buffer.size());
        return std::make_pair(mean,stddev);
    }
};

using TA = Eigen::MatrixXd;
using TB = Eigen::VectorXd;
using TR = Eigen::VectorXd;
using solve_fcn = std::function<TR(TA,TB)>;

template<typename F>
auto bench_solvelin(F&& func, int n){
    TA A = TA::Random(n,n);
    TB b = TB::Random(n);
    A = A.transpose()*A; // make it positive matrix to avoid errors
    return measure<>::run(func, A, b);
}

struct Problem {
    TA A;
    TB b;
    TR x;
};

int main(){
    
    std::cout << "Version:" << NON_LIN_OPTIM_VERSION << std::endl;
    
    const int times = 10;
    
    int sizes[] = {20, 50, 100};
    
    std::vector<std::pair<std::string, solve_fcn> > functions = {
        {"solvelin::MoorePenrose", solvelin::MoorePenrose<TA,TB>},
        {"solvelin::lu::fullPiv", solvelin::lu::fullPiv<TA,TB>},
        {"solvelin::qr::householderQr", solvelin::qr::householderQr<TA,TB>},
        {"solvelin::qr::colPivHouseholderQr", solvelin::qr::colPivHouseholderQr<TA,TB>},
        {"solvelin::qr::fullPivHouseholderQr", solvelin::qr::fullPivHouseholderQr<TA,TB>},
        {"solvelin::qr::completeOrthogonalDecomposition", solvelin::qr::completeOrthogonalDecomposition<TA,TB>},
        {"solvelin::cholesky::llt", solvelin::cholesky::llt<TA,TB>},
        {"solvelin::cholesky::ldlt", solvelin::cholesky::ldlt<TA,TB>},
        {"solvelin::svd::bdc", solvelin::svd::bdc<TA,TB>},
        {"solvelin::svd::jacobi", solvelin::svd::jacobi<TA,TB>}
    };
    
    std::cout << "=== Performance ===" << std::endl;
    for(int n:sizes){
        std::cout << "--- Matrix size (" << n << "," << n << ") ---" << std::endl;
        
        for(auto& [name,func]:functions){
            auto [mean,std] = mean_stddev<times>::run([func,n](){return bench_solvelin<>(func, n);});
            std::cout << name << ":" 
                      << std::setw(55-name.length()) << mean << "(+-" << std::setprecision(6) 
                      << std::setw(8) << std << ") [ms]" 
                      << std::endl;       
        }
    }
    
    const int n = 50;
    std::cout << "=== Accuracy (residual_error, error) ===" << std::endl;
    for(int m:sizes){
        std::cout << "--- Matrix size (" << n << "," << m << ") ---" << std::endl;    
        
        TA A = TA::Random(n,m);
        TB b = TB::Random(n);
        if(n==m) 
            A = A.transpose()*A; // make it positive matrix to avoid errors
        TR x = solvelin::svd::bdc<TA,TB>(A, b);
        
        for(auto& [name,func]:functions){
            
            try {
                TR x_hat = func(A,b);
                
                double residual_error = (A*x_hat - b).norm();
                double error = (x-x_hat).norm();

                std::cout << name << ":" 
                          << std::setw(60-name.length()) << residual_error << std::setprecision(6)
                          << std::setw(15) << error << std::setprecision(6)
                          << std::endl;                 
            } catch (std::exception& e){
                std::cout << name << " " << e.what() << std::endl;
            }
        }    
    }
    
    return 0;
}