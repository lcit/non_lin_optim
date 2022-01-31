#include <doctest/doctest.h>

#include <non_lin_optim/solvelin.h>
#include <non_lin_optim/version.h>

#include <iostream>
#include <Eigen/Dense>

bool print = false;

template<typename T, typename U>
void f(const Eigen::MatrixBase<T>& A, 
       const Eigen::MatrixBase<U>& b, 
       const Eigen::MatrixBase<U>& x, 
       bool positive){
    
    auto x_hat = solvelin::lu::fullPiv(A, b);
    CHECK((x_hat-x).cwiseAbs().maxCoeff() == doctest::Approx(0).epsilon(0.0001));    
    if(print) std::cout << "x_hat lu::fullPiv \n" << x_hat << std::endl;   
    
    x_hat = solvelin::qr::householderQr(A, b);
    CHECK((x_hat-x).cwiseAbs().maxCoeff() == doctest::Approx(0).epsilon(0.0001));    
    if(print) std::cout << "x_hat qr::householderQr \n" << x_hat << std::endl;
    
    x_hat = solvelin::qr::colPivHouseholderQr(A, b);
    CHECK((x_hat-x).cwiseAbs().maxCoeff() == doctest::Approx(0).epsilon(0.0001));    
    if(print) std::cout << "x_hat qr::colPivHouseholderQr \n" << x_hat << std::endl;  
    
    x_hat = solvelin::qr::fullPivHouseholderQr(A, b);
    CHECK((x_hat-x).cwiseAbs().maxCoeff() == doctest::Approx(0).epsilon(0.0001));    
    if(print) std::cout << "x_hat qr::fullPivHouseholderQr \n" << x_hat << std::endl;   
    
    x_hat = solvelin::qr::completeOrthogonalDecomposition(A, b);
    CHECK((x_hat-x).cwiseAbs().maxCoeff() == doctest::Approx(0).epsilon(0.0001));    
    if(print) std::cout << "x_hat qr::completeOrthogonalDecomposition \n" << x_hat << std::endl;    
    
    if(positive){
        x_hat = solvelin::cholesky::llt(A, b);
        CHECK((x_hat-x).cwiseAbs().maxCoeff() == doctest::Approx(0).epsilon(0.0001));
        if(print) std::cout << "x_hat cholesky::llt \n" << x_hat << std::endl;
        
        x_hat = solvelin::cholesky::ldlt(A, b);
        CHECK((x_hat-x).cwiseAbs().maxCoeff() == doctest::Approx(0).epsilon(0.0001));    
        if(print) std::cout << "x_hat cholesky::ldlt \n" << x_hat << std::endl;         
    }else{
        CHECK_THROWS(solvelin::cholesky::llt(A, b));
        if(print) std::cout << "cholesky::llt cannot solve non-positive matrices!\n" << std::endl;
        
        CHECK_THROWS(solvelin::cholesky::ldlt(A, b));
        if(print) std::cout << "cholesky::ldlt cannot solve non-positive matrices!\n" << std::endl;        
    }   
    
    x_hat = solvelin::svd::bdc(A, b);
    CHECK((x_hat-x).cwiseAbs().maxCoeff() == doctest::Approx(0).epsilon(0.0001));    
    if(print) std::cout << "x_hat svd::bdc \n" << x_hat << std::endl; 
    
    x_hat = solvelin::svd::jacobi(A, b);
    CHECK((x_hat-x).cwiseAbs().maxCoeff() == doctest::Approx(0).epsilon(0.0001));    
    if(print) std::cout << "x_hat svd::jacobi \n" << x_hat << std::endl;       
}

TEST_CASE("Positive definite") {

    Eigen::MatrixXf A(3,3);
    Eigen::Vector3f b;
    Eigen::Vector3f x;
    A << 2,-1,0,  -1,2,-1,  0,-1,2;   
    b << 3, 3, 4;
    x << 4.75, 6.5, 5.25;
    
    if(print) std::cout << "Positive definite" << std::endl;
    f(A, b, x, true);
}

TEST_CASE("Normal case") {

    Eigen::MatrixXf A(3,3);
    Eigen::Vector3f b;
    Eigen::Vector3f x;
    A << 1,2,3,  4,5,6,  7,8,10;  
    b << 3, 3, 4;
    x << -2, 1, 1;
    
    if(print) std::cout << "Normal case" << std::endl;
    f(A, b, x, false);
}

TEST_CASE("Singular Matrix") {
    
    if(print) std::cout << "Singular Matrix" << std::endl;

    Eigen::MatrixXf A(3,3);
    Eigen::Vector3f b;
    A << 9,17,1, 12,18,9, 11,13,14;
    b << 18, 0, 18;
    
    auto x_hat = solvelin::lu::fullPiv(A, b);
    if(print) std::cout << "x_hat lu::fullPiv \n" << x_hat << std::endl;
    
    x_hat = solvelin::MoorePenrose(A, b);
    if(print) std::cout << "x_hat MoorePenrose \n" << x_hat << std::endl;
    
    x_hat = solvelin::qr::householderQr(A, b);
    if(print) std::cout << "x_hat qr::householderQr \n" << x_hat << std::endl;  
    
    x_hat = solvelin::qr::colPivHouseholderQr(A, b);    
    if(print) std::cout << "x_hat qr::colPivHouseholderQr \n" << x_hat << std::endl;  
    
    x_hat = solvelin::qr::fullPivHouseholderQr(A, b);  
    if(print) std::cout << "x_hat qr::fullPivHouseholderQr \n" << x_hat << std::endl;   
    
    x_hat = solvelin::qr::completeOrthogonalDecomposition(A, b);    
    if(print) std::cout << "x_hat qr::completeOrthogonalDecomposition \n" << x_hat << std::endl;    
    /*
    x_hat = solvelin::cholesky::llt(A, b);    
    if(print) std::cout << "x_hat cholesky::llt \n" << x_hat << std::endl;   
    
    x_hat = solvelin::cholesky::ldlt(A, b);  
    if(print) std::cout << "x_hat cholesky::ldlt \n" << x_hat << std::endl;
    */
    x_hat = solvelin::svd::bdc(A, b);  
    if(print) std::cout << "x_hat svd::bdc \n" << x_hat << std::endl; 
    
    x_hat = solvelin::svd::jacobi(A, b);    
    if(print) std::cout << "x_hat svd::jacobi \n" << x_hat << std::endl;    
}

TEST_CASE("Dynamic") {

    Eigen::MatrixXd A(3,3);
    Eigen::VectorXd b(3);
    Eigen::VectorXd x(3);
    A << 1,2,3,  4,5,6,  7,8,10;  
    b << 3, 3, 4;
    x << -2, 1, 1;   
    
    if(print) std::cout << "Dynamic matrix" << std::endl;
    f(A, b, x, false);
}

TEST_CASE("Version") {
    static_assert(std::string_view(NON_LIN_OPTIM_VERSION) == std::string_view("1.0"));
    CHECK(std::string(NON_LIN_OPTIM_VERSION) == std::string("1.0"));
}
