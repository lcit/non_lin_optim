#include <doctest/doctest.h>

#include <non_lin_optim/numerical_deriv.h>
#include <non_lin_optim/types.h>
#include <non_lin_optim/version.h>

#include <iostream>

using namespace non_lin_optim;

constexpr double precision = 1e-6;
constexpr double precision_jacobian = 1e-4;
constexpr double precision_hessian = 1e-3;

TEST_CASE("Case Jacobian & Hessian Scalar 1") {
            
    auto func = [&](const Vector& x) -> Scalar {
        return 1-(pow(x(0)*(-0.5)-0.2, 2) + pow(x(1)-0.4, 4))/(0.1+x(0)*x(1));
    };
    
    Vector x(3);
    x(0) = 0.2;
    x(1) = 0.3; 
    x(2) = -0.4;
    
    Scalar y_hat = func(x);
    
    CHECK(y_hat == doctest::Approx(0.436875).epsilon(precision));
    
    Matrix J = numerical_deriv::JacobianApprox(func, x);
    
    CHECK(x(0) == doctest::Approx(0.2).epsilon(precision)); 
    CHECK(x(1) == doctest::Approx(0.3).epsilon(precision)); 
    CHECK(x(2) == doctest::Approx(-0.4).epsilon(precision));
    
    CHECK(J.size() == 3);    
    CHECK(J(0) == doctest::Approx(-0.81914).epsilon(precision_jacobian)); 
    CHECK(J(1) == doctest::Approx(0.72890).epsilon(precision_jacobian));
    CHECK(J(2) == doctest::Approx(0.0).epsilon(precision_jacobian));
    
    Matrix H = numerical_deriv::HessianApprox(func, x, 1e-6);
    
    CHECK(x(0) == doctest::Approx(0.2).epsilon(precision)); 
    CHECK(x(1) == doctest::Approx(0.3).epsilon(precision)); 
    CHECK(x(2) == doctest::Approx(-0.4).epsilon(precision));  
    
    CHECK(H.rows() == 3);
    CHECK(H.cols() == 3);
    CHECK(H(0,0) == doctest::Approx(-0.05322).epsilon(precision_hessian)); 
    CHECK(H(0,1) == doctest::Approx(3.17675).epsilon(precision_hessian));
    CHECK(H(0,2) == doctest::Approx(0.0).epsilon(precision_hessian)); 
    
    CHECK(H(1,0) == doctest::Approx(3.17675).epsilon(precision_hessian));
    CHECK(H(1,1) == doctest::Approx(-2.5722).epsilon(precision_hessian)); 
    CHECK(H(1,2) == doctest::Approx(0.0).epsilon(precision_hessian));
    
    CHECK(H(2,0) == doctest::Approx(0.0).epsilon(precision_hessian)); 
    CHECK(H(2,1) == doctest::Approx(0.0).epsilon(precision_hessian));
    CHECK(H(2,2) == doctest::Approx(0.0).epsilon(precision_hessian));
}

TEST_CASE("Case Jacobian vector 1") {

    Vector t(1);
    t(0) = 1;

    Vector y(1);
    y(0) = 1;    
            
    auto func = [&](const Vector& x) -> Vector {
        Vector y_hat(1);
        for(int i=0; i<t.size(); ++i){
            y_hat(i) = pow(x(0)*t(i)-0.2, 2) + pow(x(1)-0.4, 4);
        }
        return y-y_hat;
    };
    
    Vector x(2);
    x(0) = 0;
    x(1) = 0;  
    
    Vector y_hat = func(x);
    
    CHECK(y_hat.rows() == 1);
    CHECK(y_hat.cols() == 1);
    CHECK(y_hat(0) == doctest::Approx(0.9344).epsilon(precision));     
    
    Matrix J = numerical_deriv::JacobianApprox(func, x); 
    
    CHECK(x(0) == doctest::Approx(0.0).epsilon(precision)); 
    CHECK(x(1) == doctest::Approx(0.0).epsilon(precision));    
    
    CHECK(J.rows() == 1);
    CHECK(J.cols() == 2);    
    CHECK(J(0,0) == doctest::Approx(0.4).epsilon(precision_jacobian)); 
    CHECK(J(0,1) == doctest::Approx(0.256).epsilon(precision_jacobian));   
}

TEST_CASE("Case Jacobian vector 2") {

    Vector t(2);
    t(0) = 1;
    t(1) = 2;

    Vector y(2);
    y(0) = 1;
    y(1) = 1;
            
    auto func = [&](const Vector& x) -> Vector {
        Vector y_hat(2);
        for(int i=0; i<t.size(); ++i){
            y_hat(i) = pow(x(0)*t(i)-0.2, 2) + pow(x(1)-0.4, 4);
        }
        return y-y_hat;
    };
    
    Vector x(2);
    x(0) = 0.2;
    x(1) = 0.3;  
    
    Vector y_hat = func(x);
    
    CHECK(y_hat.rows() == 2);
    CHECK(y_hat.cols() == 1);
    CHECK(y_hat(0) == doctest::Approx(0.9999).epsilon(precision));     
    CHECK(y_hat(1) == doctest::Approx(0.9599).epsilon(precision));
    
    Matrix J = numerical_deriv::JacobianApprox(func, x);
    
    CHECK(x(0) == doctest::Approx(0.2).epsilon(precision)); 
    CHECK(x(1) == doctest::Approx(0.3).epsilon(precision));     
    
    CHECK(J.rows() == 2);
    CHECK(J.cols() == 2);    
    CHECK(J(0,0) == doctest::Approx(0.0).epsilon(precision_jacobian)); 
    CHECK(J(0,1) == doctest::Approx(0.004).epsilon(precision_jacobian));
    CHECK(J(1,0) == doctest::Approx(-0.8).epsilon(precision_jacobian)); 
    CHECK(J(1,1) == doctest::Approx(0.004).epsilon(precision_jacobian));
}

TEST_CASE("Case Jacobian vector 3") {

    Vector t(1);
    t(0) = -0.5;

    Vector y(1);
    y(0) = 1;
            
    auto func = [&](const Vector& x) -> Vector {
        Vector y_hat(1);
        for(int i=0; i<t.size(); ++i){
            y_hat(i) = (pow(x(0)*t(i)-0.2, 2) + pow(x(1)-0.4, 4))/(0.1+x(0)+x(1));
        }
        return y-y_hat;
    };
    
    Vector x(3);
    x(0) = 0.2;
    x(1) = 0.3; 
    x(2) = -0.4;
    
    Vector y_hat = func(x);
    
    CHECK(y_hat.rows() == 1);
    CHECK(y_hat.cols() == 1);
    CHECK(y_hat(0) == doctest::Approx(0.84983333).epsilon(precision));
    
    Matrix J = numerical_deriv::JacobianApprox(func, x);
    
    CHECK(x(0) == doctest::Approx(0.2).epsilon(precision)); 
    CHECK(x(1) == doctest::Approx(0.3).epsilon(precision)); 
    CHECK(x(2) == doctest::Approx(-0.4).epsilon(precision));
    
    CHECK(J.rows() == 1);
    CHECK(J.cols() == 3);    
    CHECK(J(0,0) == doctest::Approx(-0.24972).epsilon(precision_jacobian)); 
    CHECK(J(0,1) == doctest::Approx(0.25694).epsilon(precision_jacobian));
    CHECK(J(0,2) == doctest::Approx(0.0).epsilon(precision_jacobian));
}

TEST_CASE("Version") {
    static_assert(std::string_view(NON_LIN_OPTIM_VERSION) == std::string_view("1.0"));
    CHECK(std::string(NON_LIN_OPTIM_VERSION) == std::string("1.0"));
}
