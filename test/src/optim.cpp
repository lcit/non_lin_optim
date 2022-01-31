#include <doctest/doctest.h>

#include <non_lin_optim/optim.h>
#include <non_lin_optim/types.h>
#include <non_lin_optim/version.h>

#include <iostream>

using namespace non_lin_optim;

constexpr double precision = 1e-3;
constexpr double precision_gd = 5e-2;

TEST_CASE("Case Bivariate Gaussian - Gaussian Newton") {

    Vector x_gt(2); 
    x_gt(0) = 1.4;
    x_gt(1) = -3.5;    
    
    Vector x(2); 
    x(0) = 1.0;
    x(1) = -2.0; 
    
    Scalar sigma = 2;
    
    auto func = [&](const Vector& x) -> Vector {
        Vector y_hat(1);
        y_hat(0) = 1-exp((pow(x(0)-x_gt(0),2)+pow(x(1)-x_gt(1),2))/sigma);
        return y_hat;
    };    
    
    auto optimizer = optim::GaussianNewton(func, 10000, 1e-12, 1);    
    optimizer.run(x);
    
    CHECK(x(0) == doctest::Approx(x_gt(0)).epsilon(precision));
    CHECK(x(1) == doctest::Approx(x_gt(1)).epsilon(precision));    
}

TEST_CASE("Case Bivariate Gaussian - Gradient Descent") {

    Vector x_gt(2); 
    x_gt(0) = 1.4;
    x_gt(1) = -3.5;    
    
    Vector x(2); 
    x(0) = 1.0;
    x(1) = -2.0;   
    
    Scalar sigma = 2;    
    
    auto func = [&](const Vector& x) -> Vector {
        Vector y_hat(1);
        y_hat(0) = 1-exp((pow(x(0)-x_gt(0),2)+pow(x(1)-x_gt(1),2))/sigma);
        return y_hat;
    };    
    
    auto optimizer = optim::GradientDescent(func, 10000, 1e-12, 0.1);    
    optimizer.run(x);
        
    CHECK(x(0) == doctest::Approx(x_gt(0)).epsilon(precision_gd));
    CHECK(x(1) == doctest::Approx(x_gt(1)).epsilon(precision_gd));    
}
    
TEST_CASE("Case Bivariate Gaussian - Newton") {

    Vector x_gt(2); 
    x_gt(0) = 1.4;
    x_gt(1) = -3.5;    
    
    Vector x(2); 
    x(0) = 1.0;
    x(1) = -2.0; 

    Scalar sigma = 2;    
    
    auto func = [&](const Vector& x) -> Scalar {
        return pow(1-exp((pow(x(0)-x_gt(0),2)+pow(x(1)-x_gt(1),2))/sigma), 2);
    };    
    
    auto optimizer = optim::Newton(func, 10000, 1e-12, 1);
    optimizer.run(x);

    CHECK(x(0) == doctest::Approx(x_gt(0)).epsilon(precision));
    CHECK(x(1) == doctest::Approx(x_gt(1)).epsilon(precision));    
}

TEST_CASE("Case gaussian - Gaussian Newton") {
            
    Vector t(10);
    for(int i=0; i<10; ++i)
        t(i) = (float)i/10;
    
    auto func = [&](const Vector& x) -> Vector {
        Vector y_hat(10);
        for(int i=0; i<t.size(); ++i){
            y_hat(i) = pow(x(0)*t(i)-0.2, 2);
        }
        return y_hat;
    };  

    Vector x_gt(1);
    x_gt(0) = 0.2;     
    Vector y = func(x_gt);
            
    auto func_residuals = [&](const Vector& x) -> Vector {
        Vector y_hat = func(x);
        return y_hat-y;
    };    
    
    Vector x(1); 
    x(0) = 0.0;
    
    auto optimizer = optim::GaussianNewton(func_residuals, 10000, 1e-12, 1);
    optimizer.run(x);
    
    CHECK(x(0) == doctest::Approx(x_gt(0)).epsilon(precision));
}

TEST_CASE("Case gaussian - Gaussian Newton 2") {
            
    Vector t(10);
    for(int i=0; i<10; ++i)
        t(i) = (float)i/10;
    
    auto func = [&](const Vector& x) -> Vector {
        Vector y_hat(10);
        for(int i=0; i<t.size(); ++i){
            y_hat(i) = pow(x(0)*t(i)-0.2, 2) + pow(x(1)-0.9, 4);
        }
        return y_hat;
    };  

    Vector x_gt(2);
    x_gt(0) = 0.2;   
    x_gt(1) = 0.3;
    Vector y = func(x_gt);
            
    auto func_residuals = [&](const Vector& x) -> Vector {
        Vector y_hat = func(x);
        return y_hat-y;
    };    
    
    Vector x(2); 
    x(0) = 0.0; 
    x(1) = 0.1; 
    
    auto optimizer = optim::GaussianNewton(func_residuals, 10000, 1e-12, 1);
    optimizer.run(x);
    
    CHECK(x(0) == doctest::Approx(x_gt(0)).epsilon(precision));
    CHECK(x(1) == doctest::Approx(x_gt(1)).epsilon(precision));
}

TEST_CASE("Case gaussian - Gradient Descent") {
            
    Vector t(10);
    for(int i=0; i<10; ++i)
        t(i) = (float)i/10;
    
    auto func = [&](const Vector& x) -> Vector {
        Vector y_hat(10);
        for(int i=0; i<t.size(); ++i){
            y_hat(i) = pow(x(0)*t(i)-0.2, 2) + pow(x(1)-0.9, 4);
        }
        return y_hat;
    };  

    Vector x_gt(2);
    x_gt(0) = 0.2;   
    x_gt(1) = 0.3;
    Vector y = func(x_gt);
            
    auto func_residuals = [&](const Vector& x) -> Vector {
        Vector y_hat = func(x);
        return y_hat-y;
    };    
    
    Vector x(2); 
    x(0) = 0.0; 
    x(1) = 0.1; 
    
    auto optimizer = optim::GradientDescent(func_residuals, 10000, 1e-12, 0.1);
    optimizer.run(x);
    
    CHECK(x(0) == doctest::Approx(x_gt(0)).epsilon(precision_gd));
    CHECK(x(1) == doctest::Approx(x_gt(1)).epsilon(precision_gd));
}

TEST_CASE("Case gaussian - Newton") {
            
    Vector t(10);
    for(int i=0; i<10; ++i)
        t(i) = (float)i/10;
    
    auto func = [&](const Vector& x) -> Vector {
        Vector y_hat(10);
        for(int i=0; i<t.size(); ++i){
            y_hat(i) = pow(x(0)*t(i)-0.2, 2) + pow(x(1)-0.9, 4);
        }
        return y_hat;
    };  

    Vector x_gt(2);
    x_gt(0) = 0.2;   
    x_gt(1) = 0.3;
    Vector y = func(x_gt);
            
    auto func_scalar = [&](const Vector& x) -> Scalar {
        Vector y_hat = func(x);
        return pow((y_hat-y).norm(), 2);
    };    
    
    Vector x(2); 
    x(0) = 0.0; 
    x(1) = 0.1; 
    
    auto optimizer = optim::Newton(func_scalar, 10000, 1e-12, 1);
    optimizer.run(x);
    
    CHECK(x(0) == doctest::Approx(x_gt(0)).epsilon(precision));
    CHECK(x(1) == doctest::Approx(x_gt(1)).epsilon(precision));
}

TEST_CASE("Version") {
    static_assert(std::string_view(NON_LIN_OPTIM_VERSION) == std::string_view("1.0"));
    CHECK(std::string(NON_LIN_OPTIM_VERSION) == std::string("1.0"));
}
