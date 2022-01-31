#include <non_lin_optim/optim.h>
#include <non_lin_optim/numerical_deriv.h>
#include <iostream>

#include <iostream>

using namespace non_lin_optim;

int main(int argc, char** argv){

    Vector t(2);
    t(0) = 1;
    t(1) = 2;

    Vector x(2); 
    x(0) = 0.0;
    x(1) = 0.0; 

    auto func = [&](const Vector& x) -> Vector {
        Vector y_hat(2);
        for(int i=0; i<t.size(); ++i){
            y_hat(i) = pow(x(0)*t(i)-0.2, 2) + x(1);
        }
        return y_hat.array()-1; // return residuals
    };    

    auto optimizer = optim::GaussianNewton(func);  
    auto info = optimizer.run(x);

    std::cout << info << std::endl;
    std::cout << "x:" << x << std::endl;

    Matrix J = numerical_deriv::JacobianApprox(func, x);
    std::cout << "J:" << J << std::endl;

    return 0;
}
