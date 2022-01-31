#pragma once

#include "types.h"
#include <iostream>
#include <math.h>

namespace non_lin_optim {

namespace numerical_deriv {
    
    inline
    Matrix JacobianApprox(const Func_s& func, Vector& x, const Scalar step=1e-6){
        Scalar f_x = func(x);
        Vector J(x.size());
        
        for(int i=0; i<x.size(); ++i){
            x(i) += step;
            Scalar f_x_p = func(x);
            x(i) -= step;
            J(i) = (f_x_p-f_x)/step;
        }
        return J;
    }    

    inline
    Matrix JacobianApprox(const Func_v& func, Vector& x, const Scalar step=1e-6){
        Vector f_x = func(x);
        Matrix J(f_x.size(), x.size());
        
        for(int i=0; i<x.size(); ++i){
            x(i) += step;
            Vector f_x_p = func(x);
            x(i) -= step;   
            J.col(i) = (f_x_p-f_x)/step;
        }
        return J;
    }

    inline
    Matrix JacobianApproxCentral(const Func_v& func, Vector& x, const Scalar step=1e-6){

        Vector f_x = func(x);
        Matrix J(f_x.size(), x.size());
        
        for(int i=0; i<x.size(); ++i){
            x(i) += step;
            Vector f_x_p = func(x);
            x(i) -= 2*step;  
            Vector f_x_n = func(x);
            x(i) += step;
            J.col(i) = (f_x_p-f_x_n)/(2*step);
        }
        return J;
    }

    inline
    Matrix HessianApprox(const Func_s& func, Vector& x, const Scalar step=1e-6){
        Scalar f_x = func(x);
        Matrix H(x.size(), x.size());
        
        for(int i=0; i<x.size(); ++i){
            for(int j=i; j<x.size(); ++j){
                 
                if(i==j){
                    x(i) += step;
                    Scalar f_x_ps = func(x);
                    x(i) += step;
                    Scalar f_x_p2s = func(x);
                    x(i) -= 3*step;
                    Scalar f_x_ns = func(x);
                    x(i) -= step;
                    Scalar f_x_n2s = func(x);
                    x(i) += 2*step;

                    H(i,j) = (-f_x_p2s + 16*f_x_ps - 30*f_x + 16*f_x_ns - f_x_n2s)/(12*step*step);
                } else {              
                    x(i) += step;
                    x(j) += step;
                    Scalar f_x_ps_ps = func(x);
                    x(i) -= 2*step;
                    Scalar f_x_ns_ps = func(x);
                    x(j) -= 2*step;
                    Scalar f_x_ns_ns = func(x);
                    x(i) += 2*step;
                    Scalar f_x_ps_ns = func(x); 
                    x(i) -= step;
                    x(j) += step;   

                    H(i,j) = (f_x_ps_ps - f_x_ps_ns - f_x_ns_ps + f_x_ns_ns)/(4*step*step);
                    H(j,i) = H(i,j);
                }
            }
        }
        return H;
    } 

} // end namespace numerical_deriv

} // end namespace non_lin_optim