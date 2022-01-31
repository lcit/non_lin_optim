#pragma once

#include <math.h>
#include <iostream>
#include <stdexcept>
#include "types.h"
#include "numerical_deriv.h"
#include "solvelin.h"

namespace non_lin_optim {

namespace optim {

    template<typename Func>
    class BaseMinimization {
    protected:
        const Func f;
        const int max_iter = 10000;
        const Scalar tol = 1e-6;
        const Scalar lambda = 1;

        std::vector<Scalar> errors;
        std::vector<Vector> deltas;

    public:
        BaseMinimization(const Func& f, const int max_iter=10000, const Scalar tol=1e-12, const Scalar lambda=1)
        : f(f), max_iter(max_iter), tol(tol), lambda(lambda) {}

        virtual Scalar compute_error(Vector& x) = 0;

        virtual Vector compute_delta(Vector& x) = 0;

        ResultInfo run(Vector& x){

            double prev_error = std::numeric_limits<Scalar>::max();
            int convergence_count = 0;
            for(int i=0; i<max_iter; ++i) {

                Scalar error = compute_error(x);
                errors.push_back(error);

                if(error<tol) return TolleranceReached;
                if(i%100==0){
                    if(error<prev_error){
                        prev_error = error;
                        convergence_count = 0;
                    }else{
                        convergence_count++;
                        if(convergence_count>10)
                            return Converged;
                    }
                }

                Vector delta = compute_delta(x);
                deltas.push_back(delta);
                
                x = x+lambda*delta;
            }
            return MaxIterationReached;
        }
    }; 

    class Newton : public BaseMinimization<Func_s> {
    public:        
        Newton(const Func_s& f, const int max_iter=10000, const Scalar tol=1e-12, const Scalar lambda=1)
        : BaseMinimization(f, max_iter, tol, lambda) {}

        Scalar compute_error(Vector& x) override {
            Scalar error = f(x);
            return error;
        }

        Vector compute_delta(Vector& x) override {
            Matrix gp = numerical_deriv::JacobianApprox(f, x);
            Matrix gpp = numerical_deriv::HessianApprox(f, x);
            Vector delta = solvelin::cholesky::ldlt(gpp, -gp);
            return delta;
        }
    };    

    class GaussianNewton : public BaseMinimization<Func_v> {
    private:
        Vector residuals;
    public:
        GaussianNewton(const Func_v& f, const int max_iter=10000, const Scalar tol=1e-12, const Scalar lambda=1)
        : BaseMinimization(f, max_iter, tol, lambda) {}
            
        Scalar compute_error(Vector& x) override {
            residuals = f(x);
            Scalar error = pow(residuals.norm(), 2);
            return error;
        }

        Vector compute_delta(Vector& x) override {
            Matrix J = numerical_deriv::JacobianApproxCentral(f, x);
            Matrix JtJ = J.transpose()*J;
            Matrix Jr = -J.transpose()*residuals;
            Vector delta;
            try{
                delta = solvelin::cholesky::ldlt(JtJ, Jr);
            } catch (const std::exception& e) {
                std::cout << "using fullPiv" << std::endl;
                delta = solvelin::lu::fullPiv(JtJ, Jr);
            }
            return delta;
        }
    };

    class GradientDescent : public BaseMinimization<Func_v> {
    private:
        Vector residuals;
    public:
        GradientDescent(const Func_v& f, const int max_iter=10000, const Scalar tol=1e-12, const Scalar lambda=1)
        : BaseMinimization(f, max_iter, tol, lambda) {}

        Scalar compute_error(Vector& x) override {
            residuals = f(x);
            Scalar error = pow(residuals.norm(), 2);
            return error;
        }

        Vector compute_delta(Vector& x) override {
            Matrix J = numerical_deriv::JacobianApproxCentral(f, x);
            Vector delta = -J.transpose()*residuals;
            return delta;
        }
    };    

} // end namespace optim

} // end namespace non_lin_optim