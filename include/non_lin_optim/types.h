#pragma once

#include<Eigen/Dense>

namespace non_lin_optim {

    using Scalar = double;
    using Matrix = Eigen::Matrix< Scalar, Eigen::Dynamic, Eigen::Dynamic >;
    using Vector = Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >;
    using Func_v = std::function< Vector(const Vector &x) >;
    using Func_s = std::function< Scalar(const Vector &x) >;

    enum ResultInfo {
        Converged,
        MaxIterationReached,
        TolleranceReached
    };

    inline
    std::ostream& operator<<(std::ostream& out, const ResultInfo value){
        const char* s = 0;
        #define PROCESS_VAL(p) case(p): s = #p; break;
        switch(value){
            PROCESS_VAL(Converged);     
            PROCESS_VAL(MaxIterationReached);
            PROCESS_VAL(TolleranceReached);
        }
        #undef PROCESS_VAL
        return out << s;
}

} // end namespace non_lin_optim