#pragma once

#include <Eigen/Dense>
#include <stdexcept>

#define MatrixTT Eigen::Matrix<typename Eigen::ScalarBinaryOpTraits<typename T::Scalar, typename T::Scalar>::ReturnType,\
                               T::RowsAtCompileTime,\
                               T::ColsAtCompileTime>

#define MatrixTU Eigen::Matrix<typename Eigen::ScalarBinaryOpTraits<typename T::Scalar, typename U::Scalar>::ReturnType,\
                               T::RowsAtCompileTime,\
                               U::ColsAtCompileTime>

namespace solvelin {

    template<typename T, typename U>
    MatrixTU MoorePenrose(const Eigen::MatrixBase<T>& A, const Eigen::MatrixBase<U>& b){
        if(A.rows()!=A.cols())
            throw std::invalid_argument("A is not square!");
        return A.inverse()*b;
    }

namespace lu {

    template<typename T, typename U>
    MatrixTU fullPiv(const Eigen::MatrixBase<T>& A, const Eigen::MatrixBase<U>& b){
        Eigen::FullPivLU<T> lu(A);
        return lu.solve(b);
    }

} // end namespace lu 

namespace qr {

    template<typename T, typename U>
    MatrixTU householderQr(const Eigen::MatrixBase<T>& A, const Eigen::MatrixBase<U>& b){
        return A.householderQr().solve(b);
    }

    template<typename T, typename U>
    MatrixTU colPivHouseholderQr(const Eigen::MatrixBase<T>& A, const Eigen::MatrixBase<U>& b){
        return A.colPivHouseholderQr().solve(b);
    }

    template<typename T, typename U>
    MatrixTU fullPivHouseholderQr(const Eigen::MatrixBase<T>& A, const Eigen::MatrixBase<U>& b){
        return A.fullPivHouseholderQr().solve(b);
    }

    template<typename T, typename U>
    MatrixTU completeOrthogonalDecomposition(const Eigen::MatrixBase<T>& A, const Eigen::MatrixBase<U>& b){
        return A.completeOrthogonalDecomposition().solve(b);
    }

} // end namespace qr

namespace cholesky {

    template<typename T, typename U>
    MatrixTU llt(const Eigen::MatrixBase<T>& A, const Eigen::MatrixBase<U>& b){
        if(A.rows()!=A.cols()){
            throw std::invalid_argument("A is not square!");
        }
        Eigen::LLT<T> llt(A);
        if(!A.isApprox(A.transpose()) || llt.info() == Eigen::NumericalIssue){
            throw std::runtime_error("Matrix A appears not to be positive definite!");
        }
        return llt.solve(b);
    }

    template<typename T, typename U>
    MatrixTU ldlt(const Eigen::MatrixBase<T>& A, const Eigen::MatrixBase<U>& b){
        if(A.rows()!=A.cols()){
            throw std::invalid_argument("A is not square!");
        }
        Eigen::LDLT<T> ldlt(A);
        if(!A.isApprox(A.transpose()) || ldlt.info() == Eigen::NumericalIssue){
            throw std::runtime_error("Matrix A appears not to be positive definite!");
        }
        return ldlt.solve(b);
    }

} // end namesapce cholesky

namespace svd {

    template<typename T, typename U>
    MatrixTU bdc(const Eigen::MatrixBase<T>& A, const Eigen::MatrixBase<U>& b){
        return A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    }  

    template<typename T, typename U>
    MatrixTU jacobi(const Eigen::MatrixBase<T>& A, const Eigen::MatrixBase<U>& b){
        return A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    }

} // end namespace svd

} // end namespace solvelin