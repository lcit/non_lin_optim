/*
In this example we refine the extrisic parameters of a camera against known 3d locations with corresponding image locations.
The image locations are already undistorted and the camera intrinsics are known.
*/
#include <math.h>
#include <non_lin_optim/optim.h>
#include <Eigen/Dense>

#include <iostream>
#include <stdexcept>

using namespace non_lin_optim;

Eigen::Matrix<Scalar,3,3> rotvec2mat(Vector rvec){
    
    Scalar theta = rvec.norm();
    if(theta < std::numeric_limits<Scalar>::epsilon()){
        Eigen::Matrix<Scalar,3,3> R = Eigen::Matrix<Scalar,3,3>::Identity();
        return R;
    }

    Scalar c = cos(theta);
    Scalar s = sin(theta);
    Scalar c1 = 1. - c;
    Scalar itheta = theta ? 1./theta : 0.;

    rvec = rvec*itheta;
    
    Eigen::Matrix<Scalar,3,3> rrt {
        {rvec(0)*rvec(0), rvec(0)*rvec(1), rvec(0)*rvec(2)}, 
        {rvec(0)*rvec(1), rvec(1)*rvec(1), rvec(1)*rvec(2)}, 
        {rvec(0)*rvec(2), rvec(1)*rvec(2), rvec(2)*rvec(2)}
    };
    Eigen::Matrix<Scalar,3,3> r_x {
        {0,       -rvec(2),  rvec(1)},
        {rvec(2),     0,    -rvec(0)},
        {-rvec(1), rvec(0),        0}
    };

    // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
    Eigen::Matrix<Scalar,3,3> R = c*Eigen::Matrix<Scalar,3,3>::Identity() + c1*rrt + s*r_x;
    
    return R;
}

// other way with Eigen
Eigen::Matrix<Scalar,3,3> rotvec2mat2(Vector rvec){
    
    Scalar theta = rvec.norm();
    if(theta < std::numeric_limits<Scalar>::epsilon()){
        Eigen::Matrix<Scalar,3,3> R = Eigen::Matrix<Scalar,3,3>::Identity();
        return R;
    }
    
    auto sinA = sin(theta / 2);
    auto cosA = cos(theta / 2);

    Eigen::Quaternion<Scalar> q;
    q.x() = rvec(0) * sinA / theta;
    q.y() = rvec(1) * sinA / theta;
    q.z() = rvec(2) * sinA / theta;
    q.w() = cosA;     
    
    Eigen::Matrix<Scalar,3,3> R = q.toRotationMatrix();
    return R;
}

int main(int argc, char** argv){
    
    Eigen::Matrix<Scalar,12,3> points_3d {
        { 2.63002813e+00,  -4.73064234e+00, 1.50618167e+00},
        {-3.70349293e+00,  3.58105095e+00,  7.86752797e-01},
        { 1.68537623e+00,  -1.86170394e+00, 5.30748778e-01},
        { 1.60094696e+00,   2.24767697e+00, 1.15206023e+00},
        { 3.43645179e+00,  -5.97678911e+00, 1.33917919e+00},
        {-2.09627415e+00, -2.84170710e+00,  1.46094717e+00},
        {-4.02440345e+00,  9.15705246e+00,  5.04084267e-01},
        { 7.86630456e-03,  -9.39415873e+00, 1.72543266e+00},
        { 6.31172900e+00,  -4.32363882e+00, 3.68004833e-02},
        {-1.59925565e+00, -3.84276957e+00,  1.06966253e+00},
        { 8.42769011e-01,  -5.29161544e-02, 1.98306005e+00},
        { 2.46506435e+00,  -2.72676961e+00, 1.45703557e+00}
    };
    
    Eigen::Matrix<Scalar,12,2> points_2d_undist {
        {1098.25084077, 319.19005757},
        {770.58023797,  904.74989049},
        {974.60672688,  420.7363441 },
        {720.49421605,  416.06945914},
        {1119.69792767, 301.9548112 },
        {1259.34672737, 535.38400951},
        {-306.55794983, 1371.9341184},
        {1452.21515004, 366.3347584 },
        {942.75102179,  300.49781786},
        {1274.85313494, 521.25535014},
        {914.6726082 ,  375.07103224},
        {1003.74442446, 335.1878995 }
    };
    
    Eigen::Matrix<Scalar,3,3> K {
        {866.4245,   0.     , 736.4805 },
        {0.      , 875.8444 , 174.83566},
        {0.      ,   0.     ,   1.     }
    };
    
    // initial solution computed with PnP
    // 3 rotation parameter in Rodrigues format
    // 3 translation parameters
    Vector x(6);  
    x << 2.2066754, 4.8928096, 10.064986, 1.2037287, -1.555427, 1.6341713;
    
    Vector x_gt(6);
    x_gt << 2.3066754, 4.6928096, 10.964986, 1.3037287, -1.6955427, 1.3341713;    
    
    auto reprojection_error = [&](const Vector& x) -> Vector {
        
        Vector t = x.block<3,1>(0,0);
        Vector rvec = x.block<3,1>(3,0);
        
        auto R = rotvec2mat2(rvec);
        
        Eigen::Matrix<Scalar,3,12> proj_xyw = K*((R*points_3d.transpose()).colwise() + t);
        Eigen::Matrix<Scalar,2,12> proj_uv  = (proj_xyw.block<2,12>(0,0).array().rowwise() /
                                                 proj_xyw.block<1,12>(2,0).array());
        
        Vector dists = (points_2d_undist-proj_uv.transpose()).rowwise().norm();
        
        std::cout << "Reprojection error:" << dists.mean() << std::endl;
        
        return dists;
    };  
    
    auto optimizer = optim::GaussianNewton(reprojection_error, 10000, 1e-12, 1);
    auto result = optimizer.run(x);
    
    std::cout << "x:" << x << std::endl;
    
    std::cout << result << std::endl;
    
    std::cout << "Error: " << (x_gt-x).norm() << std::endl;

    return 0;
}
