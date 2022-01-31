# non_lin_optim

Header only library for non-linear optimization using Eigen. I wrote this library to practice modern C++ and to rehearse optimization theory.

## Features

- Numerical derivates (Jacobian, Hessian)
- Newton-Raphson
- Gaussian-Newton
- Gradient descent
- (TODO) Levenberg-Marquard

## Usage

```cpp
#include <non_lin_optim/optim.h>
#include <iostream>

using namespace non_lin_optim;  
    
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
std::cout << x << std::endl;
```
```cpp
Matrix J = numerical_deriv::JacobianApprox(func, x);
std::cout << J << std::endl;

Matrix H = numerical_deriv::HessianApprox(func_s, x);
std::cout << H << std::endl;
```
    
### Build and run the examples

Use the following command give an example of how to build and run an example.

```bash
cmake -S examples -B build/examples/refine_camera_extrinsics
cmake --build build/refine_camera_extrinsics
./build/refine_camera_extrinsics/refine_camera_extrinsics --help
```

### Build and run test suite

Use the following commands from the project's root directory to run the test suite.

```bash
cmake -S test -B build/test
cmake --build build/test
CTEST_OUTPUT_ON_FAILURE=1 cmake --build build/test --target test

# or simply call the executable: 
./build/test/Tests
```

To collect code coverage information, run CMake with the `-DENABLE_TEST_COVERAGE=1` option.

### Run clang-format

Use the following commands from the project's root directory to check and fix C++ and CMake source style.
This requires _clang-format_, _cmake-format_ and _pyyaml_ to be installed on the current system.

```bash
cmake -S test -B build/test

# view changes
cmake --build build/test --target format

# apply changes
cmake --build build/test --target fix-format
```

See [Format.cmake](https://github.com/TheLartians/Format.cmake) for details.

### Build the documentation

The documentation is automatically built and [published](https://thelartians.github.io/ModernCppStarter) whenever a [GitHub Release](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository) is created.
To manually build documentation, call the following command.

```bash
cmake -S documentation -B build/doc
cmake --build build/doc --target GenerateDocs
# view the docs
open build/doc/doxygen/html/index.html
```

To build the documentation locally, you will need Doxygen, jinja2 and Pygments on installed your system.
