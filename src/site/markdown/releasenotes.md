finmath lib cuda extensions Release Notes
==========

****************************************

# Release Notes


## 5.1.1 (12
.03.2021)

### General

 - Created separate project splitting Cuda and OpenCL
 - Referencing finmath-lib 5.1.2
 

## 5.0.5 (29.08.2020)

### General

 - Requiring Java 11.
 - Providing a Java module defines. Module name is net.finmath.gpu
 - Referencing finmath-lib 5.0.5
 - Refactoring package names to net.finmath.cuda and net.finmath.opencl to comply with Java module system.
 

## 4.1.7 (22.03.2020)

### General

- Compatibility with finmath-lib 4.1.7.


## 4.1.1 (15.02.2020)

### General

- Extracted interface `RandomVariableDifferentiableFactory`.
- Improved support for libraries referencing finmath-lib-cuda-extensions as a Maven dependency. 
- Added examples to finmath experiments.


### Cuda

- Allowing to select the cuda version via a command line or settings.xml property (cuda.version).
- Added support for legacy versions Cuda 9.2, Cuda 6.0



## 4.0.12 (10.09.2019)

### General

- Added support for **OpenCL** via `RandomVariableOpenCLFactory` and `RandomVariableOpenCLFactory`



## 4.0.11 (31.08.2019)

### General

- Compiled for Cuda 10.1 (JCuda 10.1.10).


## 4.0.10 (31.08.2019)

### General

- Added site


## 4.0.6 (30.08.2019)

### General

- Performance improvements
- Compatibility with finmath-lib 4.0.x
- Compiled for JCuda / Cuda 10.0
- Unit Tests and Documentation.
