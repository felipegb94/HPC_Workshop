# GPGPU NOTES

## Quick References
### CUDA

### OpenACC
Quickly refresh your knowledge using the quick references
[1](http://www.openacc.org/sites/default/files/OpenACC_API_QuickRefGuide.pdf) | [2](http://www.openacc.org/sites/default/files/213462%2010_OpenACC_API_QRG_HiRes.pdf)

Dive in and figure out what directives do using the full [specification](http://www.openacc.org/sites/default/files/OpenACC.2.0a_1.pdf)

## Blue Waters Prerequisites

### CUDA
On [Blue Waters](https://bluewaters.ncsa.illinois.edu/), running `module load cudatoolkit` is required to attach modules to the current login session.
###### When compiling
On [Blue Waters](https://bluewaters.ncsa.illinois.edu/), you must compile with `cc` and `nvcc`. You can also compile using `gcc` but

`nvcc -c [...]` will allow linking with other files.

### OpenACC
Same as CUDA, we have to attach the required modules in order to run OpenACC code. Run: `module load craype-accel-nvidi35`.

###### When compiling
You must compile with `cc -h pragma=acc [...]`.

## SBEL Prerequisites

## Core functionality
Cores which are running the same task *always* run in lock step.
This means operations such as branches or jumps are *permitted* but **highly** discouraged. 

##### NOTE
OpenACC does not generate many warnings that tell you if something is wrong, because it is just asking the compiler to figure out how to create kernels for the `#pragma acc` clauses. For example, if the module is not loaded or the `pragma=acc` argument is not given, the compiler will just ignore the `#pragma acc` clauses.

### CUDA
We have a CPU host and this host will send code to GPU device. This code will be under `__global__` function

![alt text](./CUDA_Diag.png "Logo Title Text 1")


#### Constructs
##### `__global__`
######Syntax
`__global__ void functionName(...)`

Many workers are created to accelerate the `parallel` region.
##### `__host__`
######Syntax

`__host__ void functionName(...)`

##### `__device__`
######Syntax

`__device__ void functionName(...)`



### OpenACC

#### Constructs
##### parallel
Many workers are created to accelerate the `parallel` region.

##### kernels
`kernels` may not branch or jump. `kernels` may also not contain other `parallel` or `kernels` regions. `kernels` act as a black box that basically tell the compiler "do whatever you think is best to accelerate this portion of the code". So all `copyin` and `copyout` operations are hidden from the user. If you think you are smarter than the compiler you can try not to use kernels.

`kernel` clauses are a great way to quickly check if there will be some kind of performance increase if a portion of the code is parallelized using the GPU.

##### data
`data` constructs define variables which must be copied to and from the device memory before and after the region.

##### loop
`loop` constructs mark a loop which will be either split into `kernels` or other clause types depending on compiler analysis unless explicitly defined.

## Common Clauses
### OpenACC

#### copy
`copy(variable[, variable])`

Denotes variables and arrays (denoted as `variables[start:end]`) which must be copied in and out of device memory.
#### copyin
`copyin(variable[, variable])`

Denotes variables and arrays (denoted as `variables[start:end]`) which must be copied into device memory.
#### copyout
`copyout(variable[, variable])`

Denotes variables and arrays (denoted as `variables[start:end]`) which must be copied out of device memory.

## Common Gotchas
Exceeding the number of threads will result in junk, but not an error (unless explicitly checked for error).

So, carefully execute work while respecting the number of available threads in a block, and the number of blocks in a grid, by splitting for overages.