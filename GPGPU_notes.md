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
We have a CPU host and this host will send code to GPU device. This code will be under `__global__` function. 

![alt text](https://raw.githubusercontent.com/felipegb94/HPC_Workshop/master/images/CUDA_Diagram.png "Logo Title Text 1")


#### Constructs
##### `__global__`
######Syntax

Code to be executed in the device. This code is completely isolated from the CPU, so it can only access elements that are copied into the GPU memory. These elements are given as arguments in the function call. See

`__global__ void functionName(...)`

##### `__host__`
######Syntax

`__host__ void functionName(...)`

##### `__device__`
######Syntax

The `__device__` keyword let's us create function that are callable from threads on the device. This means that the code inside `__global__` kernels will be able to call functions that have the `__device__` keyword in their header. 

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

## Clauses and examples
### CUDA

#### Calculating area under the curve
This example estimates the area under a quarter of a circle. This is done by calculating the area of many rectangles like in the picture below. Each area is calculated by one thread in the GPU.

![alt text](https://raw.githubusercontent.com/felipegb94/HPC_Workshop/master/images/pi.JPG "Logo Title Text 1")

##### Kernel function
This is the portion of the code that will be executed inside the GPU. We will use the id of each thread to figure out which rectangle that thread will be calculating.

 ```
 __global__ void calculateAreas(const int numRects, const double width,
    double *dev_areas) {
  const int threadId = threadIdx.x;
  // Calculate the xCoordinate where the rectangle will start.
  const double x = (threadId * width);
  // Use unit circle equation to calculate height of rectangle x^2 + y^2 = 1
  const double heightSq = (1.0 - (x * x));
  const double height =
    /* Prevent nan value for sqrt() */
    (heightSq < DBL_EPSILON) ? (0.0) : (sqrt(heightSq));

  // Store value in the memory allocated. 
  if (threadId < numRects) {
    dev_areas[threadId] = (width * height);
  }
}
 ```

##### Allocate memory

```
void calculateArea(const int numRects, double *area) {
  // Allocate mem
  double *areas = (double*)malloc(numRects * sizeof(double));
  double *device_areas;
  int i = 0;
  cudaError_t err;

  if (areas == NULL) {
    fprintf(stderr, "malloc failed!\n");
  }

  err = cudaMalloc((void**)&dev_areas, (numRects * sizeof(double)));

  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
  }
``` 

##### Kernel call and finalize
Note that in the kernel call we pass in the pointer to the memory we allocated in the GPU. `<<<1, numRects>>>`, in this example `numRects` is specifying the number of threads that should be used. Since we are calculating an specific number of rectangles we onyl need that many threads since each one of them will calculate one of the areas.


```
	width = 1.0/numRects;
	// Kernel call
   calculateAreas<<<1, numRects>>>(numRects, width, device_areas);
   
   // Copy everything that was calculated in the GPU to the CPU
   err = cudaMemcpy(areas, device_areas, (numRects * sizeof(double)), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
   		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
  }

	// Sum areas
	(*area) = 0.0;
	for (i = 0; i < numRects; i++) {
   		(*area) += areas[i];
  	}
  	
  	// Free memory.
  	cudaFree(device_areas);
	free(areas);

```


#### Getting device information

Getting the device information is useful whenever we are trying to figure out the maximum number of threads the device can handle.

```
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device); 
  printf("Max number of threads per block = %i \n", prop.maxThreadsPerBlock);
```


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