extern "C"
__global__ void capByScalar(int n, float *a, float b, float *result)
{
	float cap = b;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] < cap ? a[i] : cap;
    }
}

extern "C"
__global__ void floorByScalar(int n, float *a, float b, float *result)
{
	float floor = b;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] > floor ? a[i] : floor;
    }
}

extern "C"
__global__ void addScalar(int n, float *a, float b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] + b;
    }
}

extern "C"
__global__ void subScalar(int n, float *a, float b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] - b;
    }
}

extern "C"
__global__ void busScalar(int n, float *a, float b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = -a[i] + b;
    }
}

extern "C"
__global__ void multScalar(int n, float *a, float b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] * b;
    }

}

extern "C"
__global__ void divScalar(int n, float *a, float b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] / b;
    }

}

extern "C"
__global__ void vidScalar(int n, float *a, float b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = b / a[i];
    }

}

extern "C"
__global__ void squared(int n, float *a, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] * a[i];
    }

}

extern "C"
__global__ void cuPow(int n, float *a, float b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = pow(a[i],b);
    }

}

extern "C"
__global__ void cuSqrt(int n, float *a, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = sqrt(a[i]);
    }
}

extern "C"
__global__ void cuExp(int n, float *a, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = (float)exp((double)a[i]);
    }
}

extern "C"
__global__ void cuLog(int n, float *a, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = (float)log((double)a[i]);
    }
}

extern "C"
__global__ void invert(int n, float *a, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = 1.0f / a[i];
    }
}

extern "C"
__global__ void cuAbs(int n, float *a, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = abs(a[i]);
    }
}


extern "C"
__global__ void cap(int n, float *a, float *b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] < b[i] ? a[i] : b[i];
    }
}

extern "C"
__global__ void cuFloor(int n, float *a, float *b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] > b[i] ? a[i] : b[i];
    }
}

extern "C"
__global__ void add(int n, float *a, float *b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] + b[i];
    }

}

extern "C"
__global__ void sub(int n, float *a, float *b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] - b[i];
    }

}

extern "C"
__global__ void mult(int n, float *a, float *b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] * b[i];
    }

}

extern "C"
__global__ void cuDiv(int n, float *a, float *b, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] / b[i];
    }

}

extern "C"
__global__ void accrue(int n, float *a, float *b, float p, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] * (1.0f + b[i] * p);
    }
}

extern "C"
__global__ void discount(int n, float *a, float *b, float p, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        // We force to avoid fma
        float prod = b[i] * p;
        float fma = (1.0f + prod);
        result[i] = a[i] / fma;
    }
}

extern "C"
__global__ void addProduct(int n, float *a, float *b, float *c, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] + b[i] * c[i];
    }
}

extern "C"
__global__ void addProduct_vs(int n, float *a, float *b, float c, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] + b[i] * c;
    }
}

extern "C"
__global__ void addRatio(int n, float *a, float *b, float *c, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] + b[i] / c[i];
    }
}

extern "C"
__global__ void subRatio(int n, float *a, float *b, float *c, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = a[i] - b[i] / c[i];
    }
}

extern "C"
__global__ void reduceFloatVectorToDoubleScalar(int size, void *data, double *result) {
	float *fdata = (float*) data;

	extern __shared__ double sdata[];
	double* s2data = sdata + blockDim.x;
	double* cdata = s2data + blockDim.x;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory unsigned int tid = threadIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

	sdata[tid] = (double)(i < size ? fdata[i] : 0) + (double)(i+blockDim.x < size ? fdata[i+blockDim.x] : 0);
	cdata[tid] = sdata[tid] - (double)(i < size ? fdata[i] : 0) - (double)(i+blockDim.x < size ? fdata[i+blockDim.x] : 0);
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			s2data[tid] = sdata[tid] + sdata[tid + s] - cdata[tid] - cdata[tid+s];
			cdata[tid] = (s2data[tid] - sdata[tid]) - sdata[tid + s];
			sdata[tid] = s2data[tid];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) result[blockIdx.x] = sdata[0];
}

/*
 * Perfom a reduction from data of length 'size' to result, where length of result will be 'number of blocks'.
 */ 
extern "C"
__global__ void reducePartial(int size, void *data, void *result) {
	float *fdata = (float*) data;
	float *sum = (float*) result;

	extern __shared__ double sdata[];
	double* s2data = sdata + blockDim.x;
	double* cdata = s2data + blockDim.x;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory unsigned int tid = threadIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = (double)(i < size ? fdata[i] : 0) + (double)(i+blockDim.x < size ? fdata[i+blockDim.x] : 0);
	cdata[tid] = sdata[tid] - (double)(i < size ? fdata[i] : 0) - (double)(i+blockDim.x < size ? fdata[i+blockDim.x] : 0);
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			s2data[tid] = sdata[tid] + sdata[tid + s] - cdata[tid] - cdata[tid+s];
			cdata[tid] = (s2data[tid] - sdata[tid]) - sdata[tid + s];
			sdata[tid] = s2data[tid];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) sum[blockIdx.x] = sdata[0];
}
