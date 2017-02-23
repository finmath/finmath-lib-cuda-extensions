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
        result[i] = exp(a[i]);
    }

}

extern "C"
__global__ void cuLog(int n, float *a, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = log(a[i]);
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
        result[i] = a[i] / (1.0f + b[i] * p);
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
