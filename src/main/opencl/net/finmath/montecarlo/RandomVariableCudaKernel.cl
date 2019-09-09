__kernel void capByScalar(int n, __global const float *a, float b, __global float *result)
{
    float cap = b;
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] < cap ? a[i] : cap;
    }
}

__kernel void floorByScalar(int n, __global const float *a, float b, __global float *result)
{
    float floor = b;
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] > floor ? a[i] : floor;
    }
}

__kernel void addScalar(int n, __global const float *a, float b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] + b;
    }
}

__kernel void subScalar(int n, __global const float *a, float b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] - b;
    }
}

__kernel void busScalar(int n, __global const float *a, float b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = -a[i] + b;
    }
}

__kernel void multScalar(int n, __global const float *a, float b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] * b;
    }
    
}

__kernel void divScalar(int n, __global const float *a, float b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] / b;
    }
    
}

__kernel void vidScalar(int n, __global const float *a, float b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = b / a[i];
    }
    
}

__kernel void squared(int n, __global const float *a, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] * a[i];
    }
    
}

__kernel void cuPow(int n, __global const float *a, float b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = pow(a[i],b);
    }
    
}

__kernel void cuSqrt(int n, __global const float *a, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = sqrt(a[i]);
    }
}

__kernel void cuExp(int n, __global const float *a, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = (float)exp((double)a[i]);
    }
}

__kernel void cuLog(int n, __global const float *a, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = (float)log((double)a[i]);
    }
}

__kernel void invert(int n, __global const float *a, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = 1.0f / a[i];
    }
}

__kernel void cuAbs(int n, __global const float *a, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = fabs(a[i]);
    }
}


__kernel void cap(int n, __global const float *a, __global const float *b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] < b[i] ? a[i] : b[i];
    }
}

__kernel void cuFloor(int n, __global const float *a, __global const float *b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] > b[i] ? a[i] : b[i];
    }
}

__kernel void add(int n, __global const float *a, __global const float *b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] + b[i];
    }
    
}

__kernel void sub(int n, __global const float *a, __global const float *b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] - b[i];
    }
    
}

__kernel void mult(int n, __global const float *a, __global const float *b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] * b[i];
    }
    
}

__kernel void cuDiv(int n, __global const float *a, __global const float *b, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = (float)((double)a[i] / (double)b[i]);
    }
    
}

__kernel void accrue(int n, __global const float *a, __global const float *b, float p, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] * (1.0f + b[i] * p);
    }
}

__kernel void discount(int n, __global const float *a, __global const float *b, float p, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        // We force to avoid fma
        float prod = b[i] * p;
        float fma = (1.0f + prod);
        result[i] = a[i] / fma;
    }
}

__kernel void addProduct(int n, __global const float *a, __global const float *b, __global const float *c, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] + b[i] * c[i];
    }
}

__kernel void addProduct_vs(int n, __global const float *a, __global const float *b, float c, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] + b[i] * c;
    }
}

__kernel void addRatio(int n, __global const float *a, __global const float *b, __global const float *c, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] + b[i] / c[i];
    }
}

__kernel void subRatio(int n, __global const float *a, __global const float *b, __global const float *c, __global float *result)
{
    int i = get_global_id(0);
    if (i<n)
    {
        result[i] = a[i] - b[i] / c[i];
    }
}
