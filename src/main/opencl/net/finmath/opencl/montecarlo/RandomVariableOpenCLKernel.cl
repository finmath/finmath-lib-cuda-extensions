__kernel void capByScalar(__global const float *a, float b, __global float *result)
{
    float cap = b;
    size_t i = get_global_id(0);

    result[i] = a[i] < cap ? a[i] : cap;
}

__kernel void floorByScalar(__global const float *a, float b, __global float *result)
{
    float floor = b;
    size_t i = get_global_id(0);

    result[i] = a[i] > floor ? a[i] : floor;
}

__kernel void addScalar(__global const float *a, float b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] + b;
}

__kernel void subScalar(__global const float *a, float b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] - b;
}

__kernel void busScalar(__global const float *a, float b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = -a[i] + b;
}

__kernel void multScalar(__global const float *a, float b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] * b;
}

__kernel void divScalar(__global const float *a, float b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] / b;
}

__kernel void vidScalar(__global const float *a, float b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = b / a[i];
}

__kernel void squared(__global const float *a, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] * a[i];
}

__kernel void cuPow(__global const float *a, float b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = pow(a[i],b);
}

__kernel void cuSqrt(__global const float *a, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = sqrt(a[i]);
}

__kernel void cuExp(__global const float *a, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = exp(a[i]);
}

__kernel void cuLog(__global const float *a, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = log(a[i]);
}

__kernel void invert(__global const float *a, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = 1.0f / a[i];
}

__kernel void cuAbs(__global const float *a, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = fabs(a[i]);
}


__kernel void cap(__global const float *a, __global const float *b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] < b[i] ? a[i] : b[i];
}

__kernel void cuFloor(__global const float *a, __global const float *b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] > b[i] ? a[i] : b[i];
}

__kernel void add(__global const float *a, __global const float *b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] + b[i];
}

__kernel void sub(__global const float *a, __global const float *b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] - b[i];
}

__kernel void mult(__global const float *a, __global const float *b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] * b[i];
}

__kernel void cuDiv(__global const float *a, __global const float *b, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = (float)((double)a[i] / (double)b[i]);
}

__kernel void accrue(__global const float *a, __global const float *b, float p, __global float *result)
{
    size_t i = get_global_id(0);

    // We force to avoid fma
    float prod = b[i] * p;
    float fma = (1.0f + prod);
    result[i] = a[i] * fma;
}

__kernel void discount(__global const float *a, __global const float *b, float p, __global float *result)
{
    size_t i = get_global_id(0);

    // We force to avoid fma
    float prod = b[i] * p;
    float fma = (1.0f + prod);
    result[i] = a[i] / fma;
}

__kernel void addProduct(__global const float *a, __global const float *b, __global const float *c, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] + b[i] * c[i];
}

__kernel void addProduct_vs(__global const float *a, __global const float *b, float c, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] + b[i] * c;
}

__kernel void addRatio(__global const float *a, __global const float *b, __global const float *c, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] + b[i] / c[i];
}

__kernel void subRatio(__global const float *a, __global const float *b, __global const float *c, __global float *result)
{
    size_t i = get_global_id(0);

    result[i] = a[i] - b[i] / c[i];
}

__kernel void reduceFloatVectorToDoubleScalar(__global float* inVector, __global float* outVector, const int inVectorSize, __local float* resultScratch){
    int gid = get_global_id(0);
    int wid = get_local_id(0);
    int wsize = get_local_size(0);
    int grid = get_group_id(0);
    int grcount = get_num_groups(0);

    int i;
    int workAmount = inVectorSize/grcount;
    int startOffest = workAmount * grid + wid;
    int maxOffset = workAmount * (grid + 1);
    if(maxOffset > inVectorSize){
        maxOffset = inVectorSize;
    }
    resultScratch[wid] = 0.0;
    for(i=startOffest;i<maxOffset;i+=wsize){
            resultScratch[wid] += inVector[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(gid == 0){
            for(i=1;i<wsize;i++){
                    resultScratch[0] += resultScratch[i];
            }
            outVector[grid] = resultScratch[0];
    }
}
