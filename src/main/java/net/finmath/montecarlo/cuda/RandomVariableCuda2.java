/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 09.02.2006
 */
package net.finmath.montecarlo.cuda;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.IOException;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;

import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import net.finmath.functions.DoubleTernaryOperator;
import net.finmath.stochastic.RandomVariable;

/**
 * The class RandomVariableFromDoubleArray represents a random variable being the evaluation of a stochastic process
 * at a certain time within a Monte-Carlo simulation.
 * It is thus essentially a vector of floating point numbers - the realizations - together with a double - the time.
 * The index of the vector represents path.
 * The class may also be used for non-stochastic quantities which may potentially be stochastic
 * (e.g. volatility). If only non-stochastic random variables are involved in an operation the class uses
 * optimized code.
 *
 * Accesses performed exclusively through the interface
 * <code>RandomVariable</code> is thread safe (and does not mutate the class).
 *
 * This implementation uses floats for the realizations (consuming less memory compared to using doubles). However,
 * the calculation of the average is performed using double precision.
 *
 * @author Christian Fries
 * @version 1.8
 */
public class RandomVariableCuda2 implements RandomVariable {

	private static final long serialVersionUID = 7620120320663270600L;

	private final double      time;	                // Time (filtration)

	// Data model for the stochastic case (otherwise null)
	private final CUdeviceptr	realizations;           // Realizations
	private final long			size;

	// Data model for the non-stochastic case (if realizations==null)
	private final double      valueIfNonStochastic;

	// @TODO: Support slices with different sizes! Handler need to check for size
	private final static Map<Integer, ReferenceQueue<RandomVariableCuda2>>		vectorsToRecycleReferenceQueueMap	= new ConcurrentHashMap<Integer, ReferenceQueue<RandomVariableCuda2>>();
	private final static Map<WeakReference<RandomVariableCuda2>, CUdeviceptr>	vectorsInUseReferenceMap			= new ConcurrentHashMap<WeakReference<RandomVariableCuda2>, CUdeviceptr>();
	private final static float	vectorsRecyclerPercentageFreeToStartGC		= 0.10f;		// should be set by monitoring GPU mem
	private final static float	vectorsRecyclerPercentageFreeToWaitForGC	= 0.05f;		// should be set by monitoring GPU mem
	private final static long	vectorsRecyclerMaxTimeOutMillis			= 100;

	private final static Logger logger = Logger.getLogger("net.finmath");

	private final static ExecutorService deviceExecutor = Executors.newSingleThreadExecutor();
	public final static CUdevice device = new CUdevice();
	public final static CUcontext context = new CUcontext();
	public final static CUmodule module = new CUmodule();

	private final static CUfunction capByScalar = new CUfunction();
	private final static CUfunction floorByScalar = new CUfunction();
	private final static CUfunction addScalar = new CUfunction();
	private final static CUfunction subScalar = new CUfunction();
	private final static CUfunction busScalar = new CUfunction();
	private final static CUfunction multScalar = new CUfunction();
	private final static CUfunction divScalar = new CUfunction();
	private final static CUfunction vidScalar = new CUfunction();
	private final static CUfunction cuPow = new CUfunction();
	private final static CUfunction cuSqrt = new CUfunction();
	private final static CUfunction cuExp = new CUfunction();
	private final static CUfunction cuLog = new CUfunction();
	private final static CUfunction invert = new CUfunction();
	private final static CUfunction cuAbs = new CUfunction();
	private final static CUfunction cap = new CUfunction();
	private final static CUfunction cuFloor = new CUfunction();
	private final static CUfunction add = new CUfunction();
	private final static CUfunction sub = new CUfunction();
	private final static CUfunction mult = new CUfunction();
	private final static CUfunction cuDiv = new CUfunction();
	private final static CUfunction accrue = new CUfunction();
	private final static CUfunction discount = new CUfunction();
	private final static CUfunction addProduct = new CUfunction();
	private final static CUfunction addProduct_vs = new CUfunction();		// add the product of a vector and a scalar
	private final static CUfunction reducePartial = new CUfunction();

	private final static int reduceGridSize = 1024;

	// Initalize cuda
	static {
		synchronized (vectorsInUseReferenceMap) {
			// Enable exceptions and omit all subsequent error checks
			JCudaDriver.setExceptionsEnabled(true);
			JCudaDriver.setLogLevel(LogLevel.LOG_DEBUG);

			// Create the PTX file by calling the NVCC
			String ptxFileName = null;
			try {
				URL cuFileURL = RandomVariableCuda2.class.getClassLoader().getResource("net/finmath/montecarlo/RandomVariableCudaKernel.cu");
				ptxFileName = net.finmath.jcuda.JCudaUtils.preparePtxFile(cuFileURL);
			} catch (IOException | URISyntaxException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			final String ptxFileName2 = ptxFileName;
			deviceExecutor.submit(new Runnable() { public void run() {
				// Initialize the driver and create a context for the first device.
				cuInit(0);
				cuDeviceGet(device, 0);
				cuCtxCreate(context, 0, device);

				// Load the ptx file.
				cuModuleLoad(module, ptxFileName2);

				// Obtain a function pointers
				cuModuleGetFunction(capByScalar, module, "capByScalar");
				cuModuleGetFunction(floorByScalar, module, "floorByScalar");
				cuModuleGetFunction(addScalar, module, "addScalar");
				cuModuleGetFunction(subScalar, module, "subScalar");
				cuModuleGetFunction(busScalar, module, "busScalar");
				cuModuleGetFunction(multScalar, module, "multScalar");
				cuModuleGetFunction(divScalar, module, "divScalar");
				cuModuleGetFunction(vidScalar, module, "vidScalar");
				cuModuleGetFunction(cuPow, module, "cuPow");
				cuModuleGetFunction(cuSqrt, module, "cuSqrt");
				cuModuleGetFunction(cuExp, module, "cuExp");
				cuModuleGetFunction(cuLog, module, "cuLog");
				cuModuleGetFunction(invert, module, "invert");
				cuModuleGetFunction(cuAbs, module, "cuAbs");
				cuModuleGetFunction(cap, module, "cap");
				cuModuleGetFunction(cuFloor, module, "cuFloor");
				cuModuleGetFunction(add, module, "add");
				cuModuleGetFunction(sub, module, "sub");
				cuModuleGetFunction(mult, module, "mult");
				cuModuleGetFunction(cuDiv, module, "cuDiv");
				cuModuleGetFunction(accrue, module, "accrue");
				cuModuleGetFunction(discount, module, "discount");
				cuModuleGetFunction(addProduct, module, "addProduct");
				cuModuleGetFunction(addProduct_vs, module, "addProduct_vs");
				cuModuleGetFunction(reducePartial, module, "reducePartial");
			}});
		}
	}

	public RandomVariableCuda2(double time, CUdeviceptr realizations, long size) {
		this.time = time;
		this.realizations = realizations;
		this.size = size;
		this.valueIfNonStochastic = Double.NaN;

		synchronized (vectorsToRecycleReferenceQueueMap)
		{
			// Manage CUdeviceptr
			ReferenceQueue<RandomVariableCuda2> vectorsToRecycleReferenceQueue = vectorsToRecycleReferenceQueueMap.get(new Integer((int)size));
			if(vectorsToRecycleReferenceQueue == null) {
				vectorsToRecycleReferenceQueueMap.put(new Integer((int)size), vectorsToRecycleReferenceQueue = new ReferenceQueue<RandomVariableCuda2>());
			}
			WeakReference<RandomVariableCuda2> reference = new WeakReference<RandomVariableCuda2>(this, vectorsToRecycleReferenceQueue);
			vectorsInUseReferenceMap.put(reference, realizations);
		}
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param value the value, a constant.
	 */
	public RandomVariableCuda2(double value) {
		this(-Double.MAX_VALUE, value);
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param value the value, a constant.
	 */
	public RandomVariableCuda2(double time, double value) {
		super();
		this.time = time;
		this.realizations = null;
		this.size = 1;
		this.valueIfNonStochastic = value;
	}

	/**
	 * Create a stochastic random variable.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param realisations the vector of realizations.
	 */
	public RandomVariableCuda2(double time, float[] realisations) {
		super();
		this.time = time;
		this.size = realisations.length;
		this.realizations = createCUdeviceptr(realisations);
		this.valueIfNonStochastic = Double.NaN;
	}

	/**
	 * Create a stochastic random variable.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param realisations the vector of realizations.
	 */
	public RandomVariableCuda2(double time, double[] realisations) {
		this(time, getFloatArray(realisations));
	}

	/**
	 * Create a stochastic random variable.
	 *
	 * @param realisations the vector of realizations.
	 */
	public RandomVariableCuda2(float[] realisations) {
		this(0.0, realisations);
	}


	public static CUdeviceptr getCUdeviceptr(final long size) {
		CUdeviceptr cuDevicePtr = null;
		synchronized (vectorsInUseReferenceMap) {
			if(false) {
			// Check for object to recycle
			ReferenceQueue<RandomVariableCuda2> vectorsToRecycleReferenceQueue = vectorsToRecycleReferenceQueueMap.get(new Integer((int)size));
			if(vectorsToRecycleReferenceQueue == null) {
				vectorsToRecycleReferenceQueueMap.put(new Integer((int)size), vectorsToRecycleReferenceQueue = new ReferenceQueue<RandomVariableCuda2>());
			}
			Reference<? extends RandomVariableCuda2> reference = vectorsToRecycleReferenceQueue.poll();
			if(reference != null) {
				cuDevicePtr = vectorsInUseReferenceMap.remove(reference);
				logger.fine("Recycling device pointer " + cuDevicePtr + " from " + reference);
				return cuDevicePtr;
			}

			float deviceFreeMemPercentage = getDeviceFreeMemPercentage();
			// No pointer found, try GC if we are above a critical level
			if(reference == null && deviceFreeMemPercentage < vectorsRecyclerPercentageFreeToStartGC) {
				try {
					System.gc();
					reference = vectorsToRecycleReferenceQueue.remove(1);
				} catch (IllegalArgumentException | InterruptedException e) {}
			}

			// Wait for GC
			if(reference == null && deviceFreeMemPercentage < vectorsRecyclerPercentageFreeToWaitForGC) {

				/*
				 * Try to obtain a reference after GC, retry with waits for 1 ms, 10 ms, 100 ms, ...
				 */
				System.gc();
				long timeOut = 1;
				while(reference == null && timeOut < vectorsRecyclerMaxTimeOutMillis) {
					try {
						//					System.err.println("Wait" + timeOut);
						reference = vectorsToRecycleReferenceQueue.remove(timeOut);
						timeOut *= 10;
					} catch (IllegalArgumentException | InterruptedException e) {}
				}

				// Still no pointer found for requested size, consider cleaning all (also other sizes)
				if(reference == null) {
					clean();
				}
			}

			if(reference != null) {
				logger.fine("Recycling device pointer " + cuDevicePtr + " from " + reference);
				cuDevicePtr = vectorsInUseReferenceMap.remove(reference);
			}
			}

			if(cuDevicePtr != null) return cuDevicePtr;

			// Still no pointer found, create new one
			try {
				cuDevicePtr =
						deviceExecutor.submit(new Callable<CUdeviceptr>() { public CUdeviceptr call() {
							CUdeviceptr cuDevicePtr =
									new CUdeviceptr();
							int succ = JCudaDriver.cuMemAlloc(cuDevicePtr, size * Sizeof.FLOAT);
							if(succ != 0) {
								cuDevicePtr = null;
								logger.finest("Failed creating device vector "+ cuDevicePtr + " with size=" + size);
							}
							else {
								logger.finest("Creating device vector "+ cuDevicePtr + " with size=" + size);
							}
							return cuDevicePtr;
						}}).get();
			} catch (InterruptedException | ExecutionException e) {
				System.out.println("Failed to allocate device vector with size=" + size);
				throw new RuntimeException(e.getCause());
			}

			if(cuDevicePtr == null) {
				System.out.println("Failed to allocate device vector with size=" + size);
				throw new OutOfMemoryError("Failed to allocate device vector with size=" + size);
			}

			return cuDevicePtr;
		}
	}

	private static RandomVariableCuda2 getRandomVariableCuda(RandomVariable randomVariable) {
		if(randomVariable instanceof RandomVariableCuda2) return (RandomVariableCuda2)randomVariable;
		else return new RandomVariableCuda2(randomVariable.getFiltrationTime(), randomVariable.getRealizations());
	}
	/**
	 * @return
	 */
	private static float getDeviceFreeMemPercentage() {
		//		synchronized (vectorsInUseReferenceMap)
		{
			long[] free = new long[1];
			long[] total = new long[1];
			jcuda.runtime.JCuda.cudaMemGetInfo(free, total);
			float freeRate = ((float)free[0]/(total[0]));
			return freeRate;
		}
	}

	private CUdeviceptr createCUdeviceptr(long size) {
		synchronized (vectorsInUseReferenceMap) {
			CUdeviceptr cuDevicePtr = getCUdeviceptr(size);

			// Manage CUdeviceptr
			ReferenceQueue<RandomVariableCuda2> vectorsToRecycleReferenceQueue = vectorsToRecycleReferenceQueueMap.get(new Integer((int)size));
			WeakReference<RandomVariableCuda2> reference = new WeakReference<RandomVariableCuda2>(this, vectorsToRecycleReferenceQueue);
			vectorsInUseReferenceMap.put(reference, cuDevicePtr);

			return cuDevicePtr;
		}
	}

	/**
	 * Create a vector on device and copy host vector to it.
	 *
	 * @param values Host vector.
	 * @return Pointer to device vector.
	 */
	private CUdeviceptr createCUdeviceptr(final float[] values) {
		synchronized (vectorsInUseReferenceMap) {
			final CUdeviceptr cuDevicePtr = createCUdeviceptr((long)values.length);

			try {
				deviceExecutor.submit(new Runnable() { public void run() {
					JCudaDriver.cuMemcpyHtoD(cuDevicePtr, Pointer.to(values), (long)values.length * Sizeof.FLOAT);
				}}).get();
			} catch (InterruptedException | ExecutionException e) { throw new RuntimeException(e.getCause()); }
			return cuDevicePtr;
		}
	}

	public static void clean() {
		// Clean up all remaining pointers
		for(ReferenceQueue<RandomVariableCuda2> vectorsToRecycleReferenceQueue : vectorsToRecycleReferenceQueueMap.values()) {
			Reference<? extends RandomVariableCuda2> reference;
			while((reference = vectorsToRecycleReferenceQueue.poll()) != null) {
				final CUdeviceptr cuDevicePtr = vectorsInUseReferenceMap.remove(reference);
				logger.fine("Freeing device pointer " + cuDevicePtr + " from " + reference);
				try {
					deviceExecutor.submit(new Runnable() { public void run() {
						JCudaDriver.cuMemFree(cuDevicePtr);
					}}).get();
				} catch (InterruptedException | ExecutionException e) { throw new RuntimeException(e.getCause()); }
			}
		}
	}

	private static float[] getFloatArray(double[] arrayOfDouble) {
		float[] arrayOfFloat = new float[arrayOfDouble.length];
		for(int i=0; i<arrayOfDouble.length; i++) {
			arrayOfFloat[i] = (float)arrayOfDouble[i];
		}
		return arrayOfFloat;
	}

	private static double[] getDoubleArray(float[] arrayOfFloat) {
		double[] arrayOfDouble = new double[arrayOfFloat.length];
		for(int i=0; i<arrayOfFloat.length; i++) {
			arrayOfDouble[i] = arrayOfFloat[i];
		}
		return arrayOfDouble;
	}

	@Override
	public boolean equals(RandomVariable randomVariable) {
		throw new UnsupportedOperationException();
		/*
		if(this.time != randomVariable.getFiltrationTime()) return false;
		if(this.isDeterministic() && randomVariable.isDeterministic()) {
			return this.valueIfNonStochastic == randomVariable.get(0);
		}

		if(this.isDeterministic() != randomVariable.isDeterministic()) return false;

		for(int i=0; i<realizations.length; i++) if(realizations[i] != randomVariable.get(i)) return false;

		return true;
		 */
	}

	@Override
	public double getFiltrationTime() {
		return time;
	}

	@Override
	public int getTypePriority() {
		return 20;
	}

	@Override
	public double get(int pathOrState) {
		if(isDeterministic())   return valueIfNonStochastic;
		else               		throw new UnsupportedOperationException();
	}

	@Override
	public int size() {
		if(isDeterministic())    return 1;
		else                     return (int)this.size;
	}

	@Override
	public double getMin() {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic()) return valueIfNonStochastic;
		double min = Double.MAX_VALUE;
		if(realizations.length != 0) min = realizations[0];     /// @see getMax()
		for(int i=0; i<realizations.length; i++) min = Math.min(realizations[i],min);
		return min;
		 */
	}

	@Override
	public double getMax() {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic()) return valueIfNonStochastic;
		double max = -Double.MAX_VALUE;
		if(realizations.length != 0) max = realizations[0];
		for(int i=0; i<realizations.length; i++) max = Math.max(realizations[i],max);
		return max;
		 */
	}

	@Override
	public double getAverage() {
		if(isDeterministic())	return valueIfNonStochastic;
		if(size() == 0)			return Double.NaN;

		return  reduce()/size();
	}

	@Override
	public double getAverage(RandomVariable probabilities) {
		return this.mult(probabilities).getAverage();
	}

	@Override
	public double getVariance() {
		if(isDeterministic())	return 0.0;
		if(size() == 0)			return Double.NaN;

		double average = getAverage();
		return this.squared().getAverage() - average*average;
	}

	@Override
	public double getVariance(RandomVariable probabilities) {
		double average = getAverage(probabilities);
		return this.squared().sub(average*average).getAverage(probabilities);
	}

	@Override
	public double getSampleVariance() {
		if(isDeterministic() || size() == 1)	return 0.0;
		if(size() == 0)							return Double.NaN;

		return getVariance() * size()/(size()-1);
	}

	@Override
	public double getStandardDeviation() {
		if(isDeterministic())	return 0.0;
		if(size() == 0)			return Double.NaN;

		return Math.sqrt(getVariance());
	}

	@Override
	public double getStandardDeviation(RandomVariable probabilities) {
		if(isDeterministic())	return 0.0;
		if(size() == 0)			return Double.NaN;

		return Math.sqrt(getVariance(probabilities));
	}

	@Override
	public double getStandardError() {
		if(isDeterministic())	return 0.0;
		if(size() == 0)			return Double.NaN;

		return getStandardDeviation()/Math.sqrt(size());
	}

	@Override
	public double getStandardError(RandomVariable probabilities) {
		if(isDeterministic())	return 0.0;
		if(size() == 0)			return Double.NaN;

		return getStandardDeviation(probabilities)/Math.sqrt(size());
	}

	@Override
	public double getQuantile(double quantile) {
		if(isDeterministic())	return valueIfNonStochastic;
		if(size() == 0)			return Double.NaN;

		throw new UnsupportedOperationException();
		/*
		float[] realizationsSorted = realizations.clone();
		java.util.Arrays.sort(realizationsSorted);

		int indexOfQuantileValue = Math.min(Math.max((int)Math.round((size()+1) * (1-quantile) - 1), 0), size()-1);

		return realizationsSorted[indexOfQuantileValue];
		 */
	}

	@Override
	public double getQuantile(double quantile, RandomVariable probabilities) {
		if(isDeterministic())	return valueIfNonStochastic;
		if(size() == 0)			return Double.NaN;

		throw new RuntimeException("Method not implemented.");
	}

	@Override
	public double getQuantileExpectation(double quantileStart, double quantileEnd) {
		if(isDeterministic())	return valueIfNonStochastic;
		if(size() == 0)			return Double.NaN;
		if(quantileStart > quantileEnd) return getQuantileExpectation(quantileEnd, quantileStart);

		throw new UnsupportedOperationException();
		/*
		float[] realizationsSorted = realizations.clone();
		java.util.Arrays.sort(realizationsSorted);

		int indexOfQuantileValueStart	= Math.min(Math.max((int)Math.round((size()+1) * quantileStart - 1), 0), size()-1);
		int indexOfQuantileValueEnd		= Math.min(Math.max((int)Math.round((size()+1) * quantileEnd - 1), 0), size()-1);

		double quantileExpectation = 0.0;
		for (int i=indexOfQuantileValueStart; i<=indexOfQuantileValueEnd;i++) {
			quantileExpectation += realizationsSorted[i];
		}
		quantileExpectation /= indexOfQuantileValueEnd-indexOfQuantileValueStart+1;

		return quantileExpectation;
		 */
	}

	@Override
	public double[] getHistogram(double[] intervalPoints)
	{
		throw new UnsupportedOperationException();
		/*
		double[] histogramValues = new double[intervalPoints.length+1];

		if(isDeterministic()) {
			java.util.Arrays.fill(histogramValues, 0.0);
			for (int intervalIndex=0; intervalIndex<intervalPoints.length; intervalIndex++)
			{
				if(valueIfNonStochastic > intervalPoints[intervalIndex]) {
					histogramValues[intervalIndex] = 1.0;
					break;
				}
			}
			histogramValues[intervalPoints.length] = 1.0;
		}
		else {
			float[] realizationsSorted = realizations.clone();
			java.util.Arrays.sort(realizationsSorted);

			int sampleIndex=0;
			for (int intervalIndex=0; intervalIndex<intervalPoints.length; intervalIndex++)
			{
				int sampleCount = 0;
				while (sampleIndex < realizationsSorted.length &&
						realizationsSorted[sampleIndex] <= intervalPoints[intervalIndex])
				{
					sampleIndex++;
					sampleCount++;
				}
				histogramValues[intervalIndex] = sampleCount;
			}
			histogramValues[intervalPoints.length] = realizationsSorted.length-sampleIndex;

			// Normalize histogramValues
			if(realizationsSorted.length > 0) {
				for(int i=0; i<histogramValues.length; i++) histogramValues[i] /= realizationsSorted.length;
			}
		}

		return histogramValues;
		 */
	}

	@Override
	public double[][] getHistogram(int numberOfPoints, double standardDeviations) {
		throw new UnsupportedOperationException();
		/*
		double[] intervalPoints = new double[numberOfPoints];
		double[] anchorPoints	= new double[numberOfPoints+1];
		double center	= getAverage();
		double radius	= standardDeviations * getStandardDeviation();
		double stepSize	= (double) (numberOfPoints-1) / 2.0;
		for(int i=0; i<numberOfPoints;i++) {
			double alpha = (-(double)(numberOfPoints-1) / 2.0 + (double)i) / stepSize;
			intervalPoints[i]	= center + alpha * radius;
			anchorPoints[i]		= center + alpha * radius - radius / (2 * stepSize);
		}
		anchorPoints[numberOfPoints] = center + 1 * radius + radius / (2 * stepSize);

		double[][] result = new double[2][];
		result[0] = anchorPoints;
		result[1] = getHistogram(intervalPoints);

		return result;
		 */
	}

	@Override
	public boolean isDeterministic() {
		return realizations == null;
	}

	public RandomVariable expand(int numberOfPaths) {
		if(isDeterministic()) {
			// Expand random variable to a vector of path values
			float[] clone = new float[numberOfPaths];
			java.util.Arrays.fill(clone,(float)valueIfNonStochastic);
			return new RandomVariableCuda2(time,clone);
		}

		return this;
	}

	@Override
	public RandomVariable cache() {
		return this;
		/*
		final float[] values = new float[(int)size];
		try {
			deviceExecutor.submit(new Runnable() { public void run() {
				cuCtxSynchronize();
				cuMemcpyDtoH(Pointer.to(values), realizations, size * Sizeof.FLOAT);
				cuCtxSynchronize();
			}}).get();
		} catch (InterruptedException | ExecutionException e) { throw new RuntimeException(e.getCause()); }
		return new RandomVariableFromFloatArray(time, values);
		 */
	}

	@Override
	public double[] getRealizations() {
		if(isDeterministic()) {
			double[] result = new double[1];
			result[0] = get(0);
			return result;
		}
		else {
			final float[] result = new float[(int)size];
			try {
				deviceExecutor.submit(new Runnable() { public void run() {
					cuCtxSynchronize();
					cuMemcpyDtoH(Pointer.to(result), realizations, size * Sizeof.FLOAT);
					cuCtxSynchronize();
				}}).get();
			} catch (InterruptedException | ExecutionException e) { throw new RuntimeException(e.getCause()); }
			return getDoubleArray(result);
		}
	}

	@Override
	public Double doubleValue() {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		} else {
			throw new UnsupportedOperationException("The random variable is non-deterministic");
		}
	}

	@Override
	public RandomVariable apply(DoubleUnaryOperator function) {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic()) {
			double newValueIfNonStochastic = function.value(valueIfNonStochastic);
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			double[] newRealizations = new double[realizations.length];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = function.value(realizations[i]);
			return new RandomVariableCuda(time, newRealizations);
		}
		 */
	}

	@Override
	public RandomVariable apply(DoubleBinaryOperator operator, RandomVariable argument) {
		throw new UnsupportedOperationException();
	}

	@Override
	public RandomVariable apply(DoubleTernaryOperator operator, RandomVariable argument1, RandomVariable argument2) {
		throw new UnsupportedOperationException();
	}

	public RandomVariable cap(double cap) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.min(valueIfNonStochastic,cap);
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(capByScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)cap }),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable floor(double floor) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.max(valueIfNonStochastic,floor);
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(floorByScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)floor }),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable add(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic + value;
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(addScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable sub(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic - value;
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(subScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable mult(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * value;
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(multScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { ((float)value) }),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable div(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic / value;
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(divScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);
			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable pow(double exponent) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.pow(valueIfNonStochastic,exponent);
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuPow, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)exponent }),
					new Pointer()}
					);
			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable squared() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * valueIfNonStochastic;
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		} else
			return this.mult(this);
	}

	@Override
	public RandomVariable sqrt() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.sqrt(valueIfNonStochastic);
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuSqrt, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariableCuda2 exp() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.exp(valueIfNonStochastic);
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuExp, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariableCuda2 log() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.log(valueIfNonStochastic);
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuLog, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	public RandomVariable sin() {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.sin(valueIfNonStochastic);
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			double[] newRealizations = new double[realizations.length];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = Math.sin(realizations[i]);
			return new RandomVariableCuda(time, newRealizations);
		}
		 */
	}

	public RandomVariable cos() {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.cos(valueIfNonStochastic);
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			double[] newRealizations = new double[realizations.length];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = Math.cos(realizations[i]);
			return new RandomVariableCuda(time, newRealizations);
		}
		 */
	}

	public RandomVariable invert() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = 1.0/valueIfNonStochastic;
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(invert, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	public RandomVariable abs() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.abs(valueIfNonStochastic);
			return new RandomVariableCuda2(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuAbs, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	/*
	 * Binary operators: checking for return type priority.
	 */

	@Override
	public RandomVariable add(RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.add(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic + randomVariable.get(0);
			return new RandomVariableCuda2(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			CUdeviceptr result = callCudaFunction(addScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(getRandomVariableCuda(randomVariable).realizations),
					Pointer.to(new float[] { (float)valueIfNonStochastic }),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
		else if(randomVariable.isDeterministic()) {
			return this.add(randomVariable.get(0));
		}
		else {
			CUdeviceptr result = callCudaFunction(add, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(getRandomVariableCuda(randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable sub(RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.bus(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic - randomVariable.get(0);
			return new RandomVariableCuda2(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			CUdeviceptr result = callCudaFunction(busScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(getRandomVariableCuda(randomVariable).realizations),
					Pointer.to(new float[] { (float)valueIfNonStochastic }),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
		else {
			CUdeviceptr result = callCudaFunction(sub, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(getRandomVariableCuda(randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable bus(RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.sub(this);
		}

		// @TODO Add native cuda implementation here
		return this.sub(randomVariable).mult(-1.0);
	}

	@Override
	public RandomVariable mult(RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.mult(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * randomVariable.get(0);
			return new RandomVariableCuda2(newTime, newValueIfNonStochastic);
		}
		else if(!isDeterministic() && randomVariable.isDeterministic()) {
			return this.mult(randomVariable.get(0));			
		}
		else if(isDeterministic() && !randomVariable.isDeterministic()) {
			CUdeviceptr result = callCudaFunction(multScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(getRandomVariableCuda(randomVariable).realizations),
					Pointer.to(new float[] { ((float)valueIfNonStochastic) }),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
		else {
			CUdeviceptr result = callCudaFunction(mult, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(getRandomVariableCuda(randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable div(RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.vid(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic / randomVariable.get(0);
			return new RandomVariableCuda2(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			CUdeviceptr result = callCudaFunction(vidScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(getRandomVariableCuda(randomVariable).realizations),
					Pointer.to(new float[] { (float)valueIfNonStochastic }),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
		else if(randomVariable.isDeterministic())
			return this.div(randomVariable.get(0));
		else {
			CUdeviceptr result = callCudaFunction(cuDiv, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(getRandomVariableCuda(randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(time, result, size());
		}
	}

	@Override
	public RandomVariable vid(RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.div(this);
		}

		// TODO write native cuda kernel
		return this.div(randomVariable).invert();
	}

	@Override
	public RandomVariable cap(RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.cap(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = Math.min(valueIfNonStochastic, randomVariable.get(0));
			return new RandomVariableCuda2(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return getRandomVariableCuda(randomVariable.cap(this));
		}
		else {
			CUdeviceptr result = callCudaFunction(cap, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(getRandomVariableCuda(randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(newTime, result, size());
		}
	}

	@Override
	public RandomVariable floor(RandomVariable randomVariable) {
		if(randomVariable.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return randomVariable.floor(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = Math.max(valueIfNonStochastic, randomVariable.get(0));
			return new RandomVariableCuda2(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return getRandomVariableCuda(randomVariable.floor(this));
		}
		else {
			CUdeviceptr result = callCudaFunction(cuFloor, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(getRandomVariableCuda(randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda2(newTime, result, size());
		}
	}

	@Override
	public RandomVariable accrue(RandomVariable rate, double periodLength) {
		if(rate.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return rate.mult(periodLength).add(1.0).mult(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, rate.getFiltrationTime());

		if(isDeterministic() && rate.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * (1 + rate.get(0) * periodLength);
			return new RandomVariableCuda2(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic() && !rate.isDeterministic()) {
			if(valueIfNonStochastic == 0) return this;
			return this.mult(rate.mult(periodLength).add(1.0));
		}
		else if(!isDeterministic() && rate.isDeterministic()) {
			double rateValue = rate.get(0);
			return this.mult((1 + rateValue * periodLength));
		}
		else {
			CUdeviceptr result = callCudaFunction(accrue, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda2)rate).realizations),
					Pointer.to(new float[] { (float)periodLength }),
					new Pointer()}
					);

			return new RandomVariableCuda2(newTime, result, size());
		}
	}

	@Override
	public RandomVariable discount(RandomVariable rate, double periodLength) {
		if(rate.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return rate.mult(periodLength).add(1.0).invert().mult(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, rate.getFiltrationTime());

		if(isDeterministic() && rate.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic / (1 + rate.get(0) * periodLength);
			return new RandomVariableCuda2(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic() && !rate.isDeterministic()) {
			if(valueIfNonStochastic == 0) return this;
			return this.div(rate.mult(periodLength).add(1.0));
		}
		else if(!isDeterministic() && rate.isDeterministic()) {
			double rateValue = rate.get(0);
			return this.div((1.0 + rateValue * periodLength));
		}
		else {
			CUdeviceptr result = callCudaFunction(discount, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda2)rate).realizations),
					Pointer.to(new float[] { (float)periodLength }),
					new Pointer()}
					);

			return new RandomVariableCuda2(newTime, result, size());
		}
	}

	@Override
	public IntToDoubleFunction getOperator() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DoubleStream getRealizationsStream() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariable average() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariable choose(RandomVariable valueIfTriggerNonNegative, RandomVariable valueIfTriggerNegative) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariable addProduct(RandomVariable factor1, double factor2) {
		if(factor1.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return factor1.mult(factor2).add(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, factor1.getFiltrationTime());

		if(this.isDeterministic() && factor1.isDeterministic())
			return new RandomVariableCuda2(newTime, valueIfNonStochastic + factor1.get(0) * factor2);
		else if(!isDeterministic() && !factor1.isDeterministic() && factor1 instanceof RandomVariableCuda2) {
			CUdeviceptr result = callCudaFunction(addProduct_vs, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda2)factor1).realizations),
					Pointer.to(new float[] { (float)factor2 }),
					new Pointer()}
					);
			return new RandomVariableCuda2(newTime, result, size());
		} else
			return this.add(factor1.mult(factor2));
	}

	@Override
	public RandomVariable addProduct(RandomVariable factor1, RandomVariable factor2) {
		if(factor1.getTypePriority() > this.getTypePriority() || factor2.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return factor1.mult(factor2).add(this);
		}

		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(Math.max(time, factor1.getFiltrationTime()), factor2.getFiltrationTime());

		if(factor1.isDeterministic() && factor2.isDeterministic())
			return add(factor1.get(0) * factor2.get(0));
		else if(factor2.isDeterministic())
			return this.addProduct(factor1, factor2.get(0));
		else if(factor1.isDeterministic())
			return this.addProduct(factor2, factor1.get(0));
		else if(!isDeterministic() && !factor1.isDeterministic() && !factor2.isDeterministic() && factor1 instanceof RandomVariableCuda2 && factor2 instanceof RandomVariableCuda2) {
			CUdeviceptr result = callCudaFunction(addProduct, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda2)factor1).realizations),
					Pointer.to(((RandomVariableCuda2)factor2).realizations),
					new Pointer()}
					);
			return new RandomVariableCuda2(newTime, result, size());
		} else
			return this.add(factor1.mult(factor2));
	}

	@Override
	public RandomVariable addRatio(RandomVariable numerator, RandomVariable denominator) {
		if(numerator.getTypePriority() > this.getTypePriority() || denominator.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return numerator.div(denominator).add(this);
		}

		// TODO Implement a kernel here
		return this.add(numerator.div(denominator));
	}

	@Override
	public RandomVariable subRatio(RandomVariable numerator, RandomVariable denominator) {
		if(numerator.getTypePriority() > this.getTypePriority() || denominator.getTypePriority() > this.getTypePriority()) {
			// Check type priority
			return numerator.div(denominator).mult(-1).add(this);
		}

		// TODO Implement a kernel here
		return this.sub(numerator.div(denominator));
	}

	@Override
	public RandomVariable isNaN() {
		throw new UnsupportedOperationException();
	}

	/*
	 * Cuda specific implementations
	 */

	private double reduce() {
		if(this.isDeterministic()) return valueIfNonStochastic;

		RandomVariableCuda2 reduced = this;
		while(reduced.size() > 1) {
			reduced = reduced.reduceBySize(reduceGridSize);
		}
		return reduced.getRealizations()[0];
	}

	private RandomVariableCuda2 reduceBySize(int bySize) {
		synchronized (vectorsInUseReferenceMap)
		{
			int blockSizeX = bySize;
			int gridSizeX = (int)Math.ceil((double)size()/2 / blockSizeX);
			CUdeviceptr reduceVector = getCUdeviceptr(gridSizeX);

			callCudaFunction(reducePartial, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(reduceVector)},
					gridSizeX, blockSizeX, blockSizeX);

			return new RandomVariableCuda2(0.0, reduceVector, gridSizeX);
		}
	}

	private CUdeviceptr callCudaFunction(CUfunction function, Pointer[] arguments) {
		synchronized (vectorsInUseReferenceMap)
		{
			// Allocate device output memory
			CUdeviceptr result = getCUdeviceptr((long)size());
			arguments[arguments.length-1] = Pointer.to(result);

			int blockSizeX = 256;
			int gridSizeX = (int)Math.ceil((double)size() / blockSizeX);
			callCudaFunction(function, arguments, gridSizeX, blockSizeX, 0);
			return result;
		}
	}

	private void callCudaFunction(final CUfunction function, Pointer[] arguments, final int gridSizeX, final int blockSizeX, final int sharedMemorySize) {
		synchronized (vectorsInUseReferenceMap)
		{
			// Set up the kernel parameters: A pointer to an array
			// of pointers which point to the actual values.
			final Pointer kernelParameters = Pointer.to(arguments);

			// Call the kernel function.
			try {
				deviceExecutor.submit(new Runnable() { public void run() {
					cuCtxSynchronize();
					cuLaunchKernel(function,
							gridSizeX,  1, 1,      // Grid dimension
							blockSizeX, 1, 1,      // Block dimension
							sharedMemorySize * Sizeof.FLOAT, null,               // Shared memory size and stream
							kernelParameters, null // Kernel- and extra parameters
							);
				}}).get();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
}