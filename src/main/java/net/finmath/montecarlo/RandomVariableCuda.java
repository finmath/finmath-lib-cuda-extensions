/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 09.02.2006
 */
package net.finmath.montecarlo;

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
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * The class RandomVariable represents a random variable being the evaluation of a stochastic process
 * at a certain time within a Monte-Carlo simulation.
 * It is thus essentially a vector of floating point numbers - the realizations - together with a double - the time.
 * The index of the vector represents path.
 * The class may also be used for non-stochastic quantities which may potentially be stochastic
 * (e.g. volatility). If only non-stochastic random variables are involved in an operation the class uses
 * optimized code.
 *
 * Accesses performed exclusively through the interface
 * <code>RandomVariableInterface</code> is thread safe (and does not mutate the class).
 *
 * This implementation uses floats for the realizations (consuming less memory compared to using doubles). However,
 * the calculation of the average is performed using double precision.
 * 
 * @author Christian Fries
 * @version 1.8
 */
public class RandomVariableCuda implements RandomVariableInterface {

	private static final long serialVersionUID = 7620120320663270600L;

	private final double      time;	                // Time (filtration)

	// Data model for the stochastic case (otherwise null)
	private final CUdeviceptr	realizations;           // Realizations
	private final long			size;

	// Data model for the non-stochastic case (if realizations==null)
	private final double      valueIfNonStochastic;

	private final static ReferenceQueue<RandomVariableCuda> referenceQueue = new ReferenceQueue<RandomVariableCuda>();
	private final static Map<WeakReference<RandomVariableCuda>, CUdeviceptr> referenceMap = new ConcurrentHashMap<WeakReference<RandomVariableCuda>, CUdeviceptr>();

	private final static Logger logger = Logger.getLogger("net.finmath");

	public final static CUdevice device;
	public final static CUcontext context;

	private final static CUfunction capByScalar;
	private final static CUfunction floorByScalar;
	private final static CUfunction addScalar;
	private final static CUfunction subScalar;
	private final static CUfunction multScalar;
	private final static CUfunction divScalar;
	private final static CUfunction cuPow;
	private final static CUfunction cuSqrt;
	private final static CUfunction cuExp;
	private final static CUfunction cuLog;
	private final static CUfunction invert;
	private final static CUfunction cuAbs;
	private final static CUfunction cap;
	private final static CUfunction cuFloor;
	private final static CUfunction add;
	private final static CUfunction sub;
	private final static CUfunction mult;
	private final static CUfunction cuDiv;
	private final static CUfunction accrue;
	private final static CUfunction discount;

	// Initalize cuda
	static {
		// Enable exceptions and omit all subsequent error checks
		JCudaDriver.setExceptionsEnabled(true);
		JCudaDriver.setLogLevel(LogLevel.LOG_DEBUG);

		// Create the PTX file by calling the NVCC
		String ptxFileName = null;
		try {
			ptxFileName = jcuda.examples.JCudaUtils.preparePtxFile("RandomVariableCudaKernel.cu");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Initialize the driver and create a context for the first device.
		cuInit(0);
		device = new CUdevice();
		cuDeviceGet(device, 0);
		context = new CUcontext();
		cuCtxCreate(context, 0, device);

		// Load the ptx file.
		CUmodule module = new CUmodule();
		cuModuleLoad(module, ptxFileName);

		// Obtain a function pointers
		capByScalar = new CUfunction();
		cuModuleGetFunction(capByScalar, module, "capByScalar");
		floorByScalar = new CUfunction();
		cuModuleGetFunction(floorByScalar, module, "floorByScalar");
		addScalar = new CUfunction();
		cuModuleGetFunction(addScalar, module, "addScalar");
		subScalar = new CUfunction();
		cuModuleGetFunction(subScalar, module, "subScalar");
		multScalar = new CUfunction();
		cuModuleGetFunction(multScalar, module, "multScalar");
		divScalar = new CUfunction();
		cuModuleGetFunction(divScalar, module, "divScalar");
		cuPow = new CUfunction();
		cuModuleGetFunction(cuPow, module, "cuPow");
		cuSqrt = new CUfunction();
		cuModuleGetFunction(cuSqrt, module, "cuSqrt");
		cuExp = new CUfunction();
		cuModuleGetFunction(cuExp, module, "cuExp");
		cuLog = new CUfunction();
		cuModuleGetFunction(cuLog, module, "cuLog");
		invert = new CUfunction();
		cuModuleGetFunction(invert, module, "invert");
		cuAbs = new CUfunction();
		cuModuleGetFunction(cuAbs, module, "cuAbs");
		cap = new CUfunction();
		cuModuleGetFunction(cap, module, "cap");
		cuFloor = new CUfunction();
		cuModuleGetFunction(cuFloor, module, "cuFloor");
		add = new CUfunction();
		cuModuleGetFunction(add, module, "add");
		sub = new CUfunction();
		cuModuleGetFunction(sub, module, "sub");
		mult = new CUfunction();
		cuModuleGetFunction(mult, module, "mult");
		cuDiv = new CUfunction();
		cuModuleGetFunction(cuDiv, module, "cuDiv");
		accrue = new CUfunction();
		cuModuleGetFunction(accrue, module, "accrue");
		discount = new CUfunction();
		cuModuleGetFunction(accrue, module, "discount");
	}

	public RandomVariableCuda(double time, CUdeviceptr realizations, long size) {
		this.time = time;
		this.realizations = realizations;
		this.size = size;
		this.valueIfNonStochastic = Double.NaN;

		// Manage CUdeviceptr
		WeakReference<RandomVariableCuda> reference = new WeakReference<RandomVariableCuda>(this, referenceQueue);
		referenceMap.put(reference, realizations);
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param value the value, a constant.
	 */
	public RandomVariableCuda(double value) {
		this(-Double.MAX_VALUE, value);
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param value the value, a constant.
	 */
	public RandomVariableCuda(double time, double value) {
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
	public RandomVariableCuda(double time, float[] realisations) {
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
	public RandomVariableCuda(double time, double[] realisations) {
		this(time, getFloatArray(realisations));
	}

	public static CUdeviceptr getCUdeviceptr(long size) {
		CUdeviceptr cuDevicePtr = null;
		synchronized (referenceMap) {
			// Check for object to recycle
			Reference<? extends RandomVariableCuda> reference = referenceQueue.poll();
			if(reference != null) {
				cuDevicePtr = referenceMap.remove(reference);
				logger.fine("Recycling device pointer " + cuDevicePtr + " from " + reference);
				return cuDevicePtr;
			}

			// No pointer found, try GC
			if(referenceMap.size() > 50) {
				System.gc();
				try {
					reference = referenceQueue.remove(1);
				} catch (IllegalArgumentException | InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				if(reference != null) {
					logger.fine("Recycling device pointer " + cuDevicePtr + " from " + reference);
					cuDevicePtr = referenceMap.remove(reference);
				}

				// Clean up all remaining pointers
				while(referenceMap.size() > 30 && (reference = referenceQueue.poll()) != null) {
					CUdeviceptr cuPtr = referenceMap.remove(reference);
					logger.fine("Freeing device pointer " + cuDevicePtr + " from " + reference);
					JCudaDriver.cuMemFree(cuPtr);
				}

				if(cuDevicePtr != null) return cuDevicePtr;
			}

			// Still no pointer found, create new one
			cuDevicePtr = new CUdeviceptr();
			int succ = JCudaDriver.cuMemAlloc(cuDevicePtr, size * Sizeof.FLOAT);
			if(succ != 0) {
				cuDevicePtr = null;
				logger.finest("Failed creating device vector "+ cuDevicePtr + " with size=" + size);
			}
			else {
				logger.finest("Creating device vector "+ cuDevicePtr + " with size=" + size);
			}

			return cuDevicePtr;
		}
	}

	private CUdeviceptr createCUdeviceptr(long size) {
		CUdeviceptr cuDevicePtr = getCUdeviceptr(size);
		// Manage CUdeviceptr
		synchronized (referenceMap) {
			WeakReference<RandomVariableCuda> reference = new WeakReference<RandomVariableCuda>(this, referenceQueue);
			referenceMap.put(reference, cuDevicePtr);
		}
		return cuDevicePtr;
	}

	/**
	 * Create a vector on device and copy host vector to it.
	 * 
	 * @param values Host vector.
	 * @return Pointer to device vector.
	 */
	private CUdeviceptr createCUdeviceptr(float[] values) {
		CUdeviceptr cuDevicePtr = createCUdeviceptr((long)values.length);
		JCudaDriver.cuMemcpyHtoD(cuDevicePtr, Pointer.to(values),
				(long)values.length * Sizeof.FLOAT);
		return cuDevicePtr;
	}

	private static float[] getFloatArray(double[] arrayOfDouble) {
		float[] arrayOfFloat = new float[arrayOfDouble.length];
		for(int i=0; i<arrayOfDouble.length; i++) arrayOfFloat[i] = (float)arrayOfDouble[i];
		return arrayOfFloat;
	}

	private static double[] getDoubleArray(float[] arrayOfFloat) {
		double[] arrayOfDouble = new double[arrayOfFloat.length];
		for(int i=0; i<arrayOfFloat.length; i++) arrayOfDouble[i] = arrayOfFloat[i];
		return arrayOfDouble;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getMutableCopy()
	 */
	public RandomVariableCuda getMutableCopy() {
		return this;

		//if(isDeterministic())	return new RandomVariable(time, valueIfNonStochastic);
		//else					return new RandomVariable(time, realizations.clone());
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#equals(net.finmath.montecarlo.RandomVariable)
	 */
	@Override
	public boolean equals(RandomVariableInterface randomVariable) {
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
	public double get(int pathOrState) {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic())   return valueIfNonStochastic;
		else               		return realizations[pathOrState];
		 */
	}

	@Override
	public int size() {
		if(isDeterministic())    return 1;
		else                     return (int)this.size;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getMin()
	 */
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

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getMax()
	 */
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

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getAverage()
	 */
	public double getAverage() {
		if(isDeterministic())	return valueIfNonStochastic;
		if(size() == 0)			return Double.NaN;

		double[] realizations = getRealizations();

		double sum = 0.0;								// Running sum
		double error = 0.0;								// Running error compensation
		for(int i=0; i<realizations.length; i++)  {
			double value = realizations[i] - error;		// Error corrected value
			double newSum = sum + value;				// New sum
			error = (newSum - sum) - value;				// New numerical error
			sum	= newSum;
		}
		return sum/realizations.length;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getAverage(net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public double getAverage(RandomVariableInterface probabilities) {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic())	return valueIfNonStochastic;
		if(size() == 0)			return Double.NaN;

		double sum = 0.0;
		double error = 0.0;														// Running error compensation
		for(int i=0; i<realizations.length; i++)  {
			double value = ((double)realizations[i]) * ((double)probabilities.get(i)) - error;		// Error corrected value
			double newSum = sum + value;				// New sum
			error = (newSum - sum) - value;				// New numerical error
			sum	= newSum;
		}
		return sum / realizations.length;
		 */
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#getVariance()
	 */
	@Override
	public double getVariance() {
		if(isDeterministic())	return 0.0;
		if(size() == 0)			return Double.NaN;

		double average = getAverage();
		return this.squared().getAverage() - average*average;
	}

	@Override
	public double getVariance(RandomVariableInterface probabilities) {
		throw new UnsupportedOperationException();
		/*
		if(isDeterministic())	return 0.0;
		if(size() == 0)			return Double.NaN;

		double sum			= 0.0;
		double sumOfSquared = 0.0;
		double errorOfSum			= 0.0;
		double errorOfSumSquared	= 0.0;
		for(int i=0; i<realizations.length; i++) {
			double value	= realizations[i] * probabilities.get(i) - errorOfSum;
			double newSum	= sum + value;
			errorOfSum		= (newSum - sum) - value;
			sum				= newSum;

			double valueSquared		= realizations[i] * realizations[i] * probabilities.get(i) - errorOfSumSquared;
			double newSumOfSquared	= sumOfSquared + valueSquared;
			errorOfSumSquared		= (newSumOfSquared-sumOfSquared) - valueSquared;
			sumOfSquared			= newSumOfSquared;
		}
		return (sumOfSquared - sum*sum)/realizations.length;
		 */
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
	public double getStandardDeviation(RandomVariableInterface probabilities) {
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
	public double getStandardError(RandomVariableInterface probabilities) {
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
	public double getQuantile(double quantile, RandomVariableInterface probabilities) {
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

	public RandomVariableInterface expand(int numberOfPaths) {
		if(isDeterministic()) {
			// Expand random variable to a vector of path values
			float[] clone = new float[numberOfPaths];
			java.util.Arrays.fill(clone,(float)valueIfNonStochastic);
			return new RandomVariableCuda(time,clone);

		}

		return this;
	}

	@Override
	public RandomVariableInterface cache() {
		return this;
	}

	@Override
	public double[] getRealizations() {
		if(isDeterministic()) {
			double[] result = new double[1];
			result[0] = get(0);
			return result;
		}
		else {
			float[] result = new float[(int)size];
			cuMemcpyDtoH(Pointer.to(result), realizations, size * Sizeof.FLOAT);
			return getDoubleArray(result);
		}
	}

	/**
	 * Returns the realizations as double array. If the random variable is deterministic, then it is expanded
	 * to the given number of paths.
	 *
	 * @param numberOfPaths Number of paths.
	 * @return The realization as double array.
	 */
	@Override
	public double[] getRealizations(int numberOfPaths) {
		throw new UnsupportedOperationException();
		/*

		if(!isDeterministic() && realizations.length != numberOfPaths) throw new RuntimeException("Inconsistent number of paths.");
		return getDoubleArray(((RandomVariableCuda)expand(numberOfPaths)).realizations);
		 */
	}

	@Override
	public RandomVariableInterface apply(org.apache.commons.math3.analysis.UnivariateFunction function) {
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

	public RandomVariableInterface cap(double cap) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.min(valueIfNonStochastic,cap);
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(capByScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)cap }),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#floor(double)
	 */
	@Override
	public RandomVariableInterface floor(double floor) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.max(valueIfNonStochastic,floor);
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(floorByScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)floor }),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface add(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic + value;
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(addScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface sub(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic - value;
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(subScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface mult(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * value;
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(multScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface div(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic / value;
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(divScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);
			return new RandomVariableCuda(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface pow(double exponent) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.pow(valueIfNonStochastic,exponent);
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuPow, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)exponent }),
					new Pointer()}
					);
			return new RandomVariableCuda(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface squared() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * valueIfNonStochastic;
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			return this.mult(this);
		}
	}

	@Override
	public RandomVariableInterface sqrt() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.sqrt(valueIfNonStochastic);
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuSqrt, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#exp()
	 */
	public RandomVariableCuda exp() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.exp(valueIfNonStochastic);
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuExp, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	public RandomVariableCuda log() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.log(valueIfNonStochastic);
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuLog, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	public RandomVariableInterface sin() {
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

	public RandomVariableInterface cos() {
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

	public RandomVariableInterface add(RandomVariableInterface randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic + randomVariable.get(0);
			return new RandomVariableCuda(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) return randomVariable.add(this);
		else {
			CUdeviceptr result = callCudaFunction(add, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#sub(net.finmath.stochastic.RandomVariableInterface)
	 */
	public RandomVariableInterface sub(RandomVariableInterface randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic - randomVariable.get(0);
			return new RandomVariableCuda(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			double[] newRealizations = new double[Math.max(size(), randomVariable.size())];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = valueIfNonStochastic - randomVariable.get(i);
			return new RandomVariableCuda(newTime, newRealizations);
		}
		else {
			CUdeviceptr result = callCudaFunction(sub, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#mult(net.finmath.stochastic.RandomVariableInterface)
	 */
	public RandomVariableInterface mult(RandomVariableInterface randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * randomVariable.get(0);
			return new RandomVariableCuda(newTime, newValueIfNonStochastic);
		}
		else if(!isDeterministic()) {
			return this.mult((float)randomVariable.get(0));
		}
		else if(isDeterministic()) {
			return randomVariable.mult(this.valueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(mult, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#div(net.finmath.stochastic.RandomVariableInterface)
	 */
	public RandomVariableInterface div(RandomVariableInterface randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic / randomVariable.get(0);
			return new RandomVariableCuda(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			double[] newRealizations = new double[Math.max(size(), randomVariable.size())];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = valueIfNonStochastic / randomVariable.get(i);
			return new RandomVariableCuda(newTime, newRealizations);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuDiv, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	public RandomVariableInterface invert() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = 1.0/valueIfNonStochastic;
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(invert, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	public RandomVariableInterface abs() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.abs(valueIfNonStochastic);
			return new RandomVariableCuda(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuAbs, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	public RandomVariableInterface cap(RandomVariableInterface randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = Math.min(valueIfNonStochastic, randomVariable.get(0));
			return new RandomVariableCuda(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) return randomVariable.cap(this);
		else {
			CUdeviceptr result = callCudaFunction(cap, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	public RandomVariableInterface floor(RandomVariableInterface randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = Math.max(valueIfNonStochastic, randomVariable.get(0));
			return new RandomVariableCuda(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) return randomVariable.floor(this);
		else {
			CUdeviceptr result = callCudaFunction(cuFloor, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}


	public RandomVariableInterface accrue(RandomVariableInterface rate, double periodLength) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, rate.getFiltrationTime());

		if(isDeterministic() && rate.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * (1 + rate.get(0) * periodLength);
			return new RandomVariableCuda(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic() && !rate.isDeterministic()) {
			return rate.mult(periodLength*valueIfNonStochastic).add(valueIfNonStochastic);
		}
		else if(!isDeterministic() && rate.isDeterministic()) {
			double rateValue = rate.get(0);
			return this.mult((1 + rateValue * periodLength));
		}
		else {
			CUdeviceptr result = callCudaFunction(accrue, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda)rate).realizations),
					Pointer.to(new float[] { (float)periodLength }),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	public RandomVariableInterface discount(RandomVariableInterface rate, double periodLength) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, rate.getFiltrationTime());

		if(isDeterministic() && rate.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic / (1 + rate.get(0) * periodLength);
			return new RandomVariableCuda(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic() && !rate.isDeterministic()) {
			if(valueIfNonStochastic == 0) return this;
			return rate.mult(periodLength/valueIfNonStochastic).add(1.0/valueIfNonStochastic).invert();
		}
		else if(!isDeterministic() && rate.isDeterministic()) {
			double rateValue = rate.get(0);
			return this.div((1.0 + rateValue * periodLength));
		}
		else {
			CUdeviceptr result = callCudaFunction(discount, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCuda)rate).realizations),
					Pointer.to(new float[] { (float)periodLength }),
					new Pointer()}
					);

			return new RandomVariableCuda(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#barrier(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface barrier(RandomVariableInterface trigger,
			RandomVariableInterface valueIfTriggerNonNegative, RandomVariableInterface valueIfTriggerNegative) {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#barrier(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface barrier(RandomVariableInterface trigger,
			RandomVariableInterface valueIfTriggerNonNegative, double valueIfTriggerNegative) {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addProduct(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface addProduct(RandomVariableInterface factor1, double factor2) {
		// TODO Implement a kernel here
		return this.add(factor1.mult(factor2));
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addProduct(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface addProduct(RandomVariableInterface factor1, RandomVariableInterface factor2) {
		// TODO Implement a kernel here
		return this.add(factor1.mult(factor2));
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addRatio(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface addRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
		// TODO Implement a kernel here
		return this.add(numerator.div(denominator));
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#subRatio(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface subRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
		// TODO Implement a kernel here
		return this.sub(numerator.div(denominator));
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#isNaN()
	 */
	@Override
	public RandomVariableInterface isNaN() {
		// TODO Auto-generated method stub
		return null;
	}

	private CUdeviceptr callCudaFunction(CUfunction function, Pointer[] arguments) {
		// Allocate device output memory
		CUdeviceptr result = getCUdeviceptr((long)size());
		arguments[arguments.length-1] = Pointer.to(result);

		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		Pointer kernelParameters = Pointer.to(arguments);

		// Call the kernel function.
		int blockSizeX = 256;
		int gridSizeX = (int)Math.ceil((double)size() / blockSizeX);
		cuLaunchKernel(function,
				gridSizeX,  1, 1,      // Grid dimension
				blockSizeX, 1, 1,      // Block dimension
				0, null,               // Shared memory size and stream
				kernelParameters, null // Kernel- and extra parameters
				);
		cuCtxSynchronize();
		return result;
	}
}