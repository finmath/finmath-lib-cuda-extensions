/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 09.02.2006
 */
package net.finmath.montecarlo.cuda.alternative;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
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
public class RandomVariableCudaWithFinalizer implements RandomVariableInterface {

	private static final long serialVersionUID = 7620120320663270600L;

	private final double      time;	                // Time (filtration)

	// Data model for the stochastic case (otherwise null)
	private final CUdeviceptr	realizations;           // Realizations
	private final long			size;

	// Data model for the non-stochastic case (if realizations==null)
	private final double      valueIfNonStochastic;

	private final static ReferenceQueue<RandomVariableCudaWithFinalizer> referenceQueue = new ReferenceQueue<RandomVariableCudaWithFinalizer>();
	private final static Map<WeakReference<RandomVariableCudaWithFinalizer>, CUdeviceptr> referenceMap = new ConcurrentHashMap<WeakReference<RandomVariableCudaWithFinalizer>, CUdeviceptr>();

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
	private final static CUfunction reducePartial;

	private final static int reduceGridSize = 1024;

	// Initalize cuda
	static {
		// Enable exceptions and omit all subsequent error checks
		JCudaDriver.setExceptionsEnabled(true);
		JCudaDriver.setLogLevel(LogLevel.LOG_DEBUG);

		// Create the PTX file by calling the NVCC
		String ptxFileName = null;
		try {
			ptxFileName = preparePtxFile("RandomVariableCudaKernel.cu");
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
		reducePartial = new CUfunction();
		cuModuleGetFunction(reducePartial, module, "reducePartial");
	}

	public RandomVariableCudaWithFinalizer(double time, CUdeviceptr realizations, long size) {
		this.time = time;
		this.realizations = realizations;
		this.size = size;
		this.valueIfNonStochastic = Double.NaN;

		// Manage CUdeviceptr
		WeakReference<RandomVariableCudaWithFinalizer> reference = new WeakReference<RandomVariableCudaWithFinalizer>(this, referenceQueue);
		referenceMap.put(reference, realizations);
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param value the value, a constant.
	 */
	public RandomVariableCudaWithFinalizer(double value) {
		this(-Double.MAX_VALUE, value);
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param value the value, a constant.
	 */
	public RandomVariableCudaWithFinalizer(double time, double value) {
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
	public RandomVariableCudaWithFinalizer(double time, float[] realisations) {
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
	public RandomVariableCudaWithFinalizer(double time, double[] realisations) {
		this(time, getFloatArray(realisations));
	}

	private CUdeviceptr createCUdeviceptr(long size) {
		CUdeviceptr cuDevicePtr = getCUdeviceptr(size);
		return cuDevicePtr;
	}

	public static CUdeviceptr getCUdeviceptr(long size) {
		CUdeviceptr cuDevicePtr = new CUdeviceptr();
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

	@Override
	protected void finalize() throws Throwable {
		System.out.println("Finalizing " + realizations);
		if(realizations != null) JCudaDriver.cuMemFree(realizations);
		super.finalize();
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
			return new RandomVariableCudaWithFinalizer(time,clone);

		}

		return this;
	}

	@Override
	public RandomVariableInterface cache() {
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
		return new RandomVariableLowMemory(time, values);
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
			throw new UnsupportedOperationException();
		}
	}

	@Override
	public RandomVariableInterface apply(DoubleUnaryOperator function) {
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
	public RandomVariableInterface apply(DoubleBinaryOperator operator, RandomVariableInterface argument) {
		throw new UnsupportedOperationException();
	}

	@Override
	public RandomVariableInterface apply(DoubleTernaryOperator operator, RandomVariableInterface argument1, RandomVariableInterface argument2) {
		throw new UnsupportedOperationException();
	}

	public RandomVariableInterface cap(double cap) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.min(valueIfNonStochastic,cap);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(capByScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)cap }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#floor(double)
	 */
	@Override
	public RandomVariableInterface floor(double floor) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.max(valueIfNonStochastic,floor);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(floorByScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)floor }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface add(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic + value;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(addScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface sub(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic - value;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(subScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface mult(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * value;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(multScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface div(double value) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic / value;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(divScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);
			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface pow(double exponent) {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.pow(valueIfNonStochastic,exponent);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuPow, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)exponent }),
					new Pointer()}
					);
			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariableInterface squared() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * valueIfNonStochastic;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			return this.mult(this);
		}
	}

	@Override
	public RandomVariableInterface sqrt() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.sqrt(valueIfNonStochastic);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuSqrt, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#exp()
	 */
	public RandomVariableCudaWithFinalizer exp() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.exp(valueIfNonStochastic);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuExp, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	public RandomVariableCudaWithFinalizer log() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.log(valueIfNonStochastic);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuLog, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
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
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) return randomVariable.add(this);
		else {
			CUdeviceptr result = callCudaFunction(add, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
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
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			double[] newRealizations = new double[Math.max(size(), randomVariable.size())];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = valueIfNonStochastic - randomVariable.get(i);
			return new RandomVariableCudaWithFinalizer(newTime, newRealizations);
		}
		else {
			CUdeviceptr result = callCudaFunction(sub, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
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
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			float[] newRealizations = new float[Math.max(size(), randomVariable.size())];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = (float) (valueIfNonStochastic * randomVariable.get(i));
			return new RandomVariableCudaWithFinalizer(newTime, newRealizations);
		}
		else {
			CUdeviceptr result = callCudaFunction(mult, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
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
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			double[] newRealizations = new double[Math.max(size(), randomVariable.size())];
			for(int i=0; i<newRealizations.length; i++) newRealizations[i]		 = valueIfNonStochastic / randomVariable.get(i);
			return new RandomVariableCudaWithFinalizer(newTime, newRealizations);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuDiv, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	public RandomVariableInterface invert() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = 1.0/valueIfNonStochastic;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(invert, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	public RandomVariableInterface abs() {
		if(isDeterministic()) {
			double newValueIfNonStochastic = Math.abs(valueIfNonStochastic);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			CUdeviceptr result = callCudaFunction(cuAbs, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	public RandomVariableInterface cap(RandomVariableInterface randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = Math.min(valueIfNonStochastic, randomVariable.get(0));
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) return randomVariable.cap(this);
		else {
			CUdeviceptr result = callCudaFunction(cap, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	public RandomVariableInterface floor(RandomVariableInterface randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			double newValueIfNonStochastic = Math.max(valueIfNonStochastic, randomVariable.get(0));
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) return randomVariable.floor(this);
		else {
			CUdeviceptr result = callCudaFunction(cuFloor, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}


	public RandomVariableInterface accrue(RandomVariableInterface rate, double periodLength) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, rate.getFiltrationTime());

		if(isDeterministic() && rate.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic * (1 + rate.get(0) * periodLength);
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
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
					Pointer.to(((RandomVariableCudaWithFinalizer)rate).realizations),
					Pointer.to(new float[] { (float)periodLength }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	public RandomVariableInterface discount(RandomVariableInterface rate, double periodLength) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		double newTime = Math.max(time, rate.getFiltrationTime());

		if(isDeterministic() && rate.isDeterministic()) {
			double newValueIfNonStochastic = valueIfNonStochastic / (1 + rate.get(0) * periodLength);
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
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
					Pointer.to(((RandomVariableCudaWithFinalizer)rate).realizations),
					Pointer.to(new float[] { (float)periodLength }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public Double doubleValue() {
		// TODO Auto-generated method stub
		return null;
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
	public RandomVariableInterface average() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariableInterface bus(RandomVariableInterface randomVariable) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariableInterface vid(RandomVariableInterface randomVariable) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariableInterface choose(RandomVariableInterface valueIfTriggerNonNegative, RandomVariableInterface valueIfTriggerNegative) {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addProduct(net.finmath.stochastic.RandomVariableInterface, double)
	 */
	@Override
	public RandomVariableInterface addProduct(RandomVariableInterface factor1, double factor2) {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addProduct(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface addProduct(RandomVariableInterface factor1, RandomVariableInterface factor2) {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#addRatio(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface addRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariableInterface#subRatio(net.finmath.stochastic.RandomVariableInterface, net.finmath.stochastic.RandomVariableInterface)
	 */
	@Override
	public RandomVariableInterface subRatio(RandomVariableInterface numerator, RandomVariableInterface denominator) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariableInterface isNaN() {
		// TODO Auto-generated method stub
		return null;
	}

	/*
	 * Cude specific implementations
	 */

	private double reduce() {
		if(this.isDeterministic()) return valueIfNonStochastic;

		RandomVariableCudaWithFinalizer reduced = this;
		while(reduced.size() > 1) reduced = reduced.reduceBySize(reduceGridSize);
		return reduced.getRealizations()[0];
	}

	private RandomVariableCudaWithFinalizer reduceBySize(int bySize) {
		int blockSizeX = bySize;
		int gridSizeX = (int)Math.ceil((double)size()/2 / blockSizeX);
		CUdeviceptr reduceVector = getCUdeviceptr(gridSizeX);

		callCudaFunction(reducePartial, new Pointer[] {
				Pointer.to(new int[] { size() }),
				Pointer.to(realizations),
				Pointer.to(reduceVector)},
				gridSizeX, blockSizeX, blockSizeX);

		return new RandomVariableCudaWithFinalizer(0.0, reduceVector, gridSizeX);
	}

	private CUdeviceptr callCudaFunction(CUfunction function, Pointer[] arguments) {
		// Allocate device output memory
		CUdeviceptr result = getCUdeviceptr((long)size());
		arguments[arguments.length-1] = Pointer.to(result);

		int blockSizeX = 256;
		int gridSizeX = (int)Math.ceil((double)size() / blockSizeX);
		callCudaFunction(function, arguments, gridSizeX, blockSizeX, 0);
		return result;
	}

	private CUdeviceptr callCudaFunction(final CUfunction function, Pointer[] arguments, final int gridSizeX, final int blockSizeX, final int sharedMemorySize) {
		// Allocate device output memory
		CUdeviceptr result = getCUdeviceptr((long)size());
		arguments[arguments.length-1] = Pointer.to(result);

		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		Pointer kernelParameters = Pointer.to(arguments);

		// Call the kernel function.
		cuLaunchKernel(function,
				gridSizeX,  1, 1,      // Grid dimension
				blockSizeX, 1, 1,      // Block dimension
				sharedMemorySize, null,               // Shared memory size and stream
				kernelParameters, null // Kernel- and extra parameters
				);
		cuCtxSynchronize();
		return result;
	}

	/**
	 * The extension of the given file name is replaced with "ptx".
	 * If the file with the resulting name does not exist, it is
	 * compiled from the given file using NVCC. The name of the
	 * PTX file is returned.
	 *
	 * @param cuFileName The name of the .CU file
	 * @return The name of the PTX file
	 * @throws IOException If an I/O error occurs
	 */
	private static String preparePtxFile(String cuFileName) throws IOException
	{
		int endIndex = cuFileName.lastIndexOf('.');
		if (endIndex == -1)
		{
			endIndex = cuFileName.length()-1;
		}
		String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
		File ptxFile = new File(ptxFileName);
		if (ptxFile.exists())
		{
			return ptxFileName;
		}

		File cuFile = new File(cuFileName);
		if (!cuFile.exists())
		{
			throw new IOException("Input file not found: "+cuFileName);
		}
		String modelString = "-m"+System.getProperty("sun.arch.data.model");
		String command =
				"nvcc " + modelString + " -ptx "+
						cuFile.getPath()+" -o "+ptxFileName;

		System.out.println("Executing\n"+command);
		Process process = Runtime.getRuntime().exec(command);

		String errorMessage =
				new String(toByteArray(process.getErrorStream()));
		String outputMessage =
				new String(toByteArray(process.getInputStream()));
		int exitValue = 0;
		try
		{
			exitValue = process.waitFor();
		}
		catch (InterruptedException e)
		{
			Thread.currentThread().interrupt();
			throw new IOException(
					"Interrupted while waiting for nvcc output", e);
		}

		if (exitValue != 0)
		{
			System.out.println("nvcc process exitValue "+exitValue);
			System.out.println("errorMessage:\n"+errorMessage);
			System.out.println("outputMessage:\n"+outputMessage);
			throw new IOException(
					"Could not create .ptx file: "+errorMessage);
		}

		System.out.println("Finished creating PTX file");
		return ptxFileName;
	}

	/**
	 * Fully reads the given InputStream and returns it as a byte array
	 *
	 * @param inputStream The input stream to read
	 * @return The byte array containing the data from the input stream
	 * @throws IOException If an I/O error occurs
	 */
	private static byte[] toByteArray(InputStream inputStream)
			throws IOException
	{
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		byte buffer[] = new byte[8192];
		while (true)
		{
			int read = inputStream.read(buffer);
			if (read == -1)
			{
				break;
			}
			baos.write(buffer, 0, read);
		}
		return baos.toByteArray();
	}
}
