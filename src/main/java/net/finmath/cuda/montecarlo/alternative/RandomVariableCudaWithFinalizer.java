/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 09.02.2006
 */
package net.finmath.cuda.montecarlo.alternative;

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
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;

import org.apache.commons.io.IOUtils;

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
public class RandomVariableCudaWithFinalizer implements RandomVariable {

	private static final long serialVersionUID = 7620120320663270600L;

	private final double      time;	                // Time (filtration)

	// Data model for the stochastic case (otherwise null)
	private final CUdeviceptr	realizations;           // Realizations
	private final long			size;

	// Data model for the non-stochastic case (if realizations==null)
	private final double      valueIfNonStochastic;

	private static final ReferenceQueue<RandomVariableCudaWithFinalizer> referenceQueue = new ReferenceQueue<RandomVariableCudaWithFinalizer>();
	private static final Map<WeakReference<RandomVariableCudaWithFinalizer>, CUdeviceptr> referenceMap = new ConcurrentHashMap<WeakReference<RandomVariableCudaWithFinalizer>, CUdeviceptr>();

	private static final Logger logger = Logger.getLogger("net.finmath");

	public static final CUdevice device;
	public static final CUcontext context;

	private static final CUfunction capByScalar;
	private static final CUfunction floorByScalar;
	private static final CUfunction addScalar;
	private static final CUfunction subScalar;
	private static final CUfunction multScalar;
	private static final CUfunction divScalar;
	private static final CUfunction cuPow;
	private static final CUfunction cuSqrt;
	private static final CUfunction cuExp;
	private static final CUfunction cuLog;
	private static final CUfunction invert;
	private static final CUfunction cuAbs;
	private static final CUfunction cap;
	private static final CUfunction cuFloor;
	private static final CUfunction add;
	private static final CUfunction sub;
	private static final CUfunction mult;
	private static final CUfunction cuDiv;
	private static final CUfunction accrue;
	private static final CUfunction discount;
	private static final CUfunction reducePartial;

	private static final int reduceGridSize = 1024;

	// Initalize cuda
	static {
		// Enable exceptions and omit all subsequent error checks
		JCudaDriver.setExceptionsEnabled(true);
		JCudaDriver.setLogLevel(LogLevel.LOG_DEBUG);

		// Create the PTX file by calling the NVCC
		String ptxFileName = null;
		try {
			ptxFileName = preparePtxFile("RandomVariableCudaKernel.cu");
		} catch (final IOException e) {
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
		final CUmodule module = new CUmodule();
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

	public RandomVariableCudaWithFinalizer(final double time, final CUdeviceptr realizations, final long size) {
		this.time = time;
		this.realizations = realizations;
		this.size = size;
		this.valueIfNonStochastic = Double.NaN;

		// Manage CUdeviceptr
		final WeakReference<RandomVariableCudaWithFinalizer> reference = new WeakReference<RandomVariableCudaWithFinalizer>(this, referenceQueue);
		referenceMap.put(reference, realizations);
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param value the value, a constant.
	 */
	public RandomVariableCudaWithFinalizer(final double value) {
		this(-Double.MAX_VALUE, value);
	}

	/**
	 * Create a non stochastic random variable, i.e. a constant.
	 *
	 * @param time the filtration time, set to 0.0 if not used.
	 * @param value the value, a constant.
	 */
	public RandomVariableCudaWithFinalizer(final double time, final double value) {
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
	public RandomVariableCudaWithFinalizer(final double time, final float[] realisations) {
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
	public RandomVariableCudaWithFinalizer(final double time, final double[] realisations) {
		this(time, getFloatArray(realisations));
	}

	private CUdeviceptr createCUdeviceptr(final long size) {
		final CUdeviceptr cuDevicePtr = getCUdeviceptr(size);
		return cuDevicePtr;
	}

	public static CUdeviceptr getCUdeviceptr(final long size) {
		CUdeviceptr cuDevicePtr = new CUdeviceptr();
		final int succ = JCudaDriver.cuMemAlloc(cuDevicePtr, size * Sizeof.FLOAT);
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
	private CUdeviceptr createCUdeviceptr(final float[] values) {
		final CUdeviceptr cuDevicePtr = createCUdeviceptr(values.length);
		JCudaDriver.cuMemcpyHtoD(cuDevicePtr, Pointer.to(values),
				(long)values.length * Sizeof.FLOAT);
		return cuDevicePtr;
	}

	@Override
	protected void finalize() throws Throwable {
		System.out.println("Finalizing " + realizations);
		if(realizations != null) {
			JCudaDriver.cuMemFree(realizations);
		}
		super.finalize();
	}


	private static float[] getFloatArray(final double[] arrayOfDouble) {
		final float[] arrayOfFloat = new float[arrayOfDouble.length];
		for(int i=0; i<arrayOfDouble.length; i++) {
			arrayOfFloat[i] = (float)arrayOfDouble[i];
		}
		return arrayOfFloat;
	}

	private static double[] getDoubleArray(final float[] arrayOfFloat) {
		final double[] arrayOfDouble = new double[arrayOfFloat.length];
		for(int i=0; i<arrayOfFloat.length; i++) {
			arrayOfDouble[i] = arrayOfFloat[i];
		}
		return arrayOfDouble;
	}

	@Override
	public boolean equals(final RandomVariable randomVariable) {
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
	public double get(final int pathOrState) {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		} else {
			throw new UnsupportedOperationException();
		}
	}

	@Override
	public int size() {
		if(isDeterministic()) {
			return 1;
		} else {
			return (int)this.size;
		}
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
		if(isDeterministic()) {
			return valueIfNonStochastic;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return  reduce()/size();
	}

	@Override
	public double getAverage(final RandomVariable probabilities) {
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
	 * @see net.finmath.stochastic.RandomVariable#getVariance()
	 */
	@Override
	public double getVariance() {
		if(isDeterministic()) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		final double average = getAverage();
		return this.squared().getAverage() - average*average;
	}

	@Override
	public double getVariance(final RandomVariable probabilities) {
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
		if(isDeterministic() || size() == 1) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return getVariance() * size()/(size()-1);
	}

	@Override
	public double getStandardDeviation() {
		if(isDeterministic()) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return Math.sqrt(getVariance());
	}

	@Override
	public double getStandardDeviation(final RandomVariable probabilities) {
		if(isDeterministic()) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return Math.sqrt(getVariance(probabilities));
	}

	@Override
	public double getStandardError() {
		if(isDeterministic()) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return getStandardDeviation()/Math.sqrt(size());
	}

	@Override
	public double getStandardError(final RandomVariable probabilities) {
		if(isDeterministic()) {
			return 0.0;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		return getStandardDeviation(probabilities)/Math.sqrt(size());
	}

	@Override
	public double getQuantile(final double quantile) {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		throw new UnsupportedOperationException();
		/*
		float[] realizationsSorted = realizations.clone();
		java.util.Arrays.sort(realizationsSorted);

		int indexOfQuantileValue = Math.min(Math.max((int)Math.round((size()+1) * (1-quantile) - 1), 0), size()-1);

		return realizationsSorted[indexOfQuantileValue];
		 */
	}

	@Override
	public double getQuantile(final double quantile, final RandomVariable probabilities) {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		}
		if(size() == 0) {
			return Double.NaN;
		}

		throw new RuntimeException("Method not implemented.");
	}

	@Override
	public double getQuantileExpectation(final double quantileStart, final double quantileEnd) {
		if(isDeterministic()) {
			return valueIfNonStochastic;
		}
		if(size() == 0) {
			return Double.NaN;
		}
		if(quantileStart > quantileEnd) {
			return getQuantileExpectation(quantileEnd, quantileStart);
		}

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
	public double[] getHistogram(final double[] intervalPoints)
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
	public double[][] getHistogram(final int numberOfPoints, final double standardDeviations) {
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

	public RandomVariable expand(final int numberOfPaths) {
		if(isDeterministic()) {
			// Expand random variable to a vector of path values
			final float[] clone = new float[numberOfPaths];
			java.util.Arrays.fill(clone,(float)valueIfNonStochastic);
			return new RandomVariableCudaWithFinalizer(time,clone);

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
			final double[] result = new double[1];
			result[0] = get(0);
			return result;
		} else {
			throw new UnsupportedOperationException();
		}
	}

	@Override
	public RandomVariable apply(final DoubleUnaryOperator function) {
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
	public RandomVariable apply(final DoubleBinaryOperator operator, final RandomVariable argument) {
		throw new UnsupportedOperationException();
	}

	@Override
	public RandomVariable apply(final DoubleTernaryOperator operator, final RandomVariable argument1, final RandomVariable argument2) {
		throw new UnsupportedOperationException();
	}

	@Override
	public RandomVariable cap(final double cap) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.min(valueIfNonStochastic,cap);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(capByScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)cap }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariable#floor(double)
	 */
	@Override
	public RandomVariable floor(final double floor) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.max(valueIfNonStochastic,floor);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(floorByScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)floor }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariable add(final double value) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic + value;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(addScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariable sub(final double value) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic - value;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(subScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariable mult(final double value) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic * value;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(multScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariable div(final double value) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic / value;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(divScalar, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)value }),
					new Pointer()}
					);
			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariable pow(final double exponent) {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.pow(valueIfNonStochastic,exponent);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(cuPow, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(new float[] { (float)exponent }),
					new Pointer()}
					);
			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariable squared() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic * valueIfNonStochastic;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		} else {
			return this.mult(this);
		}
	}

	@Override
	public RandomVariable sqrt() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.sqrt(valueIfNonStochastic);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(cuSqrt, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariable#exp()
	 */
	@Override
	public RandomVariableCudaWithFinalizer exp() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.exp(valueIfNonStochastic);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(cuExp, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariableCudaWithFinalizer log() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.log(valueIfNonStochastic);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(cuLog, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
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

	@Override
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

	@Override
	public RandomVariable add(final RandomVariable randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic + randomVariable.get(0);
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return randomVariable.add(this);
		} else {
			final CUdeviceptr result = callCudaFunction(add, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariable#sub(net.finmath.stochastic.RandomVariable)
	 */
	@Override
	public RandomVariable sub(final RandomVariable randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic - randomVariable.get(0);
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			final double[] newRealizations = new double[Math.max(size(), randomVariable.size())];
			for(int i=0; i<newRealizations.length; i++) {
				newRealizations[i]		 = valueIfNonStochastic - randomVariable.get(i);
			}
			return new RandomVariableCudaWithFinalizer(newTime, newRealizations);
		}
		else {
			final CUdeviceptr result = callCudaFunction(sub, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariable#mult(net.finmath.stochastic.RandomVariable)
	 */
	@Override
	public RandomVariable mult(final RandomVariable randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic * randomVariable.get(0);
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			final float[] newRealizations = new float[Math.max(size(), randomVariable.size())];
			for(int i=0; i<newRealizations.length; i++) {
				newRealizations[i]		 = (float) (valueIfNonStochastic * randomVariable.get(i));
			}
			return new RandomVariableCudaWithFinalizer(newTime, newRealizations);
		}
		else {
			final CUdeviceptr result = callCudaFunction(mult, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariable#div(net.finmath.stochastic.RandomVariable)
	 */
	@Override
	public RandomVariable div(final RandomVariable randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic / randomVariable.get(0);
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			final double[] newRealizations = new double[Math.max(size(), randomVariable.size())];
			for(int i=0; i<newRealizations.length; i++) {
				newRealizations[i]		 = valueIfNonStochastic / randomVariable.get(i);
			}
			return new RandomVariableCudaWithFinalizer(newTime, newRealizations);
		}
		else {
			final CUdeviceptr result = callCudaFunction(cuDiv, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariable invert() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = 1.0/valueIfNonStochastic;
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(invert, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariable abs() {
		if(isDeterministic()) {
			final double newValueIfNonStochastic = Math.abs(valueIfNonStochastic);
			return new RandomVariableCudaWithFinalizer(time, newValueIfNonStochastic);
		}
		else {
			final CUdeviceptr result = callCudaFunction(cuAbs, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariable cap(final RandomVariable randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = Math.min(valueIfNonStochastic, randomVariable.get(0));
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return randomVariable.cap(this);
		} else {
			final CUdeviceptr result = callCudaFunction(cap, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}

	@Override
	public RandomVariable floor(final RandomVariable randomVariable) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, randomVariable.getFiltrationTime());

		if(isDeterministic() && randomVariable.isDeterministic()) {
			final double newValueIfNonStochastic = Math.max(valueIfNonStochastic, randomVariable.get(0));
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic()) {
			return randomVariable.floor(this);
		} else {
			final CUdeviceptr result = callCudaFunction(cuFloor, new Pointer[] {
					Pointer.to(new int[] { size() }),
					Pointer.to(realizations),
					Pointer.to(((RandomVariableCudaWithFinalizer)randomVariable).realizations),
					new Pointer()}
					);

			return new RandomVariableCudaWithFinalizer(time, result, size());
		}
	}


	@Override
	public RandomVariable accrue(final RandomVariable rate, final double periodLength) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, rate.getFiltrationTime());

		if(isDeterministic() && rate.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic * (1 + rate.get(0) * periodLength);
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic() && !rate.isDeterministic()) {
			return rate.mult(periodLength*valueIfNonStochastic).add(valueIfNonStochastic);
		} else if(!isDeterministic() && rate.isDeterministic()) {
			final double rateValue = rate.get(0);
			return this.mult((1 + rateValue * periodLength));
		}
		else {
			final CUdeviceptr result = callCudaFunction(accrue, new Pointer[] {
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
	public RandomVariable discount(final RandomVariable rate, final double periodLength) {
		// Set time of this random variable to maximum of time with respect to which measurability is known.
		final double newTime = Math.max(time, rate.getFiltrationTime());

		if(isDeterministic() && rate.isDeterministic()) {
			final double newValueIfNonStochastic = valueIfNonStochastic / (1 + rate.get(0) * periodLength);
			return new RandomVariableCudaWithFinalizer(newTime, newValueIfNonStochastic);
		}
		else if(isDeterministic() && !rate.isDeterministic()) {
			if(valueIfNonStochastic == 0) {
				return this;
			}
			return rate.mult(periodLength/valueIfNonStochastic).add(1.0/valueIfNonStochastic).invert();
		}
		else if(!isDeterministic() && rate.isDeterministic()) {
			final double rateValue = rate.get(0);
			return this.div((1.0 + rateValue * periodLength));
		}
		else {
			final CUdeviceptr result = callCudaFunction(discount, new Pointer[] {
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
	public RandomVariable average() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariable bus(final RandomVariable randomVariable) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariable vid(final RandomVariable randomVariable) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariable choose(final RandomVariable valueIfTriggerNonNegative, final RandomVariable valueIfTriggerNegative) {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariable#addProduct(net.finmath.stochastic.RandomVariable, double)
	 */
	@Override
	public RandomVariable addProduct(final RandomVariable factor1, final double factor2) {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariable#addProduct(net.finmath.stochastic.RandomVariable, net.finmath.stochastic.RandomVariable)
	 */
	@Override
	public RandomVariable addProduct(final RandomVariable factor1, final RandomVariable factor2) {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariable#addRatio(net.finmath.stochastic.RandomVariable, net.finmath.stochastic.RandomVariable)
	 */
	@Override
	public RandomVariable addRatio(final RandomVariable numerator, final RandomVariable denominator) {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see net.finmath.stochastic.RandomVariable#subRatio(net.finmath.stochastic.RandomVariable, net.finmath.stochastic.RandomVariable)
	 */
	@Override
	public RandomVariable subRatio(final RandomVariable numerator, final RandomVariable denominator) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public RandomVariable isNaN() {
		// TODO Auto-generated method stub
		return null;
	}

	/*
	 * Cude specific implementations
	 */

	private double reduce() {
		if(this.isDeterministic()) {
			return valueIfNonStochastic;
		}

		RandomVariableCudaWithFinalizer reduced = this;
		while(reduced.size() > 1) {
			reduced = reduced.reduceBySize(reduceGridSize);
		}
		return reduced.getRealizations()[0];
	}

	private RandomVariableCudaWithFinalizer reduceBySize(final int bySize) {
		final int blockSizeX = bySize;
		final int gridSizeX = (int)Math.ceil((double)size()/2 / blockSizeX);
		final CUdeviceptr reduceVector = getCUdeviceptr(gridSizeX);

		callCudaFunction(reducePartial, new Pointer[] {
				Pointer.to(new int[] { size() }),
				Pointer.to(realizations),
				Pointer.to(reduceVector)},
				gridSizeX, blockSizeX, blockSizeX);

		return new RandomVariableCudaWithFinalizer(0.0, reduceVector, gridSizeX);
	}

	private CUdeviceptr callCudaFunction(final CUfunction function, final Pointer[] arguments) {
		// Allocate device output memory
		final CUdeviceptr result = getCUdeviceptr(size());
		arguments[arguments.length-1] = Pointer.to(result);

		final int blockSizeX = 256;
		final int gridSizeX = (int)Math.ceil((double)size() / blockSizeX);
		callCudaFunction(function, arguments, gridSizeX, blockSizeX, 0);
		return result;
	}

	private CUdeviceptr callCudaFunction(final CUfunction function, final Pointer[] arguments, final int gridSizeX, final int blockSizeX, final int sharedMemorySize) {
		// Allocate device output memory
		final CUdeviceptr result = getCUdeviceptr(size());
		arguments[arguments.length-1] = Pointer.to(result);

		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		final Pointer kernelParameters = Pointer.to(arguments);

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
	private static String preparePtxFile(final String cuFileName) throws IOException
	{
		int endIndex = cuFileName.lastIndexOf('.');
		if (endIndex == -1)
		{
			endIndex = cuFileName.length()-1;
		}
		final String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
		final File ptxFile = new File(ptxFileName);
		if (ptxFile.exists()) {
			return ptxFileName;
		}

		final File cuFile = new File(cuFileName);
		if (!cuFile.exists()) {
			throw new IOException("Input file not found: "+cuFileName);
		}
		final String modelString = "-m"+System.getProperty("sun.arch.data.model");
		final String command =
				"nvcc " + modelString + " -ptx "+
						cuFile.getPath()+" -o "+ptxFileName;

		System.out.println("Executing\n"+command);

		final Process process;
		String errorMessage = "N/A";
		String outputMessage = "N/A";
		int exitValue = -1;
		try {
			process = Runtime.getRuntime().exec(command);

			try(InputStream errorStream = process.getErrorStream(); InputStream inputStream = process.getInputStream()) {
				errorMessage = IOUtils.toString(errorStream, Charset.defaultCharset());
				outputMessage = IOUtils.toString(inputStream, Charset.defaultCharset());
				try
				{
					exitValue = process.waitFor();
				}
				catch (final InterruptedException e)
				{
					Thread.currentThread().interrupt();
					throw new IOException(
							"Interrupted while waiting for nvcc output", e);
				}
			}
		}
		catch (final IOException e)
		{
			logger.severe("Unable to run nvcc compiler. Command: "+ command);
			throw new IOException("Unable to run nvcc compiler.", e);
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
	private static byte[] toByteArray(final InputStream inputStream)
			throws IOException
	{
		final ByteArrayOutputStream baos = new ByteArrayOutputStream();
		final byte[] buffer = new byte[8192];
		while (true)
		{
			final int read = inputStream.read(buffer);
			if (read == -1)
			{
				break;
			}
			baos.write(buffer, 0, read);
		}
		return baos.toByteArray();
	}
}
