/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 09.02.2004
 */
package net.finmath.montecarlo.cuda.alternative;

import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandDestroyGenerator;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_MT19937;

import java.io.Serializable;

import jcuda.LogLevel;
import jcuda.driver.CUdeviceptr;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.runtime.JCuda;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.montecarlo.cuda.RandomVariableCuda;
import net.finmath.montecarlo.cuda.RandomVariableCuda.DevicePointerReference;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretization;

/**
 * Implementation of a time-discrete n-dimensional Brownian motion
 * <i>W = (W<sub>1</sub>,...,W<sub>n</sub>)</i> where <i>W<sub>i</sub></i> is
 * a Brownian motion and <i>W<sub>i</sub></i>, <i>W<sub>j</sub></i> are
 * independent for <i>i</i> not equal <i>j</i>.
 *
 * For a correlated Brownian motion with see
 * {@link net.finmath.montecarlo.CorrelatedBrownianMotion}.
 *
 * Here the dimension <i>n</i> is called factors since this Brownian motion is used to
 * generate multi-dimensional multi-factor Ito processes and there one might
 * use a different number of factors to generate Ito processes of different
 * dimension.
 *
 * The quadruppel (time discretization, number of factors, number of paths, seed)
 * defines the state of an object of this class, i.e., BrownianMotionLazyInit for which
 * there parameters agree, generate the same random numbers.
 *
 * The class is immutable and thread safe. It uses lazy initialization.
 *
 * @author Christian Fries
 * @version 1.6
 */
public class BrownianMotionCudaWithRandomVariableCuda implements BrownianMotion, Serializable {

	private static final long serialVersionUID = -5430067621669213475L;

	private final TimeDiscretization						timeDiscretization;

	private final int			numberOfFactors;
	private final int			numberOfPaths;
	private final int			seed;

	private final AbstractRandomVariableFactory randomVariableFactory;

	private transient	RandomVariable[][]	brownianIncrements;
	private final		Object						brownianIncrementsLazyInitLock = new Object();

	/**
	 * Construct a Brownian motion.
	 *
	 * The constructor allows to set the factory to be used for the construction of
	 * random variables. This allows to generate Brownian increments represented
	 * by different implementations of the RandomVariable (e.g. the RandomVariableFromFloatArray internally
	 * using float representations).
	 *
	 * @param timeDiscretizationFromArray The time discretization used for the Brownian increments.
	 * @param numberOfFactors Number of factors.
	 * @param numberOfPaths Number of paths to simulate.
	 * @param seed The seed of the random number generator.
	 * @param randomVariableFactory Factory to be used to create random variable.
	 */
	public BrownianMotionCudaWithRandomVariableCuda(
			TimeDiscretization timeDiscretization,
			int numberOfFactors,
			int numberOfPaths,
			int seed,
			AbstractRandomVariableFactory randomVariableFactory) {
		super();
		this.timeDiscretization = timeDiscretization;
		this.numberOfFactors	= numberOfFactors;
		this.numberOfPaths		= numberOfPaths;
		this.seed				= seed;

		this.randomVariableFactory = new RandomVariableFactory(false); /* randomVariableFactory */

		this.brownianIncrements	= null; 	// Lazy initialization
	}

	/**
	 * Construct a Brownian motion.
	 *
	 * @param timeDiscretizationFromArray The time discretization used for the Brownian increments.
	 * @param numberOfFactors Number of factors.
	 * @param numberOfPaths Number of paths to simulate.
	 * @param seed The seed of the random number generator.
	 */
	public BrownianMotionCudaWithRandomVariableCuda(
			TimeDiscretization timeDiscretization,
			int numberOfFactors,
			int numberOfPaths,
			int seed) {
		this(timeDiscretization, numberOfFactors, numberOfPaths, seed, new RandomVariableFactory());
	}

	@Override
	public BrownianMotion getCloneWithModifiedSeed(int seed) {
		return new BrownianMotionCudaWithRandomVariableCuda(getTimeDiscretization(), getNumberOfFactors(), getNumberOfPaths(), seed);
	}

	@Override
	public BrownianMotion getCloneWithModifiedTimeDiscretization(TimeDiscretization newTimeDiscretization) {
		/// @TODO This can be improved: a complete recreation of the Brownian motion wouldn't be necessary!
		return new BrownianMotionCudaWithRandomVariableCuda(newTimeDiscretization, getNumberOfFactors(), getNumberOfPaths(), getSeed());
	}

	@Override
	public RandomVariable getBrownianIncrement(int timeIndex, int factor) {

		// Thread safe lazy initialization
		synchronized(brownianIncrementsLazyInitLock) {
			if(brownianIncrements == null) {
				doGenerateBrownianMotion();
			}
		}

		/*
		 *  We return an immutable object which ensures that the receiver does not alter the data.
		 */
		return brownianIncrements[timeIndex][factor];
	}

	/**
	 * Lazy initialization of brownianIncrement. Synchronized to ensure thread safety of lazy init.
	 */
	private void doGenerateBrownianMotion() {
		if(brownianIncrements != null) return;	// Nothing to do

		// Enable exceptions and omit all subsequent error checks
		JCuda.setExceptionsEnabled(true);
		JCurand.setExceptionsEnabled(true);
		JCuda.setLogLevel(LogLevel.LOG_DEBUG);

		// Hack: It is important to init the context first. - Clean up.
		RandomVariable rv = new RandomVariableCuda(0.0);

		curandGenerator generator = new curandGenerator();

		// Create pseudo-random number generator
		curandCreateGenerator(generator, CURAND_RNG_PSEUDO_MT19937);

		// Set seed
		curandSetPseudoRandomGeneratorSeed(generator, seed);

		// Allocate memory for RandomVariableFromDoubleArray wrapper objects.
		brownianIncrements = new RandomVariable[timeDiscretization.getNumberOfTimeSteps()][numberOfFactors];

		// Pre-calculate square roots of deltaT
		for(int timeIndex=0; timeIndex<timeDiscretization.getNumberOfTimeSteps(); timeIndex++) {
			double time = timeDiscretization.getTime(timeIndex+1);
			float sqrtOfTimeStep = (float)Math.sqrt(timeDiscretization.getTimeStep(timeIndex));

			for(int factor=0; factor<numberOfFactors; factor++) {
				// Generate n floats on device
				DevicePointerReference realizations = RandomVariableCuda.getCUdeviceptr((long)numberOfPaths);
				jcuda.jcurand.JCurand.curandGenerateNormal(generator, realizations.get(), numberOfPaths, 0.0f /* mean */, sqrtOfTimeStep /* stddev */);
				brownianIncrements[timeIndex][factor] = RandomVariableCuda.of(time, realizations, numberOfPaths);
			}
		}

		// Cleanup
		curandDestroyGenerator(generator);
	}

	@Override
	public TimeDiscretization getTimeDiscretization() {
		return timeDiscretization;
	}

	@Override
	public int getNumberOfFactors() {
		return numberOfFactors;
	}

	@Override
	public int getNumberOfPaths() {
		return numberOfPaths;
	}

	@Override
	public RandomVariable getRandomVariableForConstant(double value) {
		return new RandomVariableCuda(value);
	}

	/**
	 * @return Returns the seed.
	 */
	public int getSeed() {
		return seed;
	}

	public String toString() {
		return super.toString()
				+ "\n" + "timeDiscretizationFromArray: " + timeDiscretization.toString()
				+ "\n" + "numberOfPaths: " + numberOfPaths
				+ "\n" + "numberOfFactors: " + numberOfFactors
				+ "\n" + "seed: " + seed;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;

		BrownianMotionCudaWithRandomVariableCuda that = (BrownianMotionCudaWithRandomVariableCuda) o;

		if (numberOfFactors != that.numberOfFactors) return false;
		if (numberOfPaths != that.numberOfPaths) return false;
		if (seed != that.seed) return false;
		if (!timeDiscretization.equals(that.timeDiscretization)) return false;

		return true;
	}

	@Override
	public RandomVariable getIncrement(int timeIndex, int factor) {
		return getBrownianIncrement(timeIndex, factor);
	}

	@Override
	public int hashCode() {
		int result = timeDiscretization.hashCode();
		result = 31 * result + numberOfFactors;
		result = 31 * result + numberOfPaths;
		result = 31 * result + seed;
		return result;
	}
}
