/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christianfries.com.
 *
 * Created on 15.06.2016
 */

package net.finmath.montecarlo.assetderivativevaluation;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Collection;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import net.finmath.exception.CalculationException;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionLazyInit;
import net.finmath.montecarlo.RandomVariableFromArrayFactory;
import net.finmath.montecarlo.assetderivativevaluation.models.BlackScholesModel;
import net.finmath.montecarlo.cuda.RandomVariableCuda;
import net.finmath.montecarlo.cuda.RandomVariableCudaFactory;
import net.finmath.montecarlo.cuda.alternative.BrownianMotionCudaWithHostRandomVariable;
import net.finmath.montecarlo.cuda.alternative.BrownianMotionCudaWithRandomVariableCuda;
import net.finmath.montecarlo.cuda.alternative.BrownianMotionJavaRandom;
import net.finmath.montecarlo.model.AbstractProcessModel;
import net.finmath.montecarlo.opencl.RandomVariableOpenCL;
import net.finmath.montecarlo.opencl.RandomVariableOpenCLFactory;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.montecarlo.process.MonteCarloProcessFromProcessModel;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * @author Christian Fries
 *
 */
@RunWith(Parameterized.class)
public class MonteCarloBlackScholesModelTest {

	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			//			{ "BrownianMotionLazyInitCPUFloat" },			// Test case 1: Java implementation using MersenneTwister
			{ "BrownianMotionLazyInitCPUDouble" },			// Test case 2: Java implementation using MersenneTwister
			//			{ "BrownianMotionJavaRandomCPUDouble" },		// Test case 3: Java implementation using Java LCG
			{ "BrownianMotionCudaWithHostRandomVariable" },	// Test case 5: Java implementation using Cuda LCG with Host RandomVariable
			{ "BrownianMotionCudaWithRandomVariableCuda" },	// Test case 6: Java implementation using Cuda LCG with Cuda RandomVariable
			{ "BrownianMotionLazyInitOpenCL" },				// Test case 4: Java implementation using MersenneTwister with OpenCL RandomVariable
		});
	}

	static final DecimalFormat formatterReal2	= new DecimalFormat(" 0.00");
	static final DecimalFormat formatterReal4	= new DecimalFormat(" 0.0000");
	static final DecimalFormat formatterSci4	= new DecimalFormat(" 0.0000E00;-0.0000E00");
	static final DecimalFormat formatterSci1	= new DecimalFormat(" 0E00;-0.E00");


	// Model properties
	private final double	initialValue   = 1.0;
	private final double	riskFreeRate   = 0.05;
	private final double	volatility     = 0.30;

	// Process discretization properties
	private final static int		numberOfPaths		= 1000000;
	private final static int		numberOfTimeSteps	= 100;
	private final static double	deltaT				= 1.0;

	private final static int		seed				= 31415;

	// Product properties
	private final int		assetIndex = 0;
	private final double	optionMaturity = 2.0;
	private final double	optionStrike = 1.05;

	static final BrownianMotion brownianCPU = new BrownianMotionLazyInit(new TimeDiscretizationFromArray(0.0 /* initial */, numberOfTimeSteps, deltaT), 1, numberOfPaths, seed,
			new RandomVariableFromArrayFactory(true));

	static final BrownianMotion brownianCL = new BrownianMotionLazyInit(new TimeDiscretizationFromArray(0.0 /* initial */, numberOfTimeSteps, deltaT), 1, numberOfPaths, seed,
			new RandomVariableOpenCLFactory());

	static final BrownianMotion brownianCuda = new BrownianMotionLazyInit(new TimeDiscretizationFromArray(0.0 /* initial */, numberOfTimeSteps, deltaT), 1, numberOfPaths, seed,
			new RandomVariableCudaFactory());

	static {
		try { brownianCuda.getBrownianIncrement(1, 0); } catch(Error e) {};
		try { brownianCL.getBrownianIncrement(1, 0); } catch(Error e) {};
		try { brownianCPU.getBrownianIncrement(1, 0); } catch(Error e) {};
	}

	private final String testCase;
	private BrownianMotion brownian;

	public MonteCarloBlackScholesModelTest(final String testCase) {
		this.testCase = testCase;
	}

	@After
	public void cleanUp() {
		try {
			RandomVariableCuda.purge();
		}
		catch(Exception | Error e) {}
		try {
			RandomVariableOpenCL.purge();
		}
		catch(Exception | Error e) {}
	}

	@Before
	public void setUp() {
		// Create a time discretizeion
		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(0.0 /* initial */, numberOfTimeSteps, deltaT);

		switch(testCase) {
		case "BrownianMotionLazyInitCPUFloat":
		default:
			brownian = new BrownianMotionLazyInit(timeDiscretization, 1, numberOfPaths, seed,
					new RandomVariableFromArrayFactory(false));
			break;
		case "BrownianMotionLazyInitCPUDouble":
			//			brownian = new BrownianMotionLazyInit(timeDiscretization, 1, numberOfPaths, seed, new RandomVariableFromArrayFactory(true));
			brownian = brownianCPU;
			break;
		case "BrownianMotionJavaRandomCPUDouble":
			brownian = new BrownianMotionJavaRandom(timeDiscretization, 1, numberOfPaths, seed, new RandomVariableFromArrayFactory(true));
			break;
		case "BrownianMotionLazyInitOpenCL":
			//			brownian = new BrownianMotionLazyInit(timeDiscretization, 1, numberOfPaths, seed, new RandomVariableOpenCLFactory());
			brownian = brownianCL;
			break;
		case "BrownianMotionCudaWithHostRandomVariable":
			// brownian = new BrownianMotionCudaWithHostRandomVariable(timeDiscretization, 1, numberOfPaths, seed);
			brownian = brownianCuda;
			break;
		case "BrownianMotionCudaWithRandomVariableCuda":
			brownian = new BrownianMotionCudaWithRandomVariableCuda(
					timeDiscretization,
					1,
					numberOfPaths,
					seed
					);
			break;
		}
	}

	@Test
	public void testProductImplementation() throws CalculationException {
		final long millisStart = System.currentTimeMillis();

		// Create a model
		final AbstractProcessModel model = new BlackScholesModel(initialValue, riskFreeRate, volatility);

		// Create a corresponding MC process
		final MonteCarloProcessFromProcessModel process = new EulerSchemeFromProcessModel(brownian);

		// Link model and process for delegation
		process.setModel(model);
		model.setProcess(process);

		/*
		 * Value a call option - directly
		 */
		final TimeDiscretization timeDiscretization = brownian.getTimeDiscretization();

		final RandomVariable asset = process.getProcessValue(timeDiscretization.getTimeIndex(optionMaturity), assetIndex);
		final RandomVariable numeraireAtPayment = model.getNumeraire(optionMaturity);
		final RandomVariable numeraireAtEval = model.getNumeraire(0.0);

		final RandomVariable payoff = asset.sub(optionStrike).floor(0.0);
		final double value = payoff.div(numeraireAtPayment).mult(numeraireAtEval).getAverage();

		final double valueAnalytic = AnalyticFormulas.blackScholesOptionValue(initialValue, riskFreeRate, volatility, optionMaturity, optionStrike);

		System.out.print(testCase + "\t");
		System.out.print("   value Monte-Carlo = " + formatterReal4.format(value));
		System.out.print("\t value analytic    = " + formatterReal4.format(valueAnalytic));

		final long millisEnd = System.currentTimeMillis();

		System.out.println("\t calculation time = " + formatterReal2.format((millisEnd - millisStart)/1000.0) + " sec.");

		Assert.assertEquals(valueAnalytic, value, 0.005);
	}
}
