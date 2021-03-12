/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christianfries.com.
 *
 * Created on 15.06.2016
 */

package net.finmath.cuda.montecarlo.assetderivativevaluation;

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
import net.finmath.montecarlo.model.ProcessModel;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.montecarlo.process.MonteCarloProcessFromProcessModel;
import net.finmath.cuda.montecarlo.RandomVariableCuda;
import net.finmath.cuda.montecarlo.RandomVariableCudaFactory;
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
			{ "SimulationUsingCPU" },			// Test case 1: Java implementation using MersenneTwister with CPU Double RandomVariable
			{ "SimulationUsingCuda" },				// Test case 2: Java implementation using MersenneTwister with Cuda RandomVariable
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
	private static final int	numberOfPaths		= 1000000;
	private static final int	numberOfTimeSteps	= 100;
	private static final double	deltaT				= 1.0;

	private static final int		seed				= 31415;

	// Product properties
	private final int		assetIndex = 0;
	private final double	optionMaturity = 2.0;
	private final double	optionStrike = 1.05;

	static final BrownianMotion brownianCPU = new BrownianMotionLazyInit(new TimeDiscretizationFromArray(0.0 /* initial */, numberOfTimeSteps, deltaT), 1, numberOfPaths, seed,
			new RandomVariableFromArrayFactory(true));

	static final BrownianMotion brownianCL = new BrownianMotionLazyInit(new TimeDiscretizationFromArray(0.0 /* initial */, numberOfTimeSteps, deltaT), 1, numberOfPaths, seed,
			new RandomVariableCudaFactory());

	static {
		try { brownianCL.getBrownianIncrement(1, 0); } catch(final Error e) {}
		try { brownianCPU.getBrownianIncrement(1, 0); } catch(final Error e) {}
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
	}

	@Before
	public void setUp() {
		// Create a time discretizeion
		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(0.0 /* initial */, numberOfTimeSteps, deltaT);

		switch(testCase) {
		case "BrownianMotionOnCPUSimulationUsingCPUFloat":
		default:
			brownian = new BrownianMotionLazyInit(timeDiscretization, 1, numberOfPaths, seed,
					new RandomVariableFromArrayFactory(false));
			break;
		case "SimulationUsingCPU":
			//			brownian = new BrownianMotionLazyInit(timeDiscretization, 1, numberOfPaths, seed, new RandomVariableFromArrayFactory(true));
			brownian = brownianCPU;
			break;
		case "SimulationUsingCuda":
			//			brownian = new BrownianMotionLazyInit(timeDiscretization, 1, numberOfPaths, seed, new RandomVariableCudaFactory());
			brownian = brownianCL;
			break;
		}
	}

	@Test
	public void testProductImplementation() throws CalculationException {
		final long millisStart = System.currentTimeMillis();

		// Create a model
		final ProcessModel model = new BlackScholesModel(initialValue, riskFreeRate, volatility);

		// Create a corresponding MC process
		final MonteCarloProcessFromProcessModel process = new EulerSchemeFromProcessModel(model, brownian);

		/*
		 * Value a call option - directly
		 */
		final TimeDiscretization timeDiscretization = brownian.getTimeDiscretization();

		final RandomVariable asset = process.getProcessValue(timeDiscretization.getTimeIndex(optionMaturity), assetIndex);
		final RandomVariable numeraireAtPayment = model.getNumeraire(process, optionMaturity);
		final RandomVariable numeraireAtEval = model.getNumeraire(process, 0.0);

		final RandomVariable payoff = asset.sub(optionStrike).floor(0.0);
		final double value = payoff.div(numeraireAtPayment).mult(numeraireAtEval).getAverage();

		final double valueAnalytic = AnalyticFormulas.blackScholesOptionValue(initialValue, riskFreeRate, volatility, optionMaturity, optionStrike);

		final long millisEnd = System.currentTimeMillis();

		System.out.print(String.format("%-25s", testCase));
		System.out.print("\t calculation time = " + formatterReal2.format((millisEnd - millisStart)/1000.0) + " sec.");
		System.out.print("\t value Monte-Carlo = " + formatterReal4.format(value));
		System.out.print("\t value analytic    = " + formatterReal4.format(valueAnalytic));
		System.out.println();

		Assert.assertEquals(valueAnalytic, value, 0.005);
	}
}
