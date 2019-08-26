package net.finmath.montecarlo;
/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 10.02.2004
 */


import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import net.finmath.montecarlo.cuda.alternative.BrownianMotionCudaWithHostRandomVariable;
import net.finmath.montecarlo.cuda.alternative.BrownianMotionCudaWithRandomVariableCuda;
import net.finmath.montecarlo.cuda.alternative.BrownianMotionJavaRandom;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * @author Christian Fries
 *
 */
@RunWith(Parameterized.class)
public class BrownianMotionTest {

	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{ "BrownianMotionLazyInit" },							// Text case 1: Java implementation
			{ "BrownianMotionJavaRandom" },					// Text case 2: Java implementation
			{ "BrownianMotionCudaWithHostRandomVariable" },	// Text case 3: Java implementation
			{ "BrownianMotionCudaWithRandomVariableCuda" }	// Text case 4: Java implementation
		});
	}

	static final DecimalFormat formatterReal2	= new DecimalFormat(" 0.00");
	static final DecimalFormat formatterSci4	= new DecimalFormat(" 0.0000E00;-0.0000E00");
	static final DecimalFormat formatterSci1	= new DecimalFormat(" 0E00;-0.E00");

	private final String testCase;

	public BrownianMotionTest(final String testCase) {
		this.testCase = testCase;
	}

	@Before
	public void setUp() {
	}

	@Test
	public void testBrownianMotion() {
		// The parameters
		final int		seed		= 1234;//53252;
		final double	lastTime	= 1;
		final double	dt			= 0.1;
		final int		numberOfPaths = 1000000;

		System.out.print("Test of performance of " + String.format("%-40s", testCase) + "\t");

		// Create the time discretization
		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(0.0, (int)(lastTime/dt), dt);

		final long millisStart = System.currentTimeMillis();

		for(int i=0; i<100; i++) {
			if(i%10 == 0) {
				System.out.print(".");
			}

			// Test the quality of the Brownian motion
			BrownianMotion brownian;

			switch(testCase) {
			case "BrownianMotionLazyInit":
			default:
				brownian = new BrownianMotionLazyInit(timeDiscretization, 1, numberOfPaths, seed,
						new RandomVariableFactory(true));
				break;
			case "BrownianMotionJavaRandom":
				brownian = new BrownianMotionJavaRandom(timeDiscretization, 1, numberOfPaths, seed,
						new RandomVariableFactory(true));
				break;
			case "BrownianMotionCudaWithHostRandomVariable":
				brownian = new BrownianMotionCudaWithHostRandomVariable(
						timeDiscretization,
						1,
						numberOfPaths,
						seed
						);
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

			final RandomVariable brownianRealization = brownian.getBrownianIncrement(0, 0);
			final double mean		= brownianRealization.getAverage();
			final double variance	= brownianRealization.getVariance();

			Assert.assertTrue(Math.abs(mean         ) < 3.0 * Math.pow(dt,0.5) / Math.pow(numberOfPaths,0.5));
			Assert.assertTrue(Math.abs(variance - dt) < 3.0 * Math.pow(dt,1.0) / Math.pow(numberOfPaths,0.5));
		}

		final long millisEnd = System.currentTimeMillis();

		System.out.println("test took " + (millisEnd-millisStart)/1000.0 + " sec.");
	}
}
