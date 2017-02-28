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

import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationInterface;

/**
 * @author Christian Fries
 * 
 */
@RunWith(Parameterized.class)
public class BrownianMotionTest {

	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
				{ "BrownianMotion" },							// Text case 1: Java implementation
				{ "BrownianMotionJavaRandom" },					// Text case 2: Java implementation
				{ "BrownianMotionCudaWithHostRandomVariable" },	// Text case 3: Java implementation
				{ "BrownianMotionCudaWithRandomVariableCuda" }	// Text case 4: Java implementation			
		});
	}

	static final DecimalFormat formatterReal2	= new DecimalFormat(" 0.00");
	static final DecimalFormat formatterSci4	= new DecimalFormat(" 0.0000E00;-0.0000E00");
	static final DecimalFormat formatterSci1	= new DecimalFormat(" 0E00;-0.E00");

	private String testCase;

	public BrownianMotionTest(String testCase) {
		this.testCase = testCase;
	}

	@Before
	public void setUp() {
	}

	@Test
	public void testBrownianMotion() {
		// The parameters
		int		seed		= 53252;
		double	lastTime	= 1;
		double	dt			= 0.1;
		int		numberOfPaths = 1000000;

		System.out.print("Test of performance of " + String.format("%-40s", testCase) + "\t");

		// Create the time discretization
		TimeDiscretizationInterface timeDiscretization = new TimeDiscretization(0.0, (int)(lastTime/dt), dt);

		long millisStart = System.currentTimeMillis();

		for(int i=0; i<100; i++) {
			if(i%10 == 0) System.out.print(".");

			// Test the quality of the Brownian motion
			BrownianMotionInterface brownian;

			switch(testCase) {
			case "BrownianMotion":
			default:
				brownian = new BrownianMotion(timeDiscretization, 1, numberOfPaths, seed,
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

			RandomVariableInterface brownianRealization = brownian.getBrownianIncrement(0, 0);
			double mean		= brownianRealization.getAverage();
			double variance	= brownianRealization.getVariance();

			Assert.assertTrue(Math.abs(mean         ) < 3.0 * Math.pow(dt,0.5) / Math.pow(numberOfPaths,0.5));
			Assert.assertTrue(Math.abs(variance - dt) < 3.0 * Math.pow(dt,1.0) / Math.pow(numberOfPaths,0.5));
		}

		long millisEnd = System.currentTimeMillis();

		System.out.println("test took " + (millisEnd-millisStart)/1000.0 + " sec.");
	}
}
