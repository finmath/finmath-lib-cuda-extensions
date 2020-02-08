package net.finmath.montecarlo;
/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 10.02.2004
 */


import java.text.DecimalFormat;

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;

import net.finmath.montecarlo.cuda.RandomVariableCuda;
import net.finmath.montecarlo.cuda.alternative.BrownianMotionCudaWithRandomVariableCuda;
import net.finmath.montecarlo.opencl.RandomVariableOpenCL;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * @author Christian Fries
 *
 */
public class BrownianMotionMemoryTest {

	static final DecimalFormat formatterReal2	= new DecimalFormat(" 0.00");
	static final DecimalFormat formatterPercent	= new DecimalFormat(" 00.0%");
	static final DecimalFormat formatterSci4	= new DecimalFormat(" 0.0000E00;-0.0000E00");
	static final DecimalFormat formatterSci1	= new DecimalFormat(" 0E00;-0.E00");

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

	@Test
	public void testBrownianMotionMemory() {
		// The parameters
		final int		seed		= 53252;
		final double	lastTime	= 1;
		final double	dt			= 0.1;

		// Create the time discretization
		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(0.0, (int)(lastTime/dt), dt);

		final long millisStart = System.currentTimeMillis();

		System.out.println("The test generates BrownianMotionCudaWithRandomVariableCuda with different paths.");
		System.out.println("You may observe that the RandomVariableCuda performs clean ups, if memory becomes below a certain level.\n");
		for(int i=0; i<=100; i++) {
			final int numberOfPaths = 100000+10000*i;
			final long[] free = new long[1];
			final long[] total = new long[1];
			jcuda.runtime.JCuda.cudaMemGetInfo(free, total);
			System.out.println("Number of paths = " + numberOfPaths + "\tDevice free memory: " + formatterPercent.format((double)free[0]/(total[0])) + ".");

			// Test the quality of the Brownian motion
			final BrownianMotion brownian = new BrownianMotionCudaWithRandomVariableCuda(
					timeDiscretization,
					1,
					numberOfPaths,
					seed
					);

			final RandomVariable brownianRealization = brownian.getBrownianIncrement(0, 0);
			final double mean		= brownianRealization.getAverage();
			final double variance	= brownianRealization.getVariance();

			Assert.assertTrue(Math.abs(mean         ) < 3.0 * Math.pow(dt,0.5) / Math.pow(numberOfPaths,0.5));
			Assert.assertTrue(Math.abs(variance - dt) < 3.0 * Math.pow(dt,1.0) / Math.pow(numberOfPaths,0.5));
		}

		final long millisEnd = System.currentTimeMillis();

		System.out.println("Test took " + (millisEnd-millisStart)/1000.0 + " sec.");
	}
}
