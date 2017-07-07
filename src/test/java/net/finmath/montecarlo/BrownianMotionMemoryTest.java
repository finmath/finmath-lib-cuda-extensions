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

import net.finmath.montecarlo.cuda.alternative.BrownianMotionCudaWithRandomVariableCuda;
import net.finmath.stochastic.RandomVariableInterface;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationInterface;

/**
 * @author Christian Fries
 * 
 */
public class BrownianMotionMemoryTest {

	static final DecimalFormat formatterReal2	= new DecimalFormat(" 0.00");
	static final DecimalFormat formatterPercent	= new DecimalFormat(" 00.0%");
	static final DecimalFormat formatterSci4	= new DecimalFormat(" 0.0000E00;-0.0000E00");
	static final DecimalFormat formatterSci1	= new DecimalFormat(" 0E00;-0.E00");

	@Test
	public void testBrownianMotionMemory() {
		// The parameters
		int		seed		= 53252;
		double	lastTime	= 1;
		double	dt			= 0.1;

		// Create the time discretization
		TimeDiscretizationInterface timeDiscretization = new TimeDiscretization(0.0, (int)(lastTime/dt), dt);

		long millisStart = System.currentTimeMillis();

		System.out.println("The test generates BrownianMotionCudaWithRandomVariableCuda with different paths.");
		System.out.println("You may observe that the RandomVariableCuda performs clean ups, if memory becomes below a certain level.\n");
		for(int i=0; i<=100; i++) {
			int numberOfPaths = 100000+10000*i;
			long[] free = new long[1];
			long[] total = new long[1];
			jcuda.runtime.JCuda.cudaMemGetInfo(free, total);
			System.out.println("Number of paths = " + numberOfPaths + "\tDevice free memory: " + formatterPercent.format((double)free[0]/(total[0])) + ".");

			// Test the quality of the Brownian motion
			BrownianMotionInterface brownian = new BrownianMotionCudaWithRandomVariableCuda(
						timeDiscretization,
						1,
						numberOfPaths,
						seed
						);

			RandomVariableInterface brownianRealization = brownian.getBrownianIncrement(0, 0);
			double mean		= brownianRealization.getAverage();
			double variance	= brownianRealization.getVariance();

			Assert.assertTrue(Math.abs(mean         ) < 3.0 * Math.pow(dt,0.5) / Math.pow(numberOfPaths,0.5));
			Assert.assertTrue(Math.abs(variance - dt) < 3.0 * Math.pow(dt,1.0) / Math.pow(numberOfPaths,0.5));
		}

		long millisEnd = System.currentTimeMillis();

		System.out.println("Test took " + (millisEnd-millisStart)/1000.0 + " sec.");
	}
}
