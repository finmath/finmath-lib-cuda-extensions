/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 10.02.2004
 */
package net.finmath.montecarlo;

import org.junit.Assert;
import org.junit.Test;

import net.finmath.montecarlo.cuda.RandomVariableCuda;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * Test cases for the class net.finmath.montecarlo.RandomVariable.
 *
 * @author Christian Fries
 * @see net.finmath.montecarlo.RandomVariable
 */
public class RandomVariableTest {

	@Test
	public void testRandomVariableDeterministc() {

		// Create a random variable with a constant
		RandomVariableInterface randomVariable = new RandomVariableCuda(2.0);

		// Perform some calculations
		randomVariable = randomVariable.mult(2.0);
		randomVariable = randomVariable.add(1.0);
		randomVariable = randomVariable.squared();
		randomVariable = randomVariable.sub(4.0);
		randomVariable = randomVariable.div(7.0);

		// The random variable has average value 3.0 (it is constant 3.0)
		Assert.assertTrue(randomVariable.getAverage() == 3.0);

		// Since the random variable is deterministic, it has zero variance
		Assert.assertTrue(randomVariable.getVariance() == 0.0);
	}

	@Test
	public void testRandomVariableStochastic() throws InterruptedException {
		RandomVariableInterface randomVariable = new RandomVariableCuda(new float[] {-4.0f, -2.0f, 0.0f, 2.0f, 4.0f} );

		// Perform some calculations
		randomVariable = randomVariable.add(4.0);
		randomVariable = randomVariable.div(2.0);
		randomVariable = randomVariable.mult(2.0);
		randomVariable = randomVariable.div(2.0);

		// The random variable has average value 2.0
		double average = randomVariable.getAverage();
		Assert.assertTrue(average == 2.0);

		double[] result = randomVariable.getRealizations();

		// The random variable has variance value 2.0 = (4 + 1 + 0 + 1 + 4) / 5
		Assert.assertTrue(randomVariable.getVariance() == 2.0);

		// Multiply two random variables, this will expand the receiver to a stochastic one
		RandomVariableInterface randomVariable2 = new RandomVariableCuda(3.0);
		randomVariable2 = randomVariable2.mult(randomVariable);

		// The random variable has average value 6.0
		Assert.assertTrue(randomVariable2.getAverage() == 6.0);

		// The random variable has variance value 2 * 9
		Assert.assertTrue(randomVariable2.getVariance() == 2.0 * 9.0);
	}

	@Test
	public void testRandomVariableAverage() throws InterruptedException {
		int size = 100000;
		float[] values = new float[size];
		for(int i=0;i<size; i++) {
			values[i] = (float)i;
		}

		RandomVariableInterface randomVariable = new RandomVariableCuda(0.0,values);

		double average = randomVariable.getAverage();

		Assert.assertEquals((double)size*((double)size-1.0)/2.0/(double)size, average, 1E-2);
	}

	@Test
	public void testRandomVariableArithmeticSqrtPow() {

		// Create a stochastic random variable
		RandomVariableInterface randomVariable = new RandomVariableCuda(0.0, new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		RandomVariableInterface check = randomVariable.sqrt().sub(randomVariable.pow(0.5));

		// The random variable is identical 0.0
		Assert.assertTrue(check.getAverage() == 0.0);
		Assert.assertTrue(check.getVariance() == 0.0);
	}

	@Test
	public void testRandomVariableArithmeticSquaredPow() {

		// Create a stochastic random variable
		RandomVariableInterface randomVariable = new RandomVariableCuda(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		RandomVariableInterface check = randomVariable.squared().sub(randomVariable.pow(2.0));

		// The random variable is identical 0.0
		Assert.assertTrue(check.getAverage() == 0.0);
		Assert.assertTrue(check.getVariance() == 0.0);
	}

	@Test
	public void testRandomVariableStandardDeviation() {

		// Create a stochastic random variable
		RandomVariableInterface randomVariable = new RandomVariableCuda(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		double check = randomVariable.getStandardDeviation() - Math.sqrt(randomVariable.getVariance());
		Assert.assertTrue(check == 0.0);
	}
}
