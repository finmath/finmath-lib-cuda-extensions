/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 10.02.2004
 */
package net.finmath.montecarlo;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

import org.junit.Assert;
import org.junit.Test;

import net.finmath.montecarlo.cuda.RandomVariableCuda;
import net.finmath.montecarlo.cuda.RandomVariableCudaFactory;
import net.finmath.stochastic.RandomVariable;

/**
 * Test cases for the class net.finmath.montecarlo.RandomVariableFromDoubleArray.
 *
 * @author Christian Fries
 * @see net.finmath.montecarlo.RandomVariableFromDoubleArray
 */
public class RandomVariableTest {

	@Test
	public void testRandomVariableDeterministc() {

		// Create a random variable with a constant
		RandomVariable randomVariable = new RandomVariableCuda(2.0);

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
		RandomVariable randomVariable = new RandomVariableCuda(new float[] {-4.0f, -2.0f, 0.0f, 2.0f, 4.0f} );

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
		RandomVariable randomVariable2 = new RandomVariableCuda(3.0);
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

		RandomVariable randomVariable = new RandomVariableCuda(0.0,values);

		double average = randomVariable.getAverage();

		Assert.assertEquals((double)size*((double)size-1.0)/2.0/(double)size, average, 1E-2);
	}

	@Test
	public void testRandomVariableArithmeticSqrtPow() {

		// Create a stochastic random variable
		RandomVariable randomVariable = new RandomVariableCuda(0.0, new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		RandomVariable check = randomVariable.sqrt().sub(randomVariable.pow(0.5));

		// The random variable is identical 0.0
		Assert.assertTrue(check.getAverage() == 0.0);
		Assert.assertTrue(check.getVariance() == 0.0);
	}

	@Test
	public void testRandomVariableArithmeticSquaredPow() {

		// Create a stochastic random variable
		RandomVariable randomVariable = new RandomVariableCuda(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		RandomVariable check = randomVariable.squared().sub(randomVariable.pow(2.0));

		// The random variable is identical 0.0
		Assert.assertTrue(check.getAverage() == 0.0);
		Assert.assertTrue(check.getVariance() == 0.0);
	}

	@Test
	public void testRandomVariableStandardDeviation() {

		// Create a stochastic random variable
		RandomVariable randomVariable = new RandomVariableCuda(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		double check = randomVariable.getStandardDeviation() - Math.sqrt(randomVariable.getVariance());
		Assert.assertTrue(check == 0.0);
	}

	@Test
	public void testRandomVariableCuda() throws InterruptedException {
		int numberOfPath = 100000;
		final double[] realizations = new double[numberOfPath];
		
		Random random = new Random(314145);
		for(int i=0; i<numberOfPath; i++) realizations[i]= random.nextDouble();

		AbstractRandomVariableFactory[] rvf = { new RandomVariableFactory(false), new RandomVariableCudaFactory() };

		BiFunction<AbstractRandomVariableFactory, Function<RandomVariable,RandomVariable>, Integer> hash = (rf, f) -> {
			RandomVariable x= rf.createRandomVariable(0.0, realizations);
			double[] xr = f.apply(x).getRealizations();
			return Arrays.hashCode(xr);
		};

		Consumer<Function<RandomVariable,RandomVariable>> test = f -> {
			if( hash.apply(rvf[0],f).intValue() != hash.apply(rvf[1],f).intValue() ) {
				RandomVariable x1 = rvf[0].createRandomVariable(0.0, realizations);
				double[] xr1 = f.apply(x1).getRealizations();
				RandomVariable x2 = rvf[1].createRandomVariable(0.0, realizations);
				double[] xr2 = f.apply(x2).getRealizations();
				System.out.println(Arrays.toString(xr1));
				System.out.println(Arrays.toString(xr2));
			}
			Assert.assertEquals("test", hash.apply(rvf[0],f) , hash.apply(rvf[1],f));			
		};
		
		System.out.println("Testing squared.");
		test.accept(x -> x.squared());
				
		System.out.println("Testing add.");
		test.accept(x -> x.add(x));

		System.out.println("Testing add scalar.");
		test.accept(x -> x.add(3.1415f));

		System.out.println("Testing sub.");
		test.accept(x -> x.sub(x));

		System.out.println("Testing sub scalar.");
		test.accept(x -> x.sub(3.1415f));

		System.out.println("Testing mult.");
		test.accept(x -> x.mult(x));

		System.out.println("Testing mult scalar.");
		test.accept(x -> x.mult(3.1415f));

		System.out.println("Testing div.");
		test.accept(x -> x.div(x));

		System.out.println("Testing div scalar.");
		test.accept(x -> x.div(3.1415f));

		System.out.println("Testing exp.");
		test.accept(x -> x.exp());

		System.out.println("Testing cap.");
		test.accept(x -> x.cap(x.sub(1.0f)));

		System.out.println("Testing floor.");
		test.accept(x -> x.floor(x.add(1.0f)));

		System.out.println("Testing accrue.");
		test.accept(x -> x.accrue(x, 2.0f));

		System.out.println("Testing discount.");
		test.accept(x -> x.discount(x, 2.0f));

		System.out.println("Testing add product");
		test.accept(x -> x.addProduct(x,x));
		
		System.out.println("Testing add product scalar");
		test.accept(x -> x.addProduct(x, 17.0f));

		System.out.println("Testing add sum product");
		test.accept(x -> x.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , x }));
		
	}
	
}
