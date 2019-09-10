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

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;

import net.finmath.montecarlo.cuda.RandomVariableCuda;
import net.finmath.montecarlo.cuda.RandomVariableCudaFactory;
import net.finmath.montecarlo.opencl.RandomVariableOpenCL;
import net.finmath.stochastic.RandomVariable;
import net.finmath.stochastic.Scalar;

/**
 * Test cases for the class net.finmath.montecarlo.RandomVariableFromDoubleArray.
 *
 * @author Christian Fries
 * @see net.finmath.montecarlo.RandomVariableFromDoubleArray
 */
public class RandomVariableCudaTest {

	@After
	public void cleanUp() {
		RandomVariableCuda.purge();
		RandomVariableOpenCL.purge();
	}

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
		final double average = randomVariable.getAverage();
		Assert.assertTrue(average == 2.0);

		final double[] result = randomVariable.getRealizations();

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
		final int size = 100000;
		final float[] values = new float[size];
		for(int i=0;i<size; i++) {
			values[i] = (float)i;
		}

		final RandomVariable randomVariable = new RandomVariableCuda(0.0,values);

		final double average = randomVariable.getAverage();

		Assert.assertEquals((double)size*((double)size-1.0)/2.0/(double)size, average, 1E-2);
	}

	@Test
	public void testRandomVariableArithmeticSqrtPow() {

		// Create a stochastic random variable
		final RandomVariable randomVariable = new RandomVariableCuda(0.0, new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		final RandomVariable check = randomVariable.sqrt().sub(randomVariable.pow(0.5));

		// The random variable is identical 0.0
		Assert.assertTrue(check.getAverage() == 0.0);
		Assert.assertTrue(check.getVariance() == 0.0);
	}

	@Test
	public void testRandomVariableArithmeticSquaredPow() {

		// Create a stochastic random variable
		final RandomVariable randomVariable = new RandomVariableCuda(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		final RandomVariable check = randomVariable.squared().sub(randomVariable.pow(2.0));

		// The random variable is identical 0.0
		Assert.assertTrue(check.getAverage() == 0.0);
		Assert.assertTrue(check.getVariance() == 0.0);
	}

	@Test
	public void testRandomVariableStandardDeviation() {

		// Create a stochastic random variable
		final RandomVariable randomVariable = new RandomVariableCuda(0.0,
				new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		final double check = randomVariable.getStandardDeviation() - Math.sqrt(randomVariable.getVariance());
		Assert.assertTrue(check == 0.0);
	}

	@Test
	public void testRandomVariableCuda() throws InterruptedException {

		final Random random = new Random();

		for(int testRun=0; testRun<10; testRun++) {
			final int numberOfPath = 100000;
			final double[] realizations = new double[numberOfPath];
			for(int i=0; i<numberOfPath; i++) {
				realizations[i]= random.nextDouble();
			}

			final AbstractRandomVariableFactory[] rvf = { new RandomVariableFloatFactory(), new RandomVariableCudaFactory() };

			final BiFunction<AbstractRandomVariableFactory, BiFunction<RandomVariable,RandomVariable,RandomVariable>, Integer> hash = (rf, f) -> {
				final RandomVariable x = rf.createRandomVariable(0.0, realizations);
				final RandomVariable y = rf.createRandomVariable(0.0, realizations[0]);
				final double[] xr = f.apply(x,y).getRealizations();
				return Arrays.hashCode(xr);
			};

			final Consumer<BiFunction<RandomVariable,RandomVariable,RandomVariable>> test = f -> {
				if( hash.apply(rvf[0],f).intValue() != hash.apply(rvf[1],f).intValue() ) {
					System.out.println(" - failed.");
				}
				else {
					System.out.println(" - ok.");
				}
				Assert.assertEquals("test", hash.apply(rvf[0],f) , hash.apply(rvf[1],f));
			};

			System.out.print("Testing squared.");
			test.accept((x,y) -> x.squared());

			System.out.print("Testing squared.");
			test.accept((x,y) -> y.squared());

			System.out.print("Testing add scalar.");
			test.accept((x,y) -> x.add(1.0/3.0));
			test.accept((x,y) -> y.add(1.0/3.0));

			System.out.print("Testing add.");
			test.accept((x,y) -> x.add(x));
			test.accept((x,y) -> x.add(y));
			test.accept((x,y) -> y.add(x));
			test.accept((x,y) -> y.add(y));

			System.out.print("Testing sub.");
			test.accept((x,y) -> x.sub(x));
			test.accept((x,y) -> x.sub(y));
			test.accept((x,y) -> y.sub(x));
			test.accept((x,y) -> y.sub(y));

			System.out.print("Testing bus.");
			test.accept((x,y) -> x.bus(x));
			test.accept((x,y) -> x.bus(y));
			test.accept((x,y) -> y.bus(x));
			test.accept((x,y) -> y.bus(y));

			System.out.print("Testing cap.");
			test.accept((x,y) -> x.cap(1.0/3.0));
			test.accept((x,y) -> y.cap(1.0/3.0));
			test.accept((x,y) -> x.cap(x.sub(1/3)));
			test.accept((x,y) -> y.cap(x.sub(1/3)));
			test.accept((x,y) -> y.cap(y.sub(1/3)));

			System.out.print("Testing floor.");
			test.accept((x,y) -> x.floor(1.0/3.0));
			test.accept((x,y) -> y.floor(1.0/3.0));
			test.accept((x,y) -> x.floor(x.add(1/3)));
			test.accept((x,y) -> y.floor(x.add(1/3)));
			test.accept((x,y) -> y.floor(y.add(1/3)));

			System.out.print("Testing mult.");
			test.accept((x,y) -> x.mult(x));
			test.accept((x,y) -> x.mult(y));
			test.accept((x,y) -> y.mult(x));
			test.accept((x,y) -> y.mult(y));

			System.out.print("Testing mult scalar.");
			test.accept((x,y) -> x.mult(3.1415));
			test.accept((x,y) -> x.mult(1.0/3.0));
			test.accept((x,y) -> y.mult(3.1415));
			test.accept((x,y) -> y.mult(1.0/3.0));

			System.out.print("Testing div.");
			test.accept((x,y) -> x.div(x));
			test.accept((x,y) -> x.div(y));
			test.accept((x,y) -> y.div(x));
			test.accept((x,y) -> y.div(y));

			System.out.print("Testing div scalar.");
			test.accept((x,y) -> x.div(3.1415));
			test.accept((x,y) -> x.div(1.0/3.0));
			test.accept((x,y) -> y.div(3.1415));
			test.accept((x,y) -> y.div(1.0/3.0));

			System.out.print("Testing vid.");
			test.accept((x,y) -> x.vid(x));
			test.accept((x,y) -> x.vid(y));
			test.accept((x,y) -> y.vid(x));
			test.accept((x,y) -> y.vid(y));

			System.out.print("Testing exp.");
			test.accept((x,y) -> x.exp());
			test.accept((x,y) -> y.exp());

			System.out.print("Testing log.");
			test.accept((x,y) -> x.log());
			test.accept((x,y) -> y.log());

			System.out.print("Testing invert.");
			test.accept((x,y) -> x.invert());
			test.accept((x,y) -> y.invert());

			System.out.print("Testing abs.");
			test.accept((x,y) -> x.abs());
			test.accept((x,y) -> y.abs());

			System.out.print("Testing accrue.");
			test.accept((x,y) -> x.accrue(x, 2.0));
			test.accept((x,y) -> x.accrue(x, 1.0/3.0));
			test.accept((x,y) -> x.accrue(y, 1.0/3.0));
			test.accept((x,y) -> y.accrue(x, 1.0/3.0));
			test.accept((x,y) -> y.accrue(y, 1.0/3.0));

			System.out.print("Testing discount.");
			test.accept((x,y) -> x.discount(x, 2.0));
			test.accept((x,y) -> x.discount(x, 1.0/3.0));
			test.accept((x,y) -> x.discount(y, 1.0/3.0));
			test.accept((x,y) -> y.discount(x, 1.0/3.0));
			test.accept((x,y) -> y.discount(y, 1.0/3.0));

			System.out.print("Testing add product");
			test.accept((x,y) -> x.addProduct(x, x));
			test.accept((x,y) -> x.addProduct(x, y));
			test.accept((x,y) -> x.addProduct(y, x));
			test.accept((x,y) -> x.addProduct(y, y));
			test.accept((x,y) -> y.addProduct(x, x));
			test.accept((x,y) -> y.addProduct(x, y));
			test.accept((x,y) -> y.addProduct(y, x));
			test.accept((x,y) -> y.addProduct(y, y));

			System.out.print("Testing add product scalar");
			test.accept((x,y) -> x.addProduct(x, 1.0/3.0));
			test.accept((x,y) -> x.addProduct(y, 1.0/3.0));
			test.accept((x,y) -> y.addProduct(x, 1.0/3.0));
			test.accept((x,y) -> y.addProduct(y, 1.0/3.0));

			System.out.print("Testing add sum product");
			test.accept((x,y) -> x.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , x }));
			test.accept((x,y) -> x.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , y }));
			test.accept((x,y) -> x.addSumProduct(new RandomVariable[] { x , y }, new RandomVariable[] { y , y }));
			test.accept((x,y) -> x.addSumProduct(new RandomVariable[] { y , y }, new RandomVariable[] { y , y }));
			test.accept((x,y) -> y.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , x }));
			test.accept((x,y) -> y.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , y }));
			test.accept((x,y) -> y.addSumProduct(new RandomVariable[] { x , y }, new RandomVariable[] { y , y }));
			test.accept((x,y) -> y.addSumProduct(new RandomVariable[] { y , y }, new RandomVariable[] { y , y }));

			System.out.print("Testing getAverage");
			test.accept((x,y) -> new Scalar(x.getAverage()));
			test.accept((x,y) -> new Scalar(y.getAverage()));
			test.accept((x,y) -> new Scalar(x.getAverage(x)));
			test.accept((x,y) -> new Scalar(y.getAverage(y)));

		}
	}
}
