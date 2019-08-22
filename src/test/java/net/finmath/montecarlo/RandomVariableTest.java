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
import net.finmath.stochastic.Scalar;

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

		Random random = new Random();
		for(int i=0; i<numberOfPath; i++) realizations[i]= random.nextDouble();

		AbstractRandomVariableFactory[] rvf = { new RandomVariableFloatFactory(), new RandomVariableCudaFactory() };

		BiFunction<AbstractRandomVariableFactory, Function<RandomVariable,RandomVariable>, Integer> hash = (rf, f) -> {
			RandomVariable x= rf.createRandomVariable(0.0, realizations);
			double[] xr = f.apply(x).getRealizations();
			return Arrays.hashCode(xr);
		};

		Consumer<Function<RandomVariable,RandomVariable>> test1 = f -> {
			if( hash.apply(rvf[0],f).intValue() != hash.apply(rvf[1],f).intValue() ) {
				System.out.println(" - failed.");
				/*
				RandomVariable x1 = rvf[0].createRandomVariable(0.0, realizations);
				double[] xr1 = f.apply(x1).getRealizations();
				RandomVariable x2 = rvf[1].createRandomVariable(0.0, realizations);
				double[] xr2 = f.apply(x2).getRealizations();
				System.out.print(Arrays.toString(xr1));
				System.out.print(Arrays.toString(xr2));
				 */
			}
			else {
				System.out.println(" - ok.");
			}
			Assert.assertEquals("test", hash.apply(rvf[0],f) , hash.apply(rvf[1],f));			
		};

		BiFunction<AbstractRandomVariableFactory, BiFunction<RandomVariable,RandomVariable,RandomVariable>, Integer> hash2 = (rf, f) -> {
			RandomVariable x = rf.createRandomVariable(0.0, realizations);
			RandomVariable y = rf.createRandomVariable(0.0, realizations[0]);
			double[] xr = f.apply(x,y).getRealizations();
			return Arrays.hashCode(xr);
		};

		Consumer<BiFunction<RandomVariable,RandomVariable,RandomVariable>> test2 = f -> {
			if( hash2.apply(rvf[0],f).intValue() != hash2.apply(rvf[1],f).intValue() ) {
				System.out.println(" - failed.");
			}
			else {
				System.out.println(" - ok.");
			}
			Assert.assertEquals("test", hash2.apply(rvf[0],f) , hash2.apply(rvf[1],f));			
		};

		System.out.print("Testing squared.");
		test2.accept((x,y) -> x.squared());

		System.out.print("Testing squared.");
		test2.accept((x,y) -> y.squared());

		System.out.print("Testing add scalar.");
		test2.accept((x,y) -> x.add(1.0/3.0));
		test2.accept((x,y) -> y.add(1.0/3.0));

		System.out.print("Testing add.");
		test2.accept((x,y) -> x.add(x));
		test2.accept((x,y) -> x.add(y));
		test2.accept((x,y) -> y.add(x));
		test2.accept((x,y) -> y.add(y));

		System.out.print("Testing sub.");
		test2.accept((x,y) -> x.sub(x));
		test2.accept((x,y) -> x.sub(y));
		test2.accept((x,y) -> y.sub(x));
		test2.accept((x,y) -> y.sub(y));

		System.out.print("Testing bus.");
		test2.accept((x,y) -> x.bus(x));
		test2.accept((x,y) -> x.bus(y));
		test2.accept((x,y) -> y.bus(x));
		test2.accept((x,y) -> y.bus(y));

		System.out.print("Testing cap.");
		test2.accept((x,y) -> x.cap(1.0/3.0));
		test2.accept((x,y) -> y.cap(1.0/3.0));
		test2.accept((x,y) -> x.cap(x.sub(1/3)));
		test2.accept((x,y) -> y.cap(x.sub(1/3)));
		test2.accept((x,y) -> y.cap(y.sub(1/3)));

		System.out.print("Testing floor.");
		test2.accept((x,y) -> x.floor(1.0/3.0));
		test2.accept((x,y) -> y.floor(1.0/3.0));
		test2.accept((x,y) -> x.floor(x.add(1/3)));
		test2.accept((x,y) -> y.floor(x.add(1/3)));
		test2.accept((x,y) -> y.floor(y.add(1/3)));

		System.out.print("Testing mult.");
		test2.accept((x,y) -> x.mult(x));
		test2.accept((x,y) -> x.mult(y));
		test2.accept((x,y) -> y.mult(x));
		test2.accept((x,y) -> y.mult(y));

		System.out.print("Testing mult scalar.");
		test2.accept((x,y) -> x.mult(3.1415));
		test2.accept((x,y) -> x.mult(1.0/3.0));
		test2.accept((x,y) -> y.mult(3.1415));
		test2.accept((x,y) -> y.mult(1.0/3.0));

		System.out.print("Testing div.");
		test2.accept((x,y) -> x.div(x));
		test2.accept((x,y) -> x.div(y));
		test2.accept((x,y) -> y.div(x));
		test2.accept((x,y) -> y.div(y));

		System.out.print("Testing div scalar.");
		test2.accept((x,y) -> x.div(3.1415));
		test2.accept((x,y) -> x.div(1.0/3.0));
		test2.accept((x,y) -> y.div(3.1415));
		test2.accept((x,y) -> y.div(1.0/3.0));

		System.out.print("Testing vid.");
		test2.accept((x,y) -> x.vid(x));
		test2.accept((x,y) -> x.vid(y));
		test2.accept((x,y) -> y.vid(x));
		test2.accept((x,y) -> y.vid(y));

		System.out.print("Testing exp.");
		test2.accept((x,y) -> x.exp());
		test2.accept((x,y) -> y.exp());

		System.out.print("Testing log.");
		test2.accept((x,y) -> x.log());
		test2.accept((x,y) -> y.log());

		System.out.print("Testing invert.");
		test2.accept((x,y) -> x.invert());
		test2.accept((x,y) -> y.invert());

		System.out.print("Testing abs.");
		test2.accept((x,y) -> x.abs());
		test2.accept((x,y) -> y.abs());

		System.out.print("Testing accrue.");
		test2.accept((x,y) -> x.accrue(x, 2.0));
		test2.accept((x,y) -> x.accrue(x, 1.0/3.0));
		test2.accept((x,y) -> x.accrue(y, 1.0/3.0));
		test2.accept((x,y) -> y.accrue(x, 1.0/3.0));
		test2.accept((x,y) -> y.accrue(y, 1.0/3.0));

		System.out.print("Testing discount.");
		test2.accept((x,y) -> x.discount(x, 2.0));
		test2.accept((x,y) -> x.discount(x, 1.0/3.0));
		test2.accept((x,y) -> x.discount(y, 1.0/3.0));
		test2.accept((x,y) -> y.discount(x, 1.0/3.0));
		test2.accept((x,y) -> y.discount(y, 1.0/3.0));

		System.out.print("Testing add product");
		test2.accept((x,y) -> x.addProduct(x, x));
		test2.accept((x,y) -> x.addProduct(x, y));
		test2.accept((x,y) -> x.addProduct(y, x));
		test2.accept((x,y) -> x.addProduct(y, y));
		test2.accept((x,y) -> y.addProduct(x, x));
		test2.accept((x,y) -> y.addProduct(x, y));
		test2.accept((x,y) -> y.addProduct(y, x));
		test2.accept((x,y) -> y.addProduct(y, y));

		System.out.print("Testing add product scalar");
		test2.accept((x,y) -> x.addProduct(x, 1.0/3.0));
		test2.accept((x,y) -> x.addProduct(y, 1.0/3.0));
		test2.accept((x,y) -> y.addProduct(x, 1.0/3.0));
		test2.accept((x,y) -> y.addProduct(y, 1.0/3.0));

		System.out.print("Testing add sum product");
		test2.accept((x,y) -> x.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , x }));
		test2.accept((x,y) -> x.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , y }));
		test2.accept((x,y) -> x.addSumProduct(new RandomVariable[] { x , y }, new RandomVariable[] { y , y }));
		test2.accept((x,y) -> x.addSumProduct(new RandomVariable[] { y , y }, new RandomVariable[] { y , y }));
		test2.accept((x,y) -> y.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , x }));
		test2.accept((x,y) -> y.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , y }));
		test2.accept((x,y) -> y.addSumProduct(new RandomVariable[] { x , y }, new RandomVariable[] { y , y }));
		test2.accept((x,y) -> y.addSumProduct(new RandomVariable[] { y , y }, new RandomVariable[] { y , y }));

		System.out.print("Testing getAverage");
		test2.accept((x,y) -> new Scalar(x.getAverage()));
		test2.accept((x,y) -> new Scalar(y.getAverage()));
		test2.accept((x,y) -> new Scalar(x.getAverage(x)));
		test2.accept((x,y) -> new Scalar(y.getAverage(y)));
	}	
}
