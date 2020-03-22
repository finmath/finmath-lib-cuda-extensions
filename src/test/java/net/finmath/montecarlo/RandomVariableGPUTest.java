/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 10.02.2004
 */
package net.finmath.montecarlo;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Consumer;

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import net.finmath.montecarlo.cpu.RandomVariableFloatFactory;
import net.finmath.montecarlo.cpu.RandomVariableFromFloatArray;
import net.finmath.montecarlo.cuda.RandomVariableCuda;
import net.finmath.montecarlo.cuda.RandomVariableCudaFactory;
import net.finmath.montecarlo.opencl.RandomVariableOpenCL;
import net.finmath.montecarlo.opencl.RandomVariableOpenCLFactory;
import net.finmath.stochastic.RandomVariable;

/**
 * Test cases for the class net.finmath.montecarlo.opencl.RandomVariableOpenCL and net.finmath.montecarlo.cuda.RandomVariableCuda
 *
 * @author Christian Fries
 * @see net.finmath.montecarlo.opencl.RandomVariableOpenCL
 * @see net.finmath.montecarlo.cuda.RandomVariableCuda
 */
@RunWith(Parameterized.class)
public class RandomVariableGPUTest {

	/*
	 * An error tolerance for the unit tests.
	 *
	 * Note: on many hardwares the test succeed with an errorTolerance of 0.
	 * However, on some systems (maybe depending on the OpenCL version) the floating
	 * point arithmetic differs by 1 ULP.
	 */
	private static final double errorTolerance = 1E-7;

	@Parameters(name="{0}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{ "OpenCL", new RandomVariableOpenCLFactory() },
			{ "Cuda", new RandomVariableCudaFactory() },
		});
	}

	private final RandomVariableFactory randomVariableFactory;

	public RandomVariableGPUTest(final String name, final RandomVariableFactory randomVariableFactory) {
		super();
		this.randomVariableFactory = randomVariableFactory;

		System.out.println("Running with " + randomVariableFactory);
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

	@Test
	public void testRandomVariableDeterministc() {

		// Create a random variable with a constant
		RandomVariable randomVariable = randomVariableFactory.createRandomVariable(2.0);

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
		RandomVariable randomVariable = randomVariableFactory.createRandomVariable(0.0, new double[] {-4.0f, -2.0f, 0.0f, 2.0f, 4.0f} );

		System.out.println(Arrays.toString(randomVariable.getRealizations()));

		// Perform some calculations
		randomVariable = randomVariable.add(4.0);

		System.out.println(Arrays.toString(randomVariable.getRealizations()));
		randomVariable = randomVariable.div(2.0);
		randomVariable = randomVariable.mult(2.0);
		randomVariable = randomVariable.div(2.0);

		System.out.println(Arrays.toString(randomVariable.getRealizations()));

		// The random variable has average value 2.0
		final double average = randomVariable.getAverage();
		Assert.assertEquals("Average", 2.0, average, 1E-7);

		final double[] result = randomVariable.getRealizations();

		// The random variable has variance value 2.0 = (4 + 1 + 0 + 1 + 4) / 5
		Assert.assertTrue(randomVariable.getVariance() == 2.0);

		// Multiply two random variables, this will expand the receiver to a stochastic one
		RandomVariable randomVariable2 = randomVariableFactory.createRandomVariable(3.0);
		randomVariable2 = randomVariable2.mult(randomVariable);

		// The random variable has average value 6.0
		Assert.assertTrue(randomVariable2.getAverage() == 6.0);

		// The random variable has variance value 2 * 9
		Assert.assertTrue(randomVariable2.getVariance() == 2.0 * 9.0);
	}

	@Test
	public void testRandomVariableAverage() throws InterruptedException {
		final int size = 10000;
		final double[] values = new double[size];
		for(int i=0;i<size; i++) {
			values[i] = i;
		}

		final RandomVariable randomVariable = randomVariableFactory.createRandomVariable(0.0, values);

		final double average = randomVariable.getAverage();

		Assert.assertEquals("Average", size*(size-1.0)/2.0/size, average, 1E-7);
	}

	@Test
	public void testRandomVariableArithmeticSqrtPow() {

		// Create a stochastic random variable
		final RandomVariable randomVariable = randomVariableFactory.createRandomVariable(0.0, new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		final RandomVariable check = randomVariable.sqrt().sub(randomVariable.pow(0.5));

		// The random variable is identical 0.0
		Assert.assertEquals(0.0, check.getAverage(), errorTolerance);
		Assert.assertEquals(0.0, check.getVariance(), errorTolerance);
	}

	@Test
	public void testRandomVariableArithmeticSquaredPow() {

		// Create a stochastic random variable
		final RandomVariable randomVariable = randomVariableFactory.createRandomVariable(0.0, new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		final RandomVariable check = randomVariable.squared().sub(randomVariable.pow(2.0));

		// The random variable is identical 0.0
		Assert.assertEquals(0.0, check.getAverage(), errorTolerance);
		Assert.assertEquals(0.0, check.getVariance(), errorTolerance);
	}

	@Test
	public void testRandomVariableStandardDeviation() {

		// Create a stochastic random variable
		final RandomVariable randomVariable = randomVariableFactory.createRandomVariable(0.0, new double[] {3.0, 1.0, 0.0, 2.0, 4.0, 1.0/3.0} );

		Assert.assertEquals(Math.sqrt(randomVariable.getVariance()), randomVariable.getStandardDeviation(), errorTolerance);
	}

	@Test
	public void testRandomVariableOperators() throws InterruptedException {

		final Random random = new Random(31415);

		for(int testRun=0; testRun<1; testRun++) {
			final int numberOfPath = 100000;
			final double[] realizations = new double[numberOfPath];
			for(int i=0; i<numberOfPath; i++) {
				realizations[i]= random.nextDouble();
			}

			final RandomVariableFactory[] rvf = { new RandomVariableFloatFactory(), randomVariableFactory };

			final BiFunction<RandomVariableFactory, BiFunction<RandomVariable,RandomVariable,RandomVariable>, double[]> hash = (rf, f) -> {
				final RandomVariable x = rf.createRandomVariable(0.0, realizations);
				final RandomVariable y = rf.createRandomVariable(0.0, realizations[0]);
				final double[] xr = f.apply(x,y).getRealizations();
				return xr;
			};

			final Consumer<BiFunction<RandomVariable,RandomVariable,RandomVariable>> test = f -> {
				final double[] xr0 = hash.apply(rvf[0],f);
				final double[] xr1 = hash.apply(rvf[1],f);

				for(int i=0; i<xr0.length; i++) {
					Assert.assertEquals(xr0[i], xr1[i], 2E-7*(1+Math.abs(xr0[i])));
				}
				System.out.print(" [ok]");
			};

			System.out.print("\nTesting squared...");
			test.accept((x,y) -> x.squared());
			test.accept((x,y) -> y.squared());

			System.out.print("\nTesting add scalar...");
			test.accept((x,y) -> x.add(1.0/3.0));
			test.accept((x,y) -> y.add(1.0/3.0));

			System.out.print("\nTesting add...");
			test.accept((x,y) -> x.add(x));
			test.accept((x,y) -> x.add(y));
			test.accept((x,y) -> y.add(x));
			test.accept((x,y) -> y.add(y));

			System.out.print("\nTesting sub...");
			test.accept((x,y) -> x.sub(x));
			test.accept((x,y) -> x.sub(y));
			test.accept((x,y) -> y.sub(x));
			test.accept((x,y) -> y.sub(y));

			System.out.print("\nTesting bus...");
			test.accept((x,y) -> x.bus(x));
			test.accept((x,y) -> x.bus(y));
			test.accept((x,y) -> y.bus(x));
			test.accept((x,y) -> y.bus(y));

			System.out.print("\nTesting cap...");
			test.accept((x,y) -> x.cap(1.0/3.0));
			test.accept((x,y) -> y.cap(1.0/3.0));
			test.accept((x,y) -> x.cap(x.sub(1/3)));
			test.accept((x,y) -> y.cap(x.sub(1/3)));
			test.accept((x,y) -> y.cap(y.sub(1/3)));

			System.out.print("\nTesting floor...");
			test.accept((x,y) -> x.floor(1.0/3.0));
			test.accept((x,y) -> y.floor(1.0/3.0));
			test.accept((x,y) -> x.floor(x.add(1/3)));
			test.accept((x,y) -> y.floor(x.add(1/3)));
			test.accept((x,y) -> y.floor(y.add(1/3)));

			System.out.print("\nTesting mult...");
			test.accept((x,y) -> x.mult(x));
			test.accept((x,y) -> x.mult(y));
			test.accept((x,y) -> y.mult(x));
			test.accept((x,y) -> y.mult(y));

			System.out.print("\nTesting mult scalar...");
			test.accept((x,y) -> x.mult(3.1415));
			test.accept((x,y) -> x.mult(1.0/3.0));
			test.accept((x,y) -> y.mult(3.1415));
			test.accept((x,y) -> y.mult(1.0/3.0));

			System.out.print("\nTesting div...");
			test.accept((x,y) -> x.div(x));
			test.accept((x,y) -> x.div(y));
			test.accept((x,y) -> y.div(x));
			test.accept((x,y) -> y.div(y));

			System.out.print("\nTesting div scalar...");
			test.accept((x,y) -> x.div(3.1415));
			test.accept((x,y) -> x.div(1.0/3.0));
			test.accept((x,y) -> y.div(3.1415));
			test.accept((x,y) -> y.div(1.0/3.0));

			System.out.print("\nTesting vid...");
			test.accept((x,y) -> x.vid(x));
			test.accept((x,y) -> x.vid(y));
			test.accept((x,y) -> y.vid(x));
			test.accept((x,y) -> y.vid(y));

			System.out.print("\nTesting exp...");
			test.accept((x,y) -> x.exp());
			test.accept((x,y) -> y.exp());

			System.out.print("\nTesting log...");
			test.accept((x,y) -> x.log());
			test.accept((x,y) -> y.log());

			System.out.print("\nTesting invert...");
			test.accept((x,y) -> x.invert());
			test.accept((x,y) -> y.invert());

			System.out.print("\nTesting abs...");
			test.accept((x,y) -> x.abs());
			test.accept((x,y) -> y.abs());

			System.out.print("\nTesting accrue...");
			test.accept((x,y) -> x.accrue(x, 2.0));
			test.accept((x,y) -> x.accrue(x, 1.0/3.0));
			test.accept((x,y) -> x.accrue(y, 1.0/3.0));
			test.accept((x,y) -> y.accrue(x, 1.0/3.0));
			test.accept((x,y) -> y.accrue(y, 1.0/3.0));

			System.out.print("\nTesting discount...");
			test.accept((x,y) -> x.discount(x, 2.0));
			test.accept((x,y) -> x.discount(x, 1.0/3.0));
			test.accept((x,y) -> x.discount(y, 1.0/3.0));
			test.accept((x,y) -> y.discount(x, 1.0/3.0));
			test.accept((x,y) -> y.discount(y, 1.0/3.0));

			System.out.print("\nTesting add product...");
			test.accept((x,y) -> x.addProduct(x, x));
			test.accept((x,y) -> x.addProduct(x, y));
			test.accept((x,y) -> x.addProduct(y, x));
			test.accept((x,y) -> x.addProduct(y, y));
			test.accept((x,y) -> y.addProduct(x, x));
			test.accept((x,y) -> y.addProduct(x, y));
			test.accept((x,y) -> y.addProduct(y, x));
			test.accept((x,y) -> y.addProduct(y, y));

			System.out.print("\nTesting add product scalar...");
			test.accept((x,y) -> x.addProduct(x, 1.0/3.0));
			test.accept((x,y) -> x.addProduct(y, 1.0/3.0));
			test.accept((x,y) -> y.addProduct(x, 1.0/3.0));
			test.accept((x,y) -> y.addProduct(y, 1.0/3.0));

			System.out.print("\nTesting add sum product...");
			test.accept((x,y) -> x.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , x }));
			test.accept((x,y) -> x.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , y }));
			test.accept((x,y) -> x.addSumProduct(new RandomVariable[] { x , y }, new RandomVariable[] { y , y }));
			test.accept((x,y) -> x.addSumProduct(new RandomVariable[] { y , y }, new RandomVariable[] { y , y }));
			test.accept((x,y) -> y.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , x }));
			test.accept((x,y) -> y.addSumProduct(new RandomVariable[] { x , x }, new RandomVariable[] { x , y }));
			test.accept((x,y) -> y.addSumProduct(new RandomVariable[] { x , y }, new RandomVariable[] { y , y }));
			test.accept((x,y) -> y.addSumProduct(new RandomVariable[] { y , y }, new RandomVariable[] { y , y }));

			System.out.print("\nTesting getAverage...");
			test.accept((x,y) -> new RandomVariableFromFloatArray(x.getAverage()));
			test.accept((x,y) -> new RandomVariableFromFloatArray(y.getAverage()));
			test.accept((x,y) -> new RandomVariableFromFloatArray(x.getAverage(x)));
			test.accept((x,y) -> new RandomVariableFromFloatArray(y.getAverage(y)));

			System.out.print("\n");

		}
	}
}
