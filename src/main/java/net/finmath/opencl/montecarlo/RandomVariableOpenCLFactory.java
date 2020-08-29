/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christianfries.com.
 *
 * Created on 21.06.2017
 */

package net.finmath.opencl.montecarlo;

import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.RandomVariableFactory;
import net.finmath.stochastic.RandomVariable;

/**
 * RandomVariableFactory creating CUDA random variables (object implementing RandomVariable running on Cuda).
 *
 * @author Christian Fries
 */
public class RandomVariableOpenCLFactory extends AbstractRandomVariableFactory implements RandomVariableFactory {

	private static final long serialVersionUID = 1L;

	public RandomVariableOpenCLFactory() {
		super();
	}

	@Override
	public RandomVariable createRandomVariable(final double time, final double value) {
		return new RandomVariableOpenCL(time, value);
	}

	@Override
	public RandomVariable createRandomVariable(final double time, final double[] values) {
		return new RandomVariableOpenCL(time, values);
	}
}
