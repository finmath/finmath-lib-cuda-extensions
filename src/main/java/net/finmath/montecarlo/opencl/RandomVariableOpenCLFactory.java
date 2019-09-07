/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christianfries.com.
 *
 * Created on 21.06.2017
 */

package net.finmath.montecarlo.opencl;

import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.stochastic.RandomVariable;

/**
 * @author Christian Fries
 *
 */
public class RandomVariableOpenCLFactory extends AbstractRandomVariableFactory {

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
