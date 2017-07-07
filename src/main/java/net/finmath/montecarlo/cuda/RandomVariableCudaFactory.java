/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christianfries.com.
 *
 * Created on 21.06.2017
 */

package net.finmath.montecarlo.cuda;

import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.stochastic.RandomVariableInterface;

/**
 * @author Christian Fries
 *
 */
public class RandomVariableCudaFactory extends AbstractRandomVariableFactory {

	public RandomVariableCudaFactory() {
		super();
	}

	@Override
	public RandomVariableInterface createRandomVariable(double time, double value) {
		return new RandomVariableCuda(time, value);
	}

	@Override
	public RandomVariableInterface createRandomVariable(double time, double[] values) {
		return new RandomVariableCuda(time, values);
	}
}
