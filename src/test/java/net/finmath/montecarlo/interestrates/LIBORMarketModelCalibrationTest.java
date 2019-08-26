package net.finmath.montecarlo.interestrates;
/*
 * (c) Copyright Christian P. Fries, Germany. All rights reserved. Contact: email@christian-fries.de.
 *
 * Created on 16.01.2015
 */


import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import net.finmath.exception.CalculationException;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveFromForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurveInterpolation;
import net.finmath.montecarlo.AbstractRandomVariableFactory;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionView;
import net.finmath.montecarlo.RandomVariableFloatFactory;
import net.finmath.montecarlo.cuda.RandomVariableCudaFactory;
import net.finmath.montecarlo.interestrate.CalibrationProduct;
import net.finmath.montecarlo.interestrate.models.LIBORMarketModelFromCovarianceModel;
import net.finmath.montecarlo.interestrate.models.covariance.AbstractLIBORCovarianceModelParametric;
import net.finmath.montecarlo.interestrate.models.covariance.BlendedLocalVolatilityModel;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCovarianceModelExponentialForm5Param;
import net.finmath.montecarlo.interestrate.models.covariance.LIBORCovarianceModelStochasticVolatility;
import net.finmath.montecarlo.interestrate.products.AbstractLIBORMonteCarloProduct;
import net.finmath.montecarlo.interestrate.products.SwaptionSimple;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * This class tests the LIBOR market model and products.
 *
 * @author Christian Fries
 */
@RunWith(Parameterized.class)
public class LIBORMarketModelCalibrationTest {

	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{ ProcessingUnit.GPU },
			{ ProcessingUnit.GPU_WITH_CPU_RANDOM },
			{ ProcessingUnit.CPU },
		});
	}

	private final int numberOfPaths		= 81920;
	private final int numberOfFactors	= 5;
	private static final int maxIterations = 25;

	private static final DecimalFormat formatterReal2		= new DecimalFormat(" 0.00");
	private static final DecimalFormat formatterValue		= new DecimalFormat(" ##0.000%;-##0.000%", new DecimalFormatSymbols(Locale.ENGLISH));
	private static final DecimalFormat formatterParam		= new DecimalFormat(" #0.000;-#0.000", new DecimalFormatSymbols(Locale.ENGLISH));
	private static final DecimalFormat formatterDeviation	= new DecimalFormat(" 0.00000E00;-0.00000E00", new DecimalFormatSymbols(Locale.ENGLISH));

	private enum ProcessingUnit {
		CPU,
		GPU,
		GPU_WITH_CPU_RANDOM
	}

	private ProcessingUnit processingUnit;

	public LIBORMarketModelCalibrationTest(ProcessingUnit processingUnit) throws CalculationException {
		this.processingUnit = processingUnit;
	}

	private CalibrationProduct createCalibrationItem(double exerciseDate, double swapPeriodLength, int numberOfPeriods, double moneyness, double targetVolatility, ForwardCurve forwardCurve, DiscountCurve discountCurve) throws CalculationException {

		double[]	fixingDates			= new double[numberOfPeriods];
		double[]	paymentDates		= new double[numberOfPeriods];
		double[]	swapTenor			= new double[numberOfPeriods + 1];

		for (int periodStartIndex = 0; periodStartIndex < numberOfPeriods; periodStartIndex++) {
			fixingDates[periodStartIndex] = exerciseDate + periodStartIndex * swapPeriodLength;
			paymentDates[periodStartIndex] = exerciseDate + (periodStartIndex + 1) * swapPeriodLength;
			swapTenor[periodStartIndex] = exerciseDate + periodStartIndex * swapPeriodLength;
		}
		swapTenor[numberOfPeriods] = exerciseDate + numberOfPeriods * swapPeriodLength;

		// Swaptions swap rate
		double swaprate = moneyness + getParSwaprate(forwardCurve, discountCurve, swapTenor);

		// Set swap rates for each period
		double[] swaprates = new double[numberOfPeriods];
		Arrays.fill(swaprates, swaprate);

		/*
		 * We use Monte-Carlo calibration on implied volatility.
		 * Alternatively you may change here to Monte-Carlo valuation on price or
		 * use an analytic approximation formula, etc.
		 */
		SwaptionSimple swaptionMonteCarlo = new SwaptionSimple(swaprate, swapTenor, SwaptionSimple.ValueUnit.VOLATILITYLOGNORMAL);
		//		double targetValuePrice = AnalyticFormulas.blackModelSwaptionValue(swaprate, targetVolatility, fixingDates[0], swaprate, getSwapAnnuity(discountCurve, swapTenor));
		return new CalibrationProduct(swaptionMonteCarlo, targetVolatility, 1.0);
	}

	@Test
	public void testSwaptionSmileCalibration() throws CalculationException {
		long millisStart = System.currentTimeMillis();

		/*
		 * Calibration test
		 */
		System.out.println("Calibration to Swaptions using " + processingUnit.name());

		double[] fixingTimes = new double[] {
				0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5, 50.0
		};

		double[] forwardRates = new double[] {
				0.61/100.0,	0.61/100.0,	0.67/100.0,	0.73/100.0,	0.80/100.0,	0.92/100.0,	1.11/100.0,	1.36/100.0,	1.60/100.0,	1.82/100.0,	2.02/100.0,	2.17/100.0,	2.27/100.0,	2.36/100.0,	2.46/100.0,	2.52/100.0,	2.54/100.0,	2.57/100.0,	2.68/100.0,	2.82/100.0,	2.92/100.0,	2.98/100.0,	3.00/100.0,	2.99/100.0,	2.95/100.0,	2.89/100.0,	2.82/100.0,	2.74/100.0,	2.66/100.0,	2.59/100.0,	2.52/100.0,	2.47/100.0,	2.42/100.0,	2.38/100.0,	2.35/100.0,	2.33/100.0,	2.31/100.0,	2.30/100.0,	2.29/100.0,	2.28/100.0,	2.27/100.0,	2.27/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.26/100.0,	2.27/100.0,	2.28/100.0,	2.28/100.0,	2.30/100.0,	2.31/100.0,	2.32/100.0,	2.34/100.0,	2.35/100.0,	2.37/100.0,	2.39/100.0,	2.42/100.0,	2.44/100.0,	2.47/100.0,	2.50/100.0,	2.52/100.0,	2.56/100.0,	2.59/100.0,	2.62/100.0,	2.65/100.0,	2.68/100.0,	2.72/100.0,	2.75/100.0,	2.78/100.0,	2.81/100.0,	2.83/100.0,	2.86/100.0,	2.88/100.0,	2.91/100.0,	2.93/100.0,	2.94/100.0,	2.96/100.0,	2.97/100.0,	2.97/100.0,	2.97/100.0,	2.97/100.0,	2.97/100.0,	2.96/100.0,	2.95/100.0,	2.94/100.0,	2.93/100.0,	2.91/100.0,	2.89/100.0,	2.87/100.0,	2.85/100.0,	2.83/100.0,	2.80/100.0,	2.78/100.0,	2.75/100.0,	2.72/100.0,	2.69/100.0,	2.67/100.0,	2.64/100.0,	2.64/100.0
		};

		double liborPeriodLength = 0.5;

		// Create the forward curve (initial value of the LIBOR market model)
		ForwardCurve forwardCurve = ForwardCurveInterpolation.createForwardCurveFromForwards(
				"forwardCurve"		/* name of the curve */,
				fixingTimes			/* fixings of the forward */,
				forwardRates		/* forwards */,
				liborPeriodLength	/* tenor / period length */
				);

		DiscountCurve discountCurve = new DiscountCurveFromForwardCurve(forwardCurve, liborPeriodLength);

		/*
		 * Create a set of calibration products.
		 */
		ArrayList<CalibrationProduct> calibrationProducts = new ArrayList<CalibrationProduct>();

		double	swapPeriodLength	= 0.5;
		int		numberOfPeriods		= 20;

		double[] smileMoneynesses	= { -0.02,	-0.01, -0.005, -0.0025,	0.0,	0.0025,	0.0050,	0.01,	0.02 };
		double[] smileVolatilities	= { 0.559,	0.377,	0.335,	 0.320,	0.308, 0.298, 0.290, 0.280, 0.270 };

		for(int i=0; i<smileMoneynesses.length; i++ ) {
			double	exerciseDate		= 5.0;
			double	moneyness			= smileMoneynesses[i];
			double	targetVolatility	= smileVolatilities[i];

			calibrationProducts.add(createCalibrationItem(exerciseDate, swapPeriodLength, numberOfPeriods, moneyness, targetVolatility, forwardCurve, discountCurve));
		}

		double[] atmOptionMaturities	= { 2.00, 3.00, 4.00, 5.00, 7.00, 10.00, 15.00, 20.00, 25.00, 30.00 };
		double[] atmOptionVolatilities	= { 0.385, 0.351, 0.325, 0.308, 0.288, 0.279, 0.290, 0.272, 0.235, 0.192 };

		for(int i=0; i<atmOptionMaturities.length; i++ ) {

			double	exerciseDate		= atmOptionMaturities[i];
			double	moneyness			= 0.0;
			double	targetVolatility	= atmOptionVolatilities[i];

			calibrationProducts.add(createCalibrationItem(exerciseDate, swapPeriodLength, numberOfPeriods, moneyness, targetVolatility, forwardCurve, discountCurve));
		}

		/*
		 * Create a LIBOR Market Model
		 */

		/*
		 * Create the libor tenor structure and the initial values
		 */
		double liborRateTimeHorzion	= 20.0;
		TimeDiscretizationFromArray liborPeriodDiscretization = new TimeDiscretizationFromArray(0.0, (int) (liborRateTimeHorzion / liborPeriodLength), liborPeriodLength);

		/*
		 * Create a simulation time discretization
		 */
		double lastTime	= 20.0;
		double dt		= 0.5;
		TimeDiscretizationFromArray timeDiscretizationFromArray = new TimeDiscretizationFromArray(0.0, (int) (lastTime / dt), dt);

		/*
		 * Create Brownian motions
		 */
		AbstractRandomVariableFactory randomVariableFactory;
		BrownianMotion brownianMotion;
		switch(processingUnit) {
		case CPU:
			randomVariableFactory = new RandomVariableFloatFactory();
			brownianMotion = new net.finmath.montecarlo.BrownianMotionLazyInit(timeDiscretizationFromArray, numberOfFactors + 1, numberOfPaths, 31415 /* seed */, randomVariableFactory);
			break;
		case GPU:
			randomVariableFactory = new RandomVariableCudaFactory();
			brownianMotion = new net.finmath.montecarlo.cuda.alternative.BrownianMotionCudaWithRandomVariableCuda(timeDiscretizationFromArray, numberOfFactors + 1, numberOfPaths, 31415 /* seed */);
			break;
		case GPU_WITH_CPU_RANDOM:
		default:
			randomVariableFactory = new RandomVariableCudaFactory();
			brownianMotion = new net.finmath.montecarlo.BrownianMotionLazyInit(timeDiscretizationFromArray, numberOfFactors + 1, numberOfPaths, 31415 /* seed */, randomVariableFactory);
			break;
		}

		BrownianMotion brownianMotionView1 = new BrownianMotionView(brownianMotion, new Integer[] { 0, 1, 2, 3, 4 });
		BrownianMotion brownianMotionView2 = new BrownianMotionView(brownianMotion, new Integer[] { 0, 5 });

		// Create a covariance model
		AbstractLIBORCovarianceModelParametric covarianceModelParametric = new LIBORCovarianceModelExponentialForm5Param(timeDiscretizationFromArray, liborPeriodDiscretization, numberOfFactors, new double[] { 0.20, 0.05, 0.10, 0.05, 0.10} );
		// Create blended local volatility model with fixed parameter 0.0 (that is "lognormal").
		AbstractLIBORCovarianceModelParametric covarianceModelBlended = new BlendedLocalVolatilityModel(covarianceModelParametric, forwardCurve, 0.2, true);
		// Create stochastic scaling (pass brownianMotionView2 to it)
		AbstractLIBORCovarianceModelParametric covarianceModelStochasticParametric = new LIBORCovarianceModelStochasticVolatility(covarianceModelBlended, brownianMotionView2, 0.15, -0.70, true);

		// Set model properties
		Map<String, Object> properties = new HashMap<String, Object>();

		// Choose the simulation measure
		properties.put("measure", LIBORMarketModelFromCovarianceModel.Measure.SPOT.name());

		// Choose normal state space for the Euler scheme (the covariance model above carries a linear local volatility model, such that the resulting model is log-normal).
		properties.put("stateSpace", LIBORMarketModelFromCovarianceModel.StateSpace.NORMAL.name());

		// Set calibration properties (should use our brownianMotion for calibration - needed to have to right correlation).
		Map<String, Object> calibrationParameters = new HashMap<String, Object>();
		calibrationParameters.put("accuracyParameter", new Double(1E-12));
		calibrationParameters.put("brownianMotion", brownianMotionView1);
		calibrationParameters.put("maxIterations", maxIterations);
		properties.put("calibrationParameters", calibrationParameters);

		LIBORMarketModelFromCovarianceModel liborMarketModelCalibrated = LIBORMarketModelFromCovarianceModel.of(
				liborPeriodDiscretization,
				null,
				forwardCurve,
				null,
				randomVariableFactory,
				covarianceModelStochasticParametric, calibrationProducts.toArray(new CalibrationProduct[0]), properties);

		/*
		 * Test our calibration
		 */
		System.out.println("\nCalibrated parameters are:");
		double[] param = ((AbstractLIBORCovarianceModelParametric) liborMarketModelCalibrated.getCovarianceModel()).getParameterAsDouble();
		//		((AbstractLIBORCovarianceModelParametric) liborMarketModelCalibrated.getCovarianceModel()).setParameter(param);
		for (double p : param) {
			System.out.println(formatterParam.format(p));
		}

		EulerSchemeFromProcessModel process = new EulerSchemeFromProcessModel(brownianMotionView1);
		net.finmath.montecarlo.interestrate.LIBORMonteCarloSimulationFromLIBORModel simulationCalibrated = new net.finmath.montecarlo.interestrate.LIBORMonteCarloSimulationFromLIBORModel(
				liborMarketModelCalibrated, process);

		System.out.println("\nValuation on calibrated model:");
		double deviationSum			= 0.0;
		double deviationSquaredSum	= 0.0;
		for (int i = 0; i < calibrationProducts.size(); i++) {
			AbstractLIBORMonteCarloProduct calibrationProduct = calibrationProducts.get(i).getProduct();
			try {
				double valueModel = calibrationProduct.getValue(simulationCalibrated);
				double valueTarget = calibrationProducts.get(i).getTargetValue().getAverage();
				double error = valueModel-valueTarget;
				deviationSum += error;
				deviationSquaredSum += error*error;
				System.out.println("Model: " + formatterValue.format(valueModel) + "\t Target: " + formatterValue.format(valueTarget) + "\t Deviation: " + formatterDeviation.format(valueModel-valueTarget) + "\t" + calibrationProduct.toString());
			}
			catch(Exception e) {
				//
			}
		}
		double averageDeviation = deviationSum/calibrationProducts.size();
		System.out.println("Mean Deviation:" + formatterValue.format(averageDeviation));
		System.out.println("RMS Error.....:" + formatterValue.format(Math.sqrt(deviationSquaredSum/calibrationProducts.size())));
		System.out.println("__________________________________________________________________________________________\n");

		long millisEnd = System.currentTimeMillis();

		System.out.println("\t calculation time = " + formatterReal2.format((millisEnd - millisStart)/1000.0) + " sec.");

		Assert.assertTrue(Math.abs(averageDeviation) < 1E-2);
	}

	private static double getParSwaprate(ForwardCurve forwardCurve, DiscountCurve discountCurve, double[] swapTenor) throws CalculationException {
		return net.finmath.marketdata.products.Swap.getForwardSwapRate(new TimeDiscretizationFromArray(swapTenor), new TimeDiscretizationFromArray(swapTenor), forwardCurve, discountCurve);
	}
}
