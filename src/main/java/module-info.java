
module net.finmath.gpu {
	exports net.finmath.cuda.montecarlo;
	exports net.finmath.cuda.montecarlo.alternative;

	requires net.finmath.lib;

	requires java.logging;
	requires org.apache.commons.io;

	requires jcuda;
	requires jcurand;
}
